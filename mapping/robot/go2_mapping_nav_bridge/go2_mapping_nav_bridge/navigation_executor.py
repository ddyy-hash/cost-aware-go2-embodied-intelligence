"""Executable Nav2 bridge runtime for mission-derived navigation goals."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Protocol

from .navigation_bridge import NavigationBridgeError, NavigationBridgeStore


class NavigationExecutionError(RuntimeError):
    pass


class Nav2GoalClient(Protocol):
    def wait_for_server(self, timeout_sec: float) -> bool: ...

    def send_goal(self, goal: Mapping[str, Any]) -> bool: ...

    def poll_result(self) -> str | None: ...

    def cancel_goal(self) -> None: ...


def _yaw_to_quaternion(yaw: float) -> dict[str, float]:
    half = float(yaw) / 2.0
    return {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)}


def _goal_type(record: Mapping[str, Any]) -> str:
    waypoints = list(record.get("waypoints") or [])
    if len(waypoints) > 1:
        return "FOLLOW_WAYPOINTS"
    return "NAVIGATE_TO_POSE"


def _first_point(record: Mapping[str, Any]) -> dict[str, float]:
    for key in ("waypoints", "inspection_targets"):
        points = record.get(key) or []
        if points:
            point = points[0]
            if isinstance(point, Mapping) and {"x", "y"}.issubset(point):
                return {
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "yaw": float(point.get("yaw", 0.0)),
                    "frame": str(point.get("frame") or "SESSION_ODOM_LOCAL"),
                }
    geofence = record.get("geofence") or {}
    polygon = list(geofence.get("polygon") or [])
    if polygon:
        xs = [float(point["x"]) for point in polygon]
        ys = [float(point["y"]) for point in polygon]
        centroid = {
            "x": sum(xs) / len(xs),
            "y": sum(ys) / len(ys),
            "yaw": 0.0,
            "frame": str(geofence.get("frame") or "SESSION_ODOM_LOCAL"),
        }
        return centroid
    raise NavigationExecutionError("navigation goal does not contain any target geometry")


def build_goal_plan(record: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise NavigationExecutionError("navigation record must be a mapping")
    mission_id = record.get("mission_id")
    task_id = record.get("task_id")
    goal_type = _goal_type(record)
    if not mission_id or not task_id:
        raise NavigationExecutionError("navigation record is missing mission or task identity")
    plan = {
        "mission_id": str(mission_id),
        "task_id": str(task_id),
        "goal_id": str(record.get("goal_id") or f"{mission_id}:{task_id}"),
        "goal_type": goal_type,
        "mode": str(record.get("mode") or "ASSISTED"),
        "state": str(record.get("state") or "NAV_READY"),
        "state_reason": record.get("state_reason"),
        "task_status": str(record.get("task_status") or "READY"),
        "task_revision": int(record.get("task_revision", 1)),
        "lease_id": record.get("lease_id"),
        "lease_state": str(record.get("lease_state") or "UNLEASED"),
        "max_speed_mps": float(record.get("max_speed_mps", 0.0)),
        "max_duration_s": int(record.get("max_duration_s", 0)),
        "goal_available": record.get("goal_available") is True,
        "operator_confirmation_required": record.get("operator_confirmation_required") is True,
        "safety_revision": record.get("safety_revision"),
        "mission_source_sha256": record.get("mission_source_sha256"),
        "target": _first_point(record),
        "waypoints": list(record.get("waypoints") or []),
        "inspection_targets": list(record.get("inspection_targets") or []),
        "geofence": record.get("geofence") or {"frame": "SESSION_ODOM_LOCAL", "polygon": []},
    }
    if plan["mode"] == "AUTONOMOUS" and not plan["lease_id"]:
        plan["state"] = "NAV_BLOCKED"
        plan["state_reason"] = "LEASE_REQUIRED"
        plan["goal_available"] = False
    if plan["mode"] == "MANUAL":
        plan["state"] = "MANUAL_OVERRIDE"
        plan["state_reason"] = "MANUAL_TASK"
        plan["goal_available"] = False
    return plan


class RosNav2ActionClient:
    """Optional ROS 2 action client wrapper for a single Nav2 goal stream."""

    def __init__(
        self,
        *,
        node: Any,
        client: Any,
        goal_type: Any,
        pose_stamped_type: Any,
        timeout_sec: float = 8.0,
    ) -> None:
        self._node = node
        self._client = client
        self._goal_type = goal_type
        self._pose_stamped_type = pose_stamped_type
        self._timeout_sec = timeout_sec
        self._goal_handle = None
        self._result_future = None
        self._feedback_count = 0

    @classmethod
    def from_plan(
        cls,
        plan: Mapping[str, Any],
        *,
        node_name: str = "go2_mapping_nav_bridge",
        navigate_to_pose_action_name: str = "/navigate_to_pose",
        follow_waypoints_action_name: str = "/follow_waypoints",
        timeout_sec: float = 8.0,
    ) -> "RosNav2ActionClient":
        try:
            import rclpy
            from rclpy.action import ActionClient
            from geometry_msgs.msg import PoseStamped
            from nav2_msgs.action import FollowWaypoints, NavigateToPose
        except ImportError as exc:
            raise NavigationExecutionError("ROS Nav2 runtime is unavailable in this environment") from exc

        if hasattr(rclpy, "ok") and not rclpy.ok():
            rclpy.init(args=None)
        elif not hasattr(rclpy, "ok"):
            rclpy.init(args=None)
        node = rclpy.create_node(node_name)
        goal_type = FollowWaypoints if plan.get("goal_type") == "FOLLOW_WAYPOINTS" else NavigateToPose
        client = ActionClient(
            node,
            goal_type,
            follow_waypoints_action_name if goal_type is FollowWaypoints else navigate_to_pose_action_name,
        )
        adapter = cls(
            node=node,
            client=client,
            goal_type=goal_type,
            pose_stamped_type=PoseStamped,
            timeout_sec=timeout_sec,
        )
        adapter._rclpy = rclpy  # type: ignore[attr-defined]
        return adapter

    def _pose_stamped(self, point: Mapping[str, Any]):
        pose = self._pose_stamped_type()
        pose.header.frame_id = str(point.get("frame") or "SESSION_ODOM_LOCAL")
        pose.header.stamp = self._node.get_clock().now().to_msg()
        pose.pose.position.x = float(point.get("x", 0.0))
        pose.pose.position.y = float(point.get("y", 0.0))
        pose.pose.position.z = 0.0
        q = _yaw_to_quaternion(float(point.get("yaw", 0.0)))
        pose.pose.orientation.x = q["x"]
        pose.pose.orientation.y = q["y"]
        pose.pose.orientation.z = q["z"]
        pose.pose.orientation.w = q["w"]
        return pose

    def wait_for_server(self, timeout_sec: float) -> bool:
        return bool(self._client.wait_for_server(timeout_sec=timeout_sec))

    def send_goal(self, goal: Mapping[str, Any]) -> bool:
        if goal.get("goal_type") == "FOLLOW_WAYPOINTS":
            ros_goal = self._goal_type.Goal()
            ros_goal.poses = [self._pose_stamped(point) for point in goal.get("waypoints") or []]
            if not ros_goal.poses:
                ros_goal.poses = [self._pose_stamped(goal["target"])]
        else:
            ros_goal = self._goal_type.Goal()
            ros_goal.pose = self._pose_stamped(goal["target"])
            if hasattr(ros_goal, "behavior_tree"):
                ros_goal.behavior_tree = ""
        future = self._client.send_goal_async(ros_goal, feedback_callback=self._feedback_callback)
        rclpy = getattr(self, "_rclpy", None)
        if rclpy is not None:
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=self._timeout_sec)
        goal_handle = future.result()
        self._goal_handle = goal_handle if getattr(goal_handle, "accepted", False) else None
        if self._goal_handle is not None:
            self._result_future = self._goal_handle.get_result_async()
        return bool(getattr(goal_handle, "accepted", False))

    def _feedback_callback(self, _feedback_message: Any) -> None:
        self._feedback_count += 1

    def poll_result(self) -> str | None:
        result_future = self._result_future
        if result_future is None:
            return None
        rclpy = getattr(self, "_rclpy", None)
        if rclpy is not None:
            rclpy.spin_once(self._node, timeout_sec=0.0)
        if not result_future.done():
            return None
        response = result_future.result()
        status = getattr(response, "status", None)
        status_names = {
            4: "SUCCEEDED",
            5: "CANCELED",
            6: "ABORTED",
        }
        self._result_future = None
        self._goal_handle = None
        return status_names.get(int(status), "FAILED")

    def cancel_goal(self) -> None:
        if self._goal_handle is None:
            return
        future = self._goal_handle.cancel_goal_async()
        rclpy = getattr(self, "_rclpy", None)
        if rclpy is not None:
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=self._timeout_sec)
        self._goal_handle = None
        self._result_future = None

    def close(self) -> None:
        if hasattr(self._node, "destroy_node"):
            self._node.destroy_node()
        rclpy = getattr(self, "_rclpy", None)
        if rclpy is not None and hasattr(rclpy, "shutdown"):
            try:
                rclpy.shutdown()
            except Exception:
                pass


@dataclass
class NavigationExecutionAdapter:
    store: NavigationBridgeStore
    client_factory: Callable[[dict[str, Any]], Nav2GoalClient]
    timeout_sec: float = 8.0
    _client: Nav2GoalClient | None = field(default=None, init=False, repr=False)

    def _cancel_cached_client(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            client.cancel_goal()
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def execute_once(self) -> dict[str, Any]:
        current = self.store.load()
        current_state = str(current.get("state") or "NAV_READY")
        if current_state == "NAV_ACTIVE" and self._client is not None:
            poll_result = getattr(self._client, "poll_result", None)
            result = poll_result() if callable(poll_result) else None
            if result == "SUCCEEDED":
                self._cancel_cached_client()
                return self.store.done(reason="NAV2_SUCCEEDED")
            if result in {"ABORTED", "FAILED"}:
                self._cancel_cached_client()
                return self.store.block(reason=f"NAV2_{result}")
            if result == "CANCELED":
                self._cancel_cached_client()
                return self.store.block(reason="NAV2_CANCELED")
            feedback_count = getattr(self._client, "feedback_count", None)
            if isinstance(feedback_count, int) and not isinstance(feedback_count, bool):
                previous_count = int(current.get("action_feedback_count", 0))
                if feedback_count > previous_count:
                    return self.store.record_feedback(feedback_count)
            return current
        if current_state == "NAV_PAUSED":
            self._cancel_cached_client()
            return current
        if current_state == "SAFETY_STOP":
            self._cancel_cached_client()
            return current
        if current_state == "MANUAL_OVERRIDE":
            self._cancel_cached_client()
            return current
        if current_state in {"LOCALIZATION_REQUIRED", "CANCELLED", "DONE", "QUARANTINED"}:
            self._cancel_cached_client()
            return current
        # Stay idle until the Dashboard publishes a goal.
        if not current.get("mission_id") and current.get("goal_available") is not True:
            return current
        plan = build_goal_plan(current)
        if plan["state"] == "NAV_BLOCKED":
            return self.store.block(reason=str(plan["state_reason"] or "NAV_BLOCKED"))
        if plan["mode"] == "MANUAL":
            return self.store.manual_override(reason="MANUAL_TASK")
        if not plan["goal_available"]:
            return self.store.block(reason=str(plan["state_reason"] or "GOAL_UNAVAILABLE"))
        try:
            client = self.client_factory(plan)
        except NavigationExecutionError as exc:
            return self.store.block(reason=str(exc))
        if not client.wait_for_server(self.timeout_sec):
            close = getattr(client, "close", None)
            if callable(close):
                close()
            return self.store.block(reason="NAV2_UNAVAILABLE")
        try:
            accepted = client.send_goal(plan)
        except Exception:
            try:
                client.cancel_goal()
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            return self.store.block(reason="GOAL_SEND_FAILED")
        if not accepted:
            close = getattr(client, "close", None)
            if callable(close):
                close()
            return self.store.block(reason="GOAL_REJECTED")
        try:
            result = self.store.activate()
            self._client = client
            return result
        except NavigationBridgeError as exc:
            try:
                client.cancel_goal()
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            return self.store.block(reason=str(exc))

    def cancel_active(self) -> dict[str, Any]:
        try:
            self._cancel_cached_client()
        except Exception:
            self._client = None
        return self.store.cancel()


def run_navigation_executor(
    navigation_root: str | Path,
    *,
    poll_seconds: float = 1.0,
    timeout_seconds: float = 8.0,
    once: bool = False,
) -> int:
    store = NavigationBridgeStore(navigation_root)
    adapter = NavigationExecutionAdapter(
        store=store,
        client_factory=lambda plan: RosNav2ActionClient.from_plan(plan, timeout_sec=timeout_seconds),
        timeout_sec=timeout_seconds,
    )
    if once:
        result = adapter.execute_once()
        print(json.dumps(result, sort_keys=True))
        return 0
    while True:
        result = adapter.execute_once()
        print(json.dumps(result, sort_keys=True))
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--navigation-root",
        type=Path,
        default=Path("/home/unitree/go2_mapping/runtime/navigation"),
        help="Path to the navigation bridge state directory.",
    )
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be finite and positive")
    try:
        return run_navigation_executor(
            args.navigation_root,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
            once=args.once,
        )
    except NavigationExecutionError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


def _identity_client_factory(_plan: dict[str, Any]) -> Nav2GoalClient:
    raise NavigationExecutionError("ROS Nav2 runtime is unavailable in this environment")


def create_execution_adapter(store: NavigationBridgeStore, client_factory: Callable[[dict[str, Any]], Nav2GoalClient] | None = None) -> NavigationExecutionAdapter:
    return NavigationExecutionAdapter(store=store, client_factory=client_factory or _identity_client_factory)
