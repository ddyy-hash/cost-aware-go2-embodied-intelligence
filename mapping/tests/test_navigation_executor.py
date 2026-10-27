from mapping.pc.go2_mapping_dashboard.coarse_map import build_task_graph
from mapping.pc.go2_mapping_dashboard.mission import compile_mission, update_mission_task
from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_bridge import (
    NavigationBridgeError,
    NavigationBridgeStore,
)
from mapping.robot.go2_mapping_nav_bridge.go2_mapping_nav_bridge.navigation_executor import (
    NavigationExecutionAdapter,
    RosNav2ActionClient,
    build_goal_plan,
)

import sys
import types


class _FakeClient:
    def __init__(self, *, wait_for_server=True, accept=True, result=None):
        self.wait_for_server_result = wait_for_server
        self.accept_result = accept
        self.result = result
        self.feedback_count = 0
        self.sent = []
        self.cancelled = False

    def wait_for_server(self, timeout_sec):
        self.timeout_sec = timeout_sec
        return self.wait_for_server_result

    def send_goal(self, goal):
        self.sent.append(goal)
        return self.accept_result

    def cancel_goal(self):
        self.cancelled = True

    def poll_result(self):
        result = self.result
        self.result = None
        return result


def _mission():
    coarse = {
        "schema_version": 1,
        "map_id": "demo",
        "coordinate_frame": "SESSION_ODOM_LOCAL",
        "rooms": [
            {
                "id": "room-1",
                "name": "Room 1",
                "kind": "room",
                "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
                "coverage_percent": 0,
            }
        ],
        "features": [],
    }
    mission = compile_mission(build_task_graph(coarse), mission_id="mission-1")
    task = next(item for item in mission["tasks"] if item["kind"] == "floor_coverage")
    mission = update_mission_task(mission, task["task_id"], lease_id="lease-room-1", mode="AUTONOMOUS", expected_revision=task["revision"])
    task = next(item for item in mission["tasks"] if item["task_id"] == task["task_id"])
    return mission, task


def test_navigation_executor_blocks_when_nav2_is_unavailable(tmp_path):
    mission, task = _mission()
    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    adapter = NavigationExecutionAdapter(store=store, client_factory=lambda plan: _FakeClient(wait_for_server=False))

    result = adapter.execute_once()

    assert result["state"] == "NAV_BLOCKED"
    assert result["state_reason"] == "NAV2_UNAVAILABLE"


def test_navigation_executor_stays_alive_when_no_goal_is_present(tmp_path):
    store = NavigationBridgeStore(tmp_path / "navigation")
    adapter = NavigationExecutionAdapter(store=store, client_factory=lambda plan: _FakeClient())

    result = adapter.execute_once()

    assert result["state"] == "NAV_READY"
    assert result["state_reason"] == "NO_GOAL"
    assert result["goal_available"] is False


def test_navigation_executor_activates_goal_when_nav2_accepts_it(tmp_path):
    mission, task = _mission()
    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    client = _FakeClient(wait_for_server=True, accept=True)
    adapter = NavigationExecutionAdapter(store=store, client_factory=lambda plan: client)

    result = adapter.execute_once()

    assert client.sent
    assert build_goal_plan(result)["state"] == "NAV_ACTIVE"
    assert result["state"] == "NAV_ACTIVE"
    assert result["lease_id"] == "lease-room-1"


def test_navigation_executor_does_not_resend_an_active_goal_on_poll(tmp_path):
    mission, task = _mission()
    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    clients = []

    def factory(_plan):
        client = _FakeClient(wait_for_server=True, accept=True)
        clients.append(client)
        return client

    adapter = NavigationExecutionAdapter(store=store, client_factory=factory)

    first = adapter.execute_once()
    second = adapter.execute_once()

    assert first["state"] == "NAV_ACTIVE"
    assert second["state"] == "NAV_ACTIVE"
    assert len(clients) == 1
    assert len(clients[0].sent) == 1


def test_navigation_executor_records_a_successful_nav2_result_as_done(tmp_path):
    mission, task = _mission()
    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    client = _FakeClient(wait_for_server=True, accept=True, result="SUCCEEDED")
    adapter = NavigationExecutionAdapter(store=store, client_factory=lambda plan: client)

    active = adapter.execute_once()
    completed = adapter.execute_once()

    assert active["state"] == "NAV_ACTIVE"
    assert completed["state"] == "DONE"
    assert completed["state_reason"] == "NAV2_SUCCEEDED"
    assert len(client.sent) == 1


def test_navigation_executor_persists_nav2_feedback_progress(tmp_path):
    mission, task = _mission()
    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    client = _FakeClient(wait_for_server=True, accept=True)
    adapter = NavigationExecutionAdapter(store=store, client_factory=lambda plan: client)

    adapter.execute_once()
    client.feedback_count = 3
    progress = adapter.execute_once()

    assert progress["state"] == "NAV_ACTIVE"
    assert progress["action_feedback_count"] == 3
    assert progress["action_feedback_updated_at_unix"] is not None


def test_navigation_executor_blocks_after_a_nav2_abort_result(tmp_path):
    mission, task = _mission()
    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    client = _FakeClient(wait_for_server=True, accept=True, result="ABORTED")
    adapter = NavigationExecutionAdapter(store=store, client_factory=lambda plan: client)

    adapter.execute_once()
    failed = adapter.execute_once()

    assert failed["state"] == "NAV_BLOCKED"
    assert failed["state_reason"] == "NAV2_ABORTED"
    assert len(client.sent) == 1


def test_navigation_executor_cancels_an_accepted_goal_when_state_activation_fails(tmp_path):
    mission, task = _mission()

    class _ActivationRejectingStore(NavigationBridgeStore):
        def activate(self):
            raise NavigationBridgeError("state activation failed")

    store = _ActivationRejectingStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    client = _FakeClient(wait_for_server=True, accept=True)
    adapter = NavigationExecutionAdapter(store=store, client_factory=lambda plan: client)

    result = adapter.execute_once()

    assert result["state"] == "NAV_BLOCKED"
    assert result["state_reason"] == "state activation failed"
    assert client.cancelled is True


def test_navigation_executor_cleans_up_when_nav2_goal_submission_raises(tmp_path):
    mission, task = _mission()

    class _SendFailClient(_FakeClient):
        def send_goal(self, goal):
            raise RuntimeError("goal serialization failed")

    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    client = _SendFailClient(wait_for_server=True, accept=True)
    adapter = NavigationExecutionAdapter(store=store, client_factory=lambda plan: client)

    result = adapter.execute_once()

    assert result["state"] == "NAV_BLOCKED"
    assert result["state_reason"] == "GOAL_SEND_FAILED"
    assert client.cancelled is True


def test_navigation_executor_cancel_active_uses_the_existing_client(tmp_path):
    mission, task = _mission()
    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    clients = []

    def factory(_plan):
        client = _FakeClient(wait_for_server=True, accept=True)
        clients.append(client)
        return client

    adapter = NavigationExecutionAdapter(store=store, client_factory=factory)
    adapter.execute_once()

    cancelled = adapter.cancel_active()

    assert cancelled["state"] == "CANCELLED"
    assert len(clients) == 1
    assert clients[0].cancelled is True


def test_navigation_executor_resends_only_after_a_paused_goal_is_resumed(tmp_path):
    mission, task = _mission()
    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    clients = []

    def factory(_plan):
        client = _FakeClient(wait_for_server=True, accept=True)
        clients.append(client)
        return client

    adapter = NavigationExecutionAdapter(store=store, client_factory=factory)
    adapter.execute_once()
    store.pause()

    paused = adapter.execute_once()

    assert paused["state"] == "NAV_PAUSED"
    assert clients[0].cancelled is True
    assert len(clients) == 1

    store.resume()
    resumed = adapter.execute_once()

    assert resumed["state"] == "NAV_ACTIVE"
    assert len(clients) == 2
    assert len(clients[1].sent) == 1


def test_ros_nav2_action_client_uses_rclpy_action_client(monkeypatch, tmp_path):
    mission, task = _mission()
    store = NavigationBridgeStore(tmp_path / "navigation")
    goal = store.compile_from_mission(mission, task)
    plan = build_goal_plan(goal)
    nodes = []
    action_clients = []

    class _FakeNode:
        def __init__(self, name):
            self.name = name

        def get_clock(self):
            raise AssertionError("pose construction is not part of this test")

        def destroy_node(self):
            return None

    class _FakeRclpy(types.ModuleType):
        def __init__(self):
            super().__init__("rclpy")
            self.initialized = False
            self.shutdown_calls = 0

        def ok(self):
            return self.initialized

        def init(self, args=None):
            self.initialized = True

        def create_node(self, name):
            node = _FakeNode(name)
            nodes.append(node)
            return node

        def shutdown(self):
            self.shutdown_calls += 1

    class _FakeActionClient:
        def __init__(self, node, action_type, action_name):
            self.node = node
            self.action_type = action_type
            self.action_name = action_name
            action_clients.append(self)

    rclpy_module = _FakeRclpy()
    rclpy_action_module = types.ModuleType("rclpy.action")
    rclpy_action_module.ActionClient = _FakeActionClient
    geometry_module = types.ModuleType("geometry_msgs")
    geometry_msg_module = types.ModuleType("geometry_msgs.msg")
    geometry_msg_module.PoseStamped = object
    nav2_module = types.ModuleType("nav2_msgs")
    nav2_action_module = types.ModuleType("nav2_msgs.action")

    class _NavigateToPose:
        class Goal:
            pass

    class _FollowWaypoints:
        class Goal:
            pass

    nav2_action_module.NavigateToPose = _NavigateToPose
    nav2_action_module.FollowWaypoints = _FollowWaypoints
    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "rclpy.action", rclpy_action_module)
    monkeypatch.setitem(sys.modules, "geometry_msgs", geometry_module)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geometry_msg_module)
    monkeypatch.setitem(sys.modules, "nav2_msgs", nav2_module)
    monkeypatch.setitem(sys.modules, "nav2_msgs.action", nav2_action_module)

    adapter = RosNav2ActionClient.from_plan(plan)

    assert len(nodes) == 1
    assert len(action_clients) == 1
    assert action_clients[0].node is nodes[0]
    assert action_clients[0].action_type is _FollowWaypoints
    assert action_clients[0].action_name == "/follow_waypoints"
    adapter.close()
    assert rclpy_module.shutdown_calls == 1


def test_navigation_executor_preserves_safety_stop_on_poll(tmp_path):
    mission, task = _mission()
    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    client = _FakeClient(wait_for_server=True, accept=True)
    adapter = NavigationExecutionAdapter(store=store, client_factory=lambda plan: client)
    adapter.execute_once()
    store.safety_stop(reason="OBSTACLE")

    result = adapter.execute_once()

    assert result["state"] == "SAFETY_STOP"
    assert result["state_reason"] == "OBSTACLE"
    assert len(client.sent) == 1
    assert client.cancelled is True


def test_navigation_executor_rejects_manual_tasks_and_can_cancel(tmp_path):
    mission, task = _mission()
    mission = update_mission_task(mission, task["task_id"], mode="MANUAL", expected_revision=task["revision"])
    task = next(item for item in mission["tasks"] if item["task_id"] == task["task_id"])
    store = NavigationBridgeStore(tmp_path / "navigation")
    store.compile_from_mission(mission, task)
    client = _FakeClient(wait_for_server=True, accept=True)
    adapter = NavigationExecutionAdapter(store=store, client_factory=lambda plan: client)

    result = adapter.execute_once()
    assert result["state"] == "MANUAL_OVERRIDE"
    assert result["state_reason"] == "MANUAL_TASK"

    cancelled = adapter.cancel_active()
    assert cancelled["state"] == "CANCELLED"
    assert client.cancelled is False
