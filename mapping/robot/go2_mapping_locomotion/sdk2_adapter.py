"""Explicitly disarmed Unitree SDK2 transport adapter for Go2.

The adapter is intentionally unusable for actuation by default. It supports
read-only low-state subscription and deterministic conversion of a validated
12-joint target frame, but only an explicit, separately audited activation
path may create the ``rt/lowcmd`` publisher.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .policy_runtime import GO2_JOINT_ORDER, JointTargetFrame


# PGTT's upstream deployment maps SDK motor order into its policy order with
# this permutation. Keep it named and tested instead of hiding it in a loop.
PGTT_TO_UNITREE_MOTOR_INDEX = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)


class SDK2AdapterError(RuntimeError):
    """Raised when the SDK2 transport contract is not satisfied."""


@dataclass(frozen=True)
class SDK2AdapterConfig:
    network_interface: str = "eth0"
    lowstate_topic: str = "rt/lowstate"
    lowcmd_topic: str = "rt/lowcmd"
    control_frequency_hz: int = 50
    armed: bool = False

    def __post_init__(self) -> None:
        if not self.network_interface.strip():
            raise SDK2AdapterError("network_interface is required")
        if self.control_frequency_hz <= 0:
            raise SDK2AdapterError("control frequency must be positive")
        if self.armed:
            raise SDK2AdapterError("SDK2 adapter must be activated through the hardware gate")


class UnitreeSDK2Go2Adapter:
    """Read-only SDK2 adapter until an independently approved actuation path exists."""

    def __init__(self, *, config: SDK2AdapterConfig | None = None) -> None:
        self.config = config or SDK2AdapterConfig()
        self._sdk: Any = None
        self._lowstate_type: Any = None
        self._subscriber: Any = None
        self._publisher: Any = None
        self._latest_lowstate: Any = None
        self._closed = False

    def initialize_readonly(self) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        except ImportError as exc:
            raise SDK2AdapterError("unitree_sdk2py is unavailable") from exc
        try:
            ChannelFactoryInitialize(0, self.config.network_interface)
            self._subscriber = ChannelSubscriber(self.config.lowstate_topic, LowState_)
            self._subscriber.Init(self._on_lowstate, 10)
            self._lowstate_type = LowState_
        except Exception as exc:
            raise SDK2AdapterError("SDK2 read-only initialization failed") from exc

    def _on_lowstate(self, message: Any) -> None:
        if self._closed:
            return
        self._latest_lowstate = message

    @property
    def latest_lowstate(self) -> Any:
        return self._latest_lowstate

    def measured_policy_joint_state(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        state = self._latest_lowstate
        if state is None:
            raise SDK2AdapterError("no lowstate sample is available")
        try:
            positions = [float(state.motor_state[index].q) for index in PGTT_TO_UNITREE_MOTOR_INDEX]
            velocities = [float(state.motor_state[index].dq) for index in PGTT_TO_UNITREE_MOTOR_INDEX]
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise SDK2AdapterError("lowstate motor layout is incompatible with Go2") from exc
        return tuple(positions), tuple(velocities)

    def build_unarmed_lowcmd(self, frame: JointTargetFrame, *, kp: float = 40.0, kd: float = 3.0) -> Any:
        """Build a message for offline inspection; never publishes it."""
        if self.config.armed or frame.transport_authorized:
            raise SDK2AdapterError("unarmed adapter received an actuation-authorized frame")
        if frame.joint_order != GO2_JOINT_ORDER or len(frame.positions_rad) != 12:
            raise SDK2AdapterError("joint target frame does not use the Go2 policy contract")
        try:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as exc:
            raise SDK2AdapterError("unitree_sdk2py lowcmd types are unavailable") from exc
        if not all(isinstance(value, (int, float)) and value == value and abs(value) != float("inf") for value in frame.positions_rad):
            raise SDK2AdapterError("joint target frame contains non-finite values")
        command = unitree_go_msg_dds__LowCmd_()
        for policy_index, motor_index in enumerate(PGTT_TO_UNITREE_MOTOR_INDEX):
            motor = command.motor_cmd[motor_index]
            motor.mode = 0x01
            motor.q = float(frame.positions_rad[policy_index])
            motor.kp = float(kp)
            motor.dq = 0.0
            motor.kd = float(kd)
            motor.tau = 0.0
        command.crc = CRC().Crc(command)
        return command

    def close(self) -> None:
        self._closed = True
        # SDK2's callback reader joins its own callback thread from Close(),
        # which can block indefinitely when DDS is still receiving rt/lowstate.
        # Keep the reader alive but inert; the owning process boundary releases
        # DDS resources on exit. No publisher is ever created by this adapter.
        self._publisher = None


__all__ = [
    "PGTT_TO_UNITREE_MOTOR_INDEX",
    "SDK2AdapterConfig",
    "SDK2AdapterError",
    "UnitreeSDK2Go2Adapter",
]
