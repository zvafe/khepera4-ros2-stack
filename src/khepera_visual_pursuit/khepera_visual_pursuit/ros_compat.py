# Copyright 2026 Andrei Berteanu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ROS 2 compatibility layer providing fallback definitions when running without ROS 2."""

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from geometry_msgs.msg import Point32, Polygon, PolygonStamped, Twist, Vector3
    from rcl_interfaces.msg import SetParametersResult
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import Image, LaserScan
    from std_msgs.msg import Bool, Float32, Header, String
    from vision_msgs.msg import (
        BoundingBox2D,
        Detection2D,
        Detection2DArray,
        ObjectHypothesis,
        ObjectHypothesisWithPose,
        Pose2D,
    )

    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.debug("ROS 2 libraries not found. Operating with compatibility classes.")

    class SetParametersResult:
        """Fallback for rcl_interfaces.msg.SetParametersResult."""

        def __init__(self, successful: bool = True, reason: str = "") -> None:
            """Initialize SetParametersResult."""
            self.successful = successful
            self.reason = reason

    class Parameter:
        """Fallback for rclpy.parameter.Parameter."""

        def __init__(self, name: str = "", value: Any = None) -> None:
            """Initialize Parameter."""
            self.name = name
            self.value = value

    class ExternalShutdownException(Exception):
        """Fallback for rclpy.executors.ExternalShutdownException."""

        pass

    class Header:
        """Stand-in for std_msgs.msg.Header."""

        def __init__(self, frame_id: str = "") -> None:
            """Initialize Header."""
            self.stamp = Time()
            self.frame_id = frame_id

    class Time:
        """Stand-in for builtin_interfaces.msg.Time."""

        def __init__(self, sec: int = 0, nanosec: int = 0) -> None:
            """Initialize Time."""
            self.sec = sec
            self.nanosec = nanosec

        def to_msg(self) -> "Time":
            """Return self as message representation."""
            return self

    class Image:
        """Stand-in for sensor_msgs.msg.Image."""

        def __init__(self) -> None:
            """Initialize Image message."""
            self.header = Header()
            self.height: int = 0
            self.width: int = 0
            self.encoding: str = "bgr8"
            self.is_bigendian: int = 0
            self.step: int = 0
            self.data: bytes = b""

    class Bool:
        """Stand-in for std_msgs.msg.Bool."""

        def __init__(self, data: bool = False) -> None:
            """Initialize Bool message."""
            self.data = bool(data)

    class Float32:
        """Stand-in for std_msgs.msg.Float32."""

        def __init__(self, data: float = 0.0) -> None:
            """Initialize Float32 message."""
            self.data = float(data)

    class String:
        """Stand-in for std_msgs.msg.String."""

        def __init__(self, data: str = "") -> None:
            """Initialize String message."""
            self.data = str(data)

    class Point32:
        """Stand-in for geometry_msgs.msg.Point32."""

        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
            """Initialize Point32 message."""
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class Polygon:
        """Stand-in for geometry_msgs.msg.Polygon."""

        def __init__(self) -> None:
            """Initialize Polygon message."""
            self.points: List[Point32] = []

    class PolygonStamped:
        """Stand-in for geometry_msgs.msg.PolygonStamped."""

        def __init__(self) -> None:
            """Initialize PolygonStamped message."""
            self.header = Header()
            self.polygon = Polygon()

    class Vector3:
        """Stand-in for geometry_msgs.msg.Vector3."""

        def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
            """Initialize Vector3 message."""
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class Twist:
        """Stand-in for geometry_msgs.msg.Twist."""

        def __init__(self) -> None:
            """Initialize Twist message."""
            self.linear = Vector3()
            self.angular = Vector3()

    class LaserScan:
        """Stand-in for sensor_msgs.msg.LaserScan."""

        def __init__(self) -> None:
            """Initialize LaserScan message."""
            self.header = Header()
            self.angle_min: float = -3.14159
            self.angle_max: float = 3.14159
            self.angle_increment: float = 0.01745
            self.range_min: float = 0.02
            self.range_max: float = 2.0
            self.ranges: List[float] = []

    class Pose2D:
        """Stand-in for vision_msgs/geometry_msgs Pose2D."""

        def __init__(self) -> None:
            """Initialize Pose2D."""
            self.position = Position2D()
            self.theta = 0.0

    class Position2D:
        """Stand-in for position representation in 2D."""

        def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
            """Initialize Position2D."""
            self.x = float(x)
            self.y = float(y)

    class BoundingBox2D:
        """Stand-in for vision_msgs.msg.BoundingBox2D."""

        def __init__(self) -> None:
            """Initialize BoundingBox2D."""
            self.center = Pose2D()
            self.size_x: float = 0.0
            self.size_y: float = 0.0

    class ObjectHypothesis:
        """Stand-in for vision_msgs.msg.ObjectHypothesis."""

        def __init__(self, class_id: str = "", score: float = 0.0) -> None:
            """Initialize ObjectHypothesis."""
            self.class_id = class_id
            self.score = float(score)

    class ObjectHypothesisWithPose:
        """Stand-in for vision_msgs.msg.ObjectHypothesisWithPose."""

        def __init__(self) -> None:
            """Initialize ObjectHypothesisWithPose."""
            self.hypothesis = ObjectHypothesis()

    class Detection2D:
        """Stand-in for vision_msgs.msg.Detection2D."""

        def __init__(self) -> None:
            """Initialize Detection2D."""
            self.header = Header()
            self.results: List[ObjectHypothesisWithPose] = []
            self.bbox = BoundingBox2D()

    class Detection2DArray:
        """Stand-in for vision_msgs.msg.Detection2DArray."""

        def __init__(self) -> None:
            """Initialize Detection2DArray."""
            self.header = Header()
            self.detections: List[Detection2D] = []

    class MockClock:
        """Stand-in for rclpy clock."""

        def now(self) -> Time:
            """Return current simulated time."""
            return Time(sec=0, nanosec=0)

    class Parameter:
        """Stand-in for rclpy.parameter.Parameter."""

        def __init__(self, name: str, value: Any) -> None:
            """Initialize Parameter."""
            self.name = name
            self.value = value

    class Publisher:
        """Stand-in for rclpy Publisher."""

        def __init__(self, topic: str, msg_type: type) -> None:
            """Initialize Publisher."""
            self.topic = topic
            self.msg_type = msg_type
            self.published_messages: List[Any] = []

        def publish(self, msg: Any) -> None:
            """Record published message."""
            self.published_messages.append(msg)

    class Subscription:
        """Stand-in for rclpy Subscription."""

        def __init__(self, topic: str, msg_type: type, callback: Callable[[Any], None]) -> None:
            """Initialize Subscription."""
            self.topic = topic
            self.msg_type = msg_type
            self.callback = callback

    class Timer:
        """Stand-in for rclpy Timer."""

        def __init__(self, period_sec: float, callback: Callable[[], None]) -> None:
            """Initialize Timer."""
            self.period_sec = period_sec
            self.callback = callback
            self.is_canceled = False

        def cancel(self) -> None:
            """Cancel timer."""
            self.is_canceled = True

        def trigger(self) -> None:
            """Manually fire callback for testing."""
            if not self.is_canceled:
                self.callback()

    class Node:
        """Stand-in for rclpy.node.Node."""

        def __init__(self, node_name: str, **kwargs: Any) -> None:
            """Initialize compatibility Node."""
            self._node_name = node_name
            self._parameters: Dict[str, Any] = {}
            self._publishers: List[Publisher] = []
            self._subscriptions: List[Subscription] = []
            self._timers: List[Timer] = []
            self._clock = MockClock()
            self._is_destroyed = False

        def get_name(self) -> str:
            """Return node name."""
            return self._node_name

        def get_clock(self) -> MockClock:
            """Return node clock."""
            return self._clock

        def declare_parameter(self, name: str, value: Any) -> Parameter:
            """Declare parameter with default value."""
            self._parameters[name] = value
            return Parameter(name, value)

        def get_parameter(self, name: str) -> Parameter:
            """Retrieve parameter."""
            value = self._parameters.get(name)
            return Parameter(name, value)

        def set_parameter(self, name: str, value: Any) -> None:
            """Set parameter value."""
            self._parameters[name] = value

        def create_publisher(
            self, msg_type: type, topic: str, qos_profile: Any = 10
        ) -> Publisher:
            """Create stand-in publisher."""
            pub = Publisher(topic, msg_type)
            self._publishers.append(pub)
            return pub

        def create_subscription(
            self,
            msg_type: type,
            topic: str,
            callback: Callable[[Any], None],
            *args: Any,
            **kwargs: Any,
        ) -> Subscription:
            """Create stand-in subscription with <= 3 positional arguments."""
            sub = Subscription(topic, msg_type, callback)
            self._subscriptions.append(sub)
            return sub

        def create_timer(self, timer_period_sec: float, callback: Callable[[], None]) -> Timer:
            """Create stand-in timer."""
            timer = Timer(timer_period_sec, callback)
            self._timers.append(timer)
            return timer

        def destroy_node(self) -> None:
            """Clean up node resources."""
            self._is_destroyed = True

        def get_logger(self) -> logging.Logger:
            """Return logger instance."""
            return logger

    class MockRclpy:
        """Stand-in for rclpy module."""

        @staticmethod
        def init(args: Optional[List[str]] = None) -> None:
            """Initialize ROS 2 mock context."""
            pass

        @staticmethod
        def shutdown() -> None:
            """Shutdown ROS 2 mock context."""
            pass

        @staticmethod
        def ok() -> bool:
            """Return execution status."""
            return True

        @staticmethod
        def spin(node: Node) -> None:
            """Spin node once."""
            pass

        @staticmethod
        def spin_once(node: Node, timeout_sec: float = 0.0) -> None:
            """Execute pending timers or callbacks."""
            pass

    rclpy = MockRclpy()
