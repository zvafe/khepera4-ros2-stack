"""ROS 2 node for online keyframe capture, 3DGS training, and SuperSplat export."""

import threading
from typing import Optional, Tuple

import cv2
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
import tf2_ros

from .pipeline_coordinator import ReconstructionPipeline
from .spatial_decimation import SpatialDecimator


class Khepera3DGSNode(Node):
    """ROS 2 Node orchestrating online keyframe collection and on-demand 3DGS training."""

    def __init__(self) -> None:
        """Initialize parameters, subscribers, services, and pipeline components."""
        super().__init__("khepera_3dgs_node")

        # Declare parameters
        self.declare_parameter("output_dir", "output/khepera_3dgs")
        self.declare_parameter("camera_height_m", 0.0283)
        self.declare_parameter("camera_x_offset_m", 0.0682)
        self.declare_parameter("min_translation_m", 0.05)
        self.declare_parameter("min_rotation_deg", 5.0)
        self.declare_parameter("max_linear_vel_mps", 0.35)
        self.declare_parameter("max_angular_vel_radps", 1.00)
        self.declare_parameter("num_iterations", 2000)
        self.declare_parameter("voxel_size_m", 0.015)
        self.declare_parameter("export_ply_name", "khepera_supersplat.ply")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("train_on_shutdown", True)
        self.declare_parameter("enable_floor_scale", False)
        self.declare_parameter("align_supersplat_y_up", True)

        self.output_dir = str(self.get_parameter("output_dir").value)
        self.cam_h = float(self.get_parameter("camera_height_m").value)
        self.cam_xc = float(self.get_parameter("camera_x_offset_m").value)
        self.min_dist = float(self.get_parameter("min_translation_m").value)
        self.min_rot = float(self.get_parameter("min_rotation_deg").value)
        self.max_v = float(self.get_parameter("max_linear_vel_mps").value)
        self.max_w = float(self.get_parameter("max_angular_vel_radps").value)
        self.num_iters = int(self.get_parameter("num_iterations").value)
        self.voxel_size = float(self.get_parameter("voxel_size_m").value)
        self.ply_name = str(self.get_parameter("export_ply_name").value)
        self.img_topic = str(self.get_parameter("image_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.train_on_shutdown = bool(self.get_parameter("train_on_shutdown").value)
        self.enable_floor_scale = bool(self.get_parameter("enable_floor_scale").value)
        self.align_supersplat_y_up = bool(self.get_parameter("align_supersplat_y_up").value)

        # Core pipeline components
        self.decimator = SpatialDecimator(
            min_translation_m=self.min_dist,
            min_rotation_deg=self.min_rot,
            max_linear_vel_mps=self.max_v,
            max_angular_vel_radps=self.max_w,
        )

        self.coordinator = ReconstructionPipeline(
            output_dir=self.output_dir,
            camera_height_m=self.cam_h,
            camera_x_offset_m=self.cam_xc,
            voxel_size_m=self.voxel_size,
            num_iterations=self.num_iters,
            enable_floor_scale=self.enable_floor_scale,
            align_supersplat_y_up=self.align_supersplat_y_up,
        )

        # TF listener for robot map pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, self.img_topic, self._handle_image, 10
        )
        self.depth_sub = None
        if self.depth_topic:
            self.depth_sub = self.create_subscription(
                Image, self.depth_topic, self._handle_depth, 10
            )

        # Services
        self.train_srv = self.create_service(
            Trigger, "/khepera_3dgs/trigger_reconstruction", self._handle_reconstruct
        )
        self.clear_srv = self.create_service(
            Trigger, "/khepera_3dgs/clear", self._handle_clear
        )

        self._lock = threading.Lock()
        self._is_training = False
        self._prev_pose: Optional[Tuple[float, float, float]] = None
        self._prev_time: Optional[float] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._keyframe_depths: list[Optional[np.ndarray]] = []

        self.get_logger().info(
            f"Khepera3DGSNode initialized. Subscribing to {self.img_topic} "
            f"(depth: {self.depth_topic}). "
            f"Gating: Delta_d >= {self.min_dist}m, Delta_theta >= {self.min_rot}deg."
        )

    def _handle_image(self, msg: Image) -> None:
        """Synchronize camera frame with TF pose and evaluate decimation gates."""
        if self._is_training:
            return

        # Lookup transform map -> base_footprint at image timestamp (fallback to latest available)
        t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        transform: Optional[TransformStamped] = None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time.from_msg(msg.header.stamp),
                timeout=rclpy.duration.Duration(seconds=0.03),
            )
        except Exception:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.base_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.03),
                )
            except Exception:
                return

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        q = transform.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))

        # Velocity estimation
        v_lin, v_ang = 0.0, 0.0
        if self._prev_time is not None and (t_sec - self._prev_time) > 1e-4:
            dt = t_sec - self._prev_time
            px, py, pyaw = self._prev_pose
            v_lin = float(np.hypot(tx - px, ty - py) / dt)
            v_ang = float(abs(self.decimator.normalize_angle(yaw - pyaw)) / dt)

        self._prev_pose = (tx, ty, yaw)
        self._prev_time = t_sec

        # Convert Image msg to RGB array
        if msg.encoding in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            img_rgb = arr if msg.encoding == "rgb8" else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            img_rgb = arr[:, :, :3]

        with self._lock:
            kf = self.decimator.add_frame(
                timestamp=t_sec,
                image_rgb=img_rgb,
                x=tx,
                y=ty,
                yaw=yaw,
                linear_vel=v_lin,
                angular_vel=v_ang,
            )
            if kf is not None:
                if self._latest_depth is not None:
                    self._keyframe_depths.append(self._latest_depth.copy())
                else:
                    self._keyframe_depths.append(None)

        if kf is not None:
            self.get_logger().info(
                f"Registered Keyframe #{kf.id}: pose=({tx:.2f}, {ty:.2f}, {yaw:.2f}rad), "
                f"total keyframes: {len(self.decimator.keyframes)}"
            )

    def _handle_depth(self, msg: Image) -> None:
        """Cache incoming sensor depth frames."""
        if self._is_training:
            return
        try:
            if msg.encoding == "32FC1":
                depth = np.frombuffer(
                    msg.data, dtype=np.float32
                ).reshape(msg.height, msg.width).copy()
            elif msg.encoding == "16UC1":
                depth = (
                    np.frombuffer(msg.data, dtype=np.uint16)
                    .reshape(msg.height, msg.width)
                    .astype(np.float32)
                    / 1000.0
                )
            else:
                return
            with self._lock:
                self._latest_depth = depth
        except Exception:
            pass

    def _handle_reconstruct(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Trigger 3DGS training and PLY export on captured keyframes."""
        if self._is_training:
            response.success = False
            response.message = "Reconstruction already in progress."
            return response

        with self._lock:
            kfs = list(self.decimator.keyframes)
            imgs = list(self.decimator.images)
            has_all_depths = (
                len(self._keyframe_depths) == len(kfs)
                and all(d is not None for d in self._keyframe_depths)
            )
            depths = (
                [d.copy() for d in self._keyframe_depths if d is not None]
                if has_all_depths
                else None
            )

        if len(kfs) < 1:
            response.success = False
            response.message = "No keyframes captured. Drive the robot to record viewpoints."
            return response

        # If only 1 keyframe was collected, duplicate it so training pairs can optimize
        if len(kfs) == 1:
            kfs.append(kfs[0])
            imgs.append(imgs[0])
            if depths is not None:
                depths.append(depths[0])

        def _train_worker():
            self._is_training = True
            try:
                self.get_logger().info(f"Starting 3DGS optimization on {len(kfs)} keyframes...")
                ply_path = self.coordinator.run_reconstruction(
                    kfs,
                    imgs,
                    ply_filename=self.ply_name,
                    depth_maps=depths,
                    progress_callback=self.get_logger().info,
                )
                self.get_logger().info(f"Success! SuperSplat PLY exported to {ply_path}")
            except Exception as e:
                self.get_logger().error(f"3DGS Reconstruction failed: {e}")
            finally:
                self._is_training = False

        worker = threading.Thread(target=_train_worker, daemon=True)
        worker.start()

        response.success = True
        response.message = f"3DGS training launched with {len(kfs)} keyframes."
        return response

    def _handle_clear(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Purge collected keyframes."""
        with self._lock:
            self.decimator = SpatialDecimator(
                min_translation_m=self.min_dist,
                min_rotation_deg=self.min_rot,
            )
            self._prev_pose = None
            self._prev_time = None
            self._latest_depth = None
            self._keyframe_depths.clear()

        response.success = True
        response.message = "Keyframe memory cleared."
        return response

    def trigger_training_sync(self) -> Optional[str]:
        """
        Execute synchronous 3DGS training and PLY export on all collected keyframes.

        Invoked automatically on simulation exit/node shutdown if train_on_shutdown is enabled.
        :return: Path to exported PLY, or None if skipped/failed.
        """
        with self._lock:
            kfs = list(self.decimator.keyframes)
            imgs = list(self.decimator.images)

        if len(kfs) < 1:
            self.get_logger().warn(
                "Shutdown: No keyframes collected. Cannot run 3DGS training."
            )
            return None

        if len(kfs) == 1:
            kfs.append(kfs[0])
            imgs.append(imgs[0])

        self._is_training = True
        self.get_logger().info(
            f"Simulation stopping. Auto-triggering 3DGS training on {len(kfs)} keyframes..."
        )
        try:
            ply_path = self.coordinator.run_reconstruction(
                kfs, imgs, ply_filename=self.ply_name
            )
            self.get_logger().info(
                f"✓ Auto-training complete! Model saved to: {ply_path}"
            )
            return ply_path
        except Exception as e:
            self.get_logger().error(f"Auto-training on shutdown failed: {e}")
            return None
        finally:
            self._is_training = False


def main(args=None):
    """Run the Khepera 3DGS reconstruction node."""
    rclpy.init(args=args)
    node = Khepera3DGSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.train_on_shutdown and not node._is_training:
            node.trigger_training_sync()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
