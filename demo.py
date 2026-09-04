"""
demo.py — Interactive RViz Demo for PPO LiDAR Navigation
=========================================================

Click any point on the map in RViz using the "Nav2 Goal" button.
The robot will navigate there using the trained PPO policy.

Usage:
    python demo.py

Prerequisites:
    Simulation stack must be running (Gazebo + SLAM + Nav2 + RViz).
    See demo.sh for the one-command launcher.
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import nav_msgs.msg as nav_msgs
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
import tf2_ros
from rosgraph_msgs.msg import Clock

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "ppo"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "urdf_test" / "src" / "scripts"))

from actor import Actor

# ─── Constants ────────────────────────────────────────────────────────────────
CHECKPOINT    = PROJECT_ROOT / "checkpoints" / "hybrid_phase5" / "best_success.pt"
GOAL_RADIUS   = 0.30        # m — print "GOAL REACHED!" when within this distance
CONTROL_HZ    = 10          # control loop frequency
NUM_LIDAR_RAYS   = 360
NUM_SCAN_SECTORS = 36
MAX_LINEAR_VEL   = 0.33     # m/s
MIN_LINEAR_VEL   = -0.22    # m/s
MAX_ANGULAR_VEL  = 2.84     # rad/s
MAX_SCAN_RANGE   = 12.0


class DemoNode(Node):
    """Thin ROS2 node for interactive PPO demo."""

    def __init__(self):
        super().__init__("ppo_demo_node")

        # ── Publishers ────────────────────────────────────────────────────────
        self.vel_pub    = self.create_publisher(Twist,       "/cmd_vel_nav",  10)
        self.goal_pub   = self.create_publisher(PoseStamped, "/goal_pose",    10)
        marker_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(Marker,      "/goal_marker",  marker_qos)

        # ── State ─────────────────────────────────────────────────────────────
        self.laser_ranges    = np.ones(NUM_LIDAR_RAYS, dtype=np.float32) * MAX_SCAN_RANGE
        self.robot_pos       = np.array([0.0, 0.0])
        self.robot_yaw       = 0.0
        self.current_linear  = 0.0
        self.current_angular = 0.0
        self.sim_time        = 0.0
        self.current_path    = None
        self.path_recv_time  = 0.0
        self.target_position = None        # set when user clicks in RViz
        self.goal_reached    = True        # start idle until first click

        # ── TF ────────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(LaserScan,   "/scan",       self._scan_cb,   10)
        self.create_subscription(Odometry,    "/odom",       self._odom_cb,   10)
        self.create_subscription(Clock,       "/clock",      self._clock_cb,  qos_profile_sensor_data)
        self.create_subscription(nav_msgs.Path,        "/plan",       self._path_cb,   10)
        # This is the topic RViz's "Nav2 Goal" button publishes to
        self.create_subscription(PoseStamped, "/goal_pose",  self._goal_cb,   10)

        self.get_logger().info("✅ PPO Demo Node ready. Click 'Nav2 Goal' in RViz to navigate!")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _scan_cb(self, msg):
        indices = np.linspace(0, len(msg.ranges) - 1, NUM_LIDAR_RAYS, dtype=int)
        r = np.array(msg.ranges)[indices]
        r[np.isinf(r)] = msg.range_max
        r[np.isnan(r)] = msg.range_max
        r[r < 0.08]    = msg.range_max
        self.laser_ranges = r.astype(np.float32)

    def _odom_cb(self, msg):
        self.current_linear  = msg.twist.twist.linear.x
        self.current_angular = msg.twist.twist.angular.z

    def _clock_cb(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _path_cb(self, msg):
        self.current_path   = msg
        self.path_recv_time = self.sim_time

    def _goal_cb(self, msg):
        """Called when user clicks 'Nav2 Goal' in RViz."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_position = np.array([x, y])
        self.goal_reached    = False
        self.get_logger().info(f"🎯 New goal received: ({x:.2f}, {y:.2f})")
        # Re-publish so Nav2 also gets it and re-plans
        self.goal_pub.publish(msg)
        self._publish_goal_marker(x, y)

    # ── TF Pose ───────────────────────────────────────────────────────────────

    def _update_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                "map", "base_footprint",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.05)
            )
            self.robot_pos = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y
            ])
            q = tf.transform.rotation
            self.robot_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
        except Exception:
            pass

    # ── Observation ───────────────────────────────────────────────────────────

    def get_obs(self):
        # 1. LiDAR sectors (36D)
        rays_per_sector = NUM_LIDAR_RAYS // NUM_SCAN_SECTORS
        scan = np.clip(self.laser_ranges, 0.0, MAX_SCAN_RANGE) / MAX_SCAN_RANGE
        scan[~np.isfinite(scan)] = 1.0
        sectors = scan.reshape(NUM_SCAN_SECTORS, rays_per_sector)
        norm_scans = np.min(sectors, axis=1).astype(np.float32)

        # 2. Goal / waypoint (2D)
        waypoint = self.target_position
        if self.current_path is not None and len(self.current_path.poses) > 0:
            path_age = self.sim_time - self.path_recv_time
            if path_age <= 2.0:
                for pose_s in self.current_path.poses:
                    p = np.array([pose_s.pose.position.x, pose_s.pose.position.y])
                    if np.linalg.norm(p - self.robot_pos) >= 0.5:
                        waypoint = p
                        break
                else:
                    lp = self.current_path.poses[-1].pose.position
                    waypoint = np.array([lp.x, lp.y])

        goal_vec  = waypoint - self.robot_pos
        D_wp      = np.linalg.norm(goal_vec)
        heading   = math.atan2(goal_vec[1], goal_vec[0]) - self.robot_yaw
        heading   = math.atan2(math.sin(heading), math.cos(heading))
        norm_d    = float(np.clip(D_wp / 8.0, 0.0, 1.0))
        norm_h    = float(heading / math.pi)

        # 3. Velocity (2D)
        norm_lin = float(np.clip(self.current_linear  / MAX_LINEAR_VEL,  -1.0, 1.0))
        norm_ang = float(np.clip(self.current_angular / MAX_ANGULAR_VEL, -1.0, 1.0))

        # 4. Nav2 path features (10D)
        path_features = np.zeros(10, dtype=np.float32)
        if self.current_path is not None and len(self.current_path.poses) > 0:
            path_age = self.sim_time - self.path_recv_time
            if path_age <= 2.0:
                try:
                    pts = [(p.pose.position.x, p.pose.position.y)
                           for p in self.current_path.poses]
                    dists = [0.0]
                    for i in range(1, len(pts)):
                        d = math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
                        dists.append(dists[-1] + d)
                    targets = [0.5, 1.0, 1.5, 2.0, 2.5]
                    feat_pts = []
                    for t in targets:
                        if t >= dists[-1]:
                            feat_pts.append(pts[-1])
                        else:
                            for i in range(1, len(dists)):
                                if dists[i] >= t:
                                    r = (t - dists[i-1]) / max(dists[i]-dists[i-1], 1e-5)
                                    x = pts[i-1][0] + r*(pts[i][0]-pts[i-1][0])
                                    y = pts[i-1][1] + r*(pts[i][1]-pts[i-1][1])
                                    feat_pts.append((x, y))
                                    break
                    for idx, (px, py) in enumerate(feat_pts[:5]):
                        rx = (px - self.robot_pos[0]) * math.cos(-self.robot_yaw) \
                           - (py - self.robot_pos[1]) * math.sin(-self.robot_yaw)
                        ry = (px - self.robot_pos[0]) * math.sin(-self.robot_yaw) \
                           + (py - self.robot_pos[1]) * math.cos(-self.robot_yaw)
                        path_features[idx*2]   = float(np.clip(rx / 3.0, -1.0, 1.0))
                        path_features[idx*2+1] = float(np.clip(ry / 3.0, -1.0, 1.0))
                except Exception:
                    pass

        obs = np.concatenate([
            norm_scans,
            [norm_d, norm_h],
            [norm_lin, norm_ang],
            path_features
        ]).astype(np.float32)
        return obs

    # ── Action ────────────────────────────────────────────────────────────────

    def publish_velocity(self, action):
        lin = float(((np.clip(action[0], -1, 1) + 1.0) / 2.0)
                    * (MAX_LINEAR_VEL - MIN_LINEAR_VEL) + MIN_LINEAR_VEL)
        ang = float(np.clip(action[1], -1, 1) * MAX_ANGULAR_VEL)
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.vel_pub.publish(cmd)
        return lin, ang

    def stop(self):
        self.vel_pub.publish(Twist())

    # ── Marker ────────────────────────────────────────────────────────────────

    def _publish_goal_marker(self, x, y):
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns, m.id        = "goal", 0
        m.type            = Marker.SPHERE
        m.action          = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.2
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.5
        m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.4; m.color.a = 1.0
        self.marker_pub.publish(m)


def main():
    print("=" * 60)
    print("  PPO LiDAR Navigation — Interactive Demo")
    print("=" * 60)
    print(f"  Checkpoint : checkpoints/hybrid_phase5/best_success.pt")
    print(f"  Goal radius: {GOAL_RADIUS} m")
    print()
    print("  HOW TO USE:")
    print("  1. Open RViz (should open automatically with demo.sh)")
    print("  2. Click the '2D Goal Pose' / 'Nav2 Goal' button in RViz toolbar")
    print("  3. Click anywhere on the map")
    print("  4. Watch the robot navigate there!")
    print("  5. Click a new point anytime to change destination")
    print()
    print("  Press Ctrl+C to stop.")
    print("=" * 60)

    if not CHECKPOINT.exists():
        print(f"❌ Checkpoint not found: {CHECKPOINT}")
        print("   Run training first or download from GitHub Releases.")
        return

    # ── Load PPO Actor ────────────────────────────────────────────────────────
    device = torch.device("cpu")
    actor  = Actor(observation_dim=50, action_dim=2).to(device)
    ckpt   = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    # Handle both raw state_dict and wrapped checkpoint formats
    state  = ckpt.get("actor", ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt)))
    actor.load_state_dict(state)
    actor.eval()
    print("✅ PPO Actor loaded successfully.")

    # ── ROS2 ─────────────────────────────────────────────────────────────────
    rclpy.init()
    node = DemoNode()

    print("\n⏳ Waiting for sensor data (5 s)...")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    print("✅ Sensors ready. Waiting for your first RViz click...\n")

    period = 1.0 / CONTROL_HZ

    try:
        while rclpy.ok():
            loop_start = time.time()

            # Process all pending ROS messages
            rclpy.spin_once(node, timeout_sec=0.0)
            node._update_pose()

            if node.goal_reached or node.target_position is None:
                # Idle — just spin and wait for a click
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, period - elapsed))
                continue

            # ── Check if goal reached ─────────────────────────────────────────
            dist = np.linalg.norm(node.target_position - node.robot_pos)
            if dist <= GOAL_RADIUS:
                node.stop()
                node.goal_reached = True
                gx, gy = node.target_position
                print(f"✅ GOAL REACHED! ({gx:.2f}, {gy:.2f})  —  "
                      f"Click another point to continue.\n")
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, period - elapsed))
                continue

            # ── Run PPO inference ─────────────────────────────────────────────
            obs_np = node.get_obs()
            obs_t  = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action, _, _ = actor(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

            lin, ang = node.publish_velocity(action_np)

            # Status print every second
            min_lidar = float(node.laser_ranges.min())
            print(f"\r  dist={dist:.2f}m  clr={min_lidar:.2f}m  "
                  f"v={lin:+.2f}m/s  ω={ang:+.2f}rad/s    ", end="", flush=True)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        print("\n\n👋 Demo stopped. Goodbye!")


if __name__ == "__main__":
    main()
