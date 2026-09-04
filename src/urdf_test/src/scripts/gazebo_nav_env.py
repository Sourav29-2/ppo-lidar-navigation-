import gymnasium as gym
from gymnasium import spaces
import rclpy
from rclpy.node import Node
import numpy as np
import time

# ROS 2 Standard Message Formats
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, PointStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
import tf2_ros
import tf2_geometry_msgs
import math
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Clock
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker

# ── Observation constants ─────────────────────────────────────────────────────
# The LiDAR produces 360 raw rays which are reduced to 36 sectors of 10 rays
# each (taking the minimum per sector). These 36 sector values are the first
# 36 elements of the 50D observation vector.
NUM_LIDAR_RAYS   = 360  # raw rays from the LaserScan topic
NUM_SCAN_SECTORS = 36   # sectors after preprocessing (360 rays ÷ 10 per sector)
# Full observation: 36 scan sectors + 2 goal + 2 velocity + 10 Nav2 path = 50D
OBS_DIM = NUM_SCAN_SECTORS + 2 + 2 + 10  # = 50

class GazeboMacNavEnv(gym.Env):
    """Custom Gymnasium Environment for Hybrid Nav2 + Deep RL Navigation on Mac.

    Observation space (50D):
        [0:36]  - NUM_SCAN_SECTORS normalized LiDAR sectors (min of 10 rays each)
        [36:38] - Normalized distance and heading to lookahead waypoint
        [38:40] - Normalized linear and angular velocity
        [40:50] - 5 Nav2 path waypoints in robot frame (x, y) × 5, normalized by 3.0m

    Action space (2D):
        [0] linear velocity  normalized to [-1, 1]
        [1] angular velocity normalized to [-1, 1]
    """

    def __init__(self):
        super(GazeboMacNavEnv, self).__init__()
        
        # 1. Initialize ROS 2 runtime background context
        if not rclpy.ok():
            rclpy.init()
            
        self.node = Node('gazebo_mac_gym_node')
        self.num_scan_features = NUM_LIDAR_RAYS   # raw rays from LaserScan
        
        # Continuous Boundaries mapping our differential robot capacity
        self.min_linear_vel = -0.22  # m/s (allows backing up away from walls)
        self.max_linear_vel = 0.33   # m/s
        self.max_angular_vel = 2.84  # rad/s
        
        # Predefined safe target positions deep inside the bedrooms and kitchen (clear of obstacles)
        self.safe_goals = [
            np.array([-3.5, 6.0]),   # Bedroom 1 (Open space)
            np.array([3.5, 6.0]),    # Bedroom 2 (Open space)
            np.array([-3.5, -6.0]),  # Bedroom 3 (Open space)
            np.array([4.5, -5.5]),   # Kitchen (Open center)
            np.array([3.5, 0.0]),    # Hallway East
            np.array([-3.5, 0.0])    # Hallway West
        ]
        self.target_position = self.safe_goals[0]
        
        # Action space: [linear_velocity, angular_velocity] normalized to [-1.0, 1.0] for stable PPO Gaussian scaling
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        self.max_scan_range = 12.0
        
        # Observation Space: 36 preprocessed sectors + 2 normalized waypoints + 2 normalized velocities + 10 path features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32
        )
        self.node.get_logger().info(f"Initialized environment with observation shape: {self.observation_space.shape}")
        
        # TF Setup for Nav2 path transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        
        # 2. ROS 2 Communication Pipes
        self.vel_pub = self.node.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)
        self.initialpose_pub = self.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        marker_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.node.create_publisher(Marker, '/goal_marker', marker_qos)
        
        self.scan_sub = self.node.create_subscription(LaserScan, '/scan', self._scan_callback, 10)
        self.odom_sub = self.node.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        self.clock_sub = self.node.create_subscription(Clock, '/clock', self._clock_callback, qos_profile_sensor_data)
        self.path_sub = self.node.create_subscription(Path, '/plan', self._path_callback, 10)
        
        # 3. Simulator Control Service Hooks
        self.reset_sim_client = self.node.create_client(Empty, '/reset_world')
        self.unpause_physics_client = self.node.create_client(Empty, '/unpause_physics')
        self.pause_physics_client = self.node.create_client(Empty, '/pause_physics')

        # Runtime Data Arrays
        self.laser_ranges = np.ones(self.num_scan_features, dtype=np.float32) * 10.0
        self.robot_pos = np.array([0.0, 0.0])
        self.robot_yaw = 0.0
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.sim_time = 0.0
        self.current_path = None
        self.path_received_time = 0.0
        self.prev_distance_to_wp = 0.0
        self.prev_min_laser = 10.0
        
        self.step_count = 0
        self.stuck_steps = 0
        self.max_steps_per_episode = 800 # 40 seconds of simulation time (reduced to prevent excessive wandering)
        self.path_length = 0.0
        self.prev_robot_pos = np.array([0.0, 0.0])
        self.position_history = []
        
        # ── Diagnostic / Debug mode ──────────────────────────────────────────
        # Set to True externally (env.debug_mode = True) to enable per-step
        # reward-component printing and episode-level counters.
        self.debug_mode = False
        self.debug_log_interval = 100  # print every N steps in debug mode
        
        # Episode-level diagnostic counters (reset each episode)
        self._diag_reverse_attempts = 0
        self._diag_reverse_rewarded = 0
        self._diag_high_angular_steps = 0
        self._diag_near_obstacle_steps = 0
        self._diag_collision_count = 0
        self._diag_success_count = 0
        self._diag_total_progress = 0.0
        self._diag_total_clearance = 0.0
        self._diag_total_turning = 0.0
        self._diag_total_time = 0.0
        self._diag_total_inflation = 0.0
        self._diag_total_reverse = 0.0
        self._diag_total_success = 0.0
        self._diag_total_collision = 0.0
        self._diag_total_reward = 0.0
        self._diag_total_reverse_penalty = 0.0
        
        # Oscillation-aware diagnostics
        self._diag_oscillation_events = 0
        self._diag_total_oscillation_penalty = 0.0
        self._diag_total_safety_margin = 0.0
        
        self.angular_history = []
        self.last_meaningful_turn_direction = 0
        self.reversals_in_window = 0
        
        # Physical behavior diagnostic counters
        self._diag_forward_steps = 0
        self._diag_reverse_steps = 0
        self._diag_stationary_steps = 0
        self._diag_high_rotation_steps = 0
        self._diag_sum_linear_vel = 0.0
        self._diag_sum_angular_vel = 0.0
        
        self.prev_distance_to_goal = 0.0
        
    def _scan_callback(self, msg):
        if len(msg.ranges) > 0:
            self.max_scan_range = float(msg.range_max)
            indices = np.linspace(0, len(msg.ranges) - 1, self.num_scan_features, dtype=int)
            ranges = np.array(msg.ranges)[indices]
            ranges[np.isinf(ranges)] = msg.range_max
            ranges[np.isnan(ranges)] = msg.range_max
            # Filter out internal chassis self-reflections (< 0.08m)
            ranges[ranges < 0.08] = msg.range_max
            self.laser_ranges = ranges.astype(np.float32)

    def _odom_callback(self, msg):
        # We no longer use raw odom for global position to prevent frame drift bugs
        # self.robot_pos and self.robot_yaw are updated via tf2 in _update_robot_pose()
        
        # Store current velocities for ego-state feedback
        self.current_linear_vel = msg.twist.twist.linear.x
        self.current_angular_vel = msg.twist.twist.angular.z

    def _clock_callback(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _path_callback(self, msg):
        self.current_path = msg
        self.path_received_time = self.sim_time

    def _update_robot_pose(self):
        try:
            # Get physical location in the map frame to match goal frame and Nav2 path
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_footprint',
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.05)
            )
            self.robot_pos = np.array([
                transform.transform.translation.x, 
                transform.transform.translation.y
            ])
            q = transform.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_yaw = np.arctan2(siny_cosp, cosy_cosp)
        except Exception as e:
            pass # Keep previous position if transform fails

    def _call_service(self, client):
        # Spin node once to trigger discovery processing
        rclpy.spin_once(self.node, timeout_sec=0.1)
        while not client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(f'Waiting for service {client.srv_name}...')
            rclpy.spin_once(self.node, timeout_sec=0.1)
        
        req = Empty.Request()
        future = client.call_async(req)
        # Block until the call completes to ensure Gazebo executes it before we proceed
        rclpy.spin_until_future_complete(self.node, future)

    def preprocess_lidar(self, scan):
        """Sanitize, clip, normalize, and reduce raw LiDAR rays to NUM_SCAN_SECTORS sectors.

        Reduces NUM_LIDAR_RAYS (360) raw rays to NUM_SCAN_SECTORS (36) sectors
        by taking the minimum distance reading within each sector of 10 rays.
        """
        scan = np.array(scan, dtype=np.float32)
        # Ensure scan has exactly NUM_LIDAR_RAYS elements
        if len(scan) != NUM_LIDAR_RAYS:
            indices = np.linspace(0, len(scan) - 1, NUM_LIDAR_RAYS, dtype=int)
            scan = scan[indices]

        # 1. Sanitize: Replace non-finite or negative values with max_scan_range
        invalid_mask = ~np.isfinite(scan) | (scan < 0.0)
        scan[invalid_mask] = self.max_scan_range

        # 2. Clip to [0, max_scan_range]
        clipped = np.clip(scan, 0.0, self.max_scan_range)

        # 3. Normalize to [0, 1]
        normalized = clipped / self.max_scan_range

        # 4. Sector reduction: NUM_LIDAR_RAYS -> NUM_SCAN_SECTORS sectors,
        #    each covering (NUM_LIDAR_RAYS // NUM_SCAN_SECTORS) = 10 rays.
        rays_per_sector = NUM_LIDAR_RAYS // NUM_SCAN_SECTORS
        sectors = normalized.reshape(NUM_SCAN_SECTORS, rays_per_sector)
        reduced = np.min(sectors, axis=1)

        return reduced.astype(np.float32)


    def _get_obs(self):
        # 1. Base Goal & Heading (0.5m lookahead or final goal)
        waypoint = self.target_position # Default fallback
        closest_idx = 0
        if self.current_path is not None and len(self.current_path.poses) > 0:
            # Find closest idx
            min_dist = float('inf')
            for idx, pose_stamped in enumerate(self.current_path.poses):
                p = np.array([pose_stamped.pose.position.x, pose_stamped.pose.position.y])
                d = np.linalg.norm(p - self.robot_pos)
                if d < min_dist:
                    min_dist = d
                    closest_idx = idx
            
            # Find first waypoint >= 0.5m away for heading calculation
            for idx in range(closest_idx, len(self.current_path.poses)):
                pose_stamped = self.current_path.poses[idx]
                p = np.array([pose_stamped.pose.position.x, pose_stamped.pose.position.y])
                d = np.linalg.norm(p - self.robot_pos)
                if d >= 0.5:
                    waypoint = p
                    break
            else:
                last_pose = self.current_path.poses[-1].pose.position
                waypoint = np.array([last_pose.x, last_pose.y])
                
        # Compute distance and heading to lookahead waypoint
        goal_vec = waypoint - self.robot_pos
        D_wp = np.linalg.norm(goal_vec)
        global_heading = np.arctan2(goal_vec[1], goal_vec[0])
        heading_error = global_heading - self.robot_yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # 2. Extract 10 Nav2 Path Features (5 points x 2 coords)
        path_features = np.zeros(10, dtype=np.float32)
        self.last_fallback_used = True
        
        path_age = self.sim_time - self.path_received_time
        path_stale = path_age > 2.0
        
        if self.current_path is not None and len(self.current_path.poses) > 0 and not path_stale:
            try:
                # Calculate cumulative distances starting from closest_idx
                cumulative_distances = [0.0]
                for i in range(closest_idx + 1, len(self.current_path.poses)):
                    p1 = self.current_path.poses[i-1].pose.position
                    p2 = self.current_path.poses[i].pose.position
                    dist = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
                    cumulative_distances.append(cumulative_distances[-1] + dist)
                    
                total_length = cumulative_distances[-1]
                target_distances = [0.5, 1.0, 1.5, 2.0, 2.5]
                sampled_points = []
                
                for target in target_distances:
                    if target >= total_length:
                        # Use last point if path is shorter
                        last_pos = self.current_path.poses[-1].pose.position
                        sampled_points.append((last_pos.x, last_pos.y))
                        continue
                        
                    for i in range(1, len(cumulative_distances)):
                        if cumulative_distances[i] >= target:
                            d1 = cumulative_distances[i-1]
                            d2 = cumulative_distances[i]
                            ratio = (target - d1) / (d2 - d1) if (d2 - d1) > 1e-5 else 0.0
                            
                            p1 = self.current_path.poses[closest_idx + i - 1].pose.position
                            p2 = self.current_path.poses[closest_idx + i].pose.position
                            
                            sx = p1.x + ratio * (p2.x - p1.x)
                            sy = p1.y + ratio * (p2.y - p1.y)
                            sampled_points.append((sx, sy))
                            break
                            
                # Transform sampled points to robot local frame (base_footprint)
                transform = self.tf_buffer.lookup_transform(
                    'base_footprint', 
                    self.current_path.header.frame_id, 
                    rclpy.time.Time(), 
                    rclpy.duration.Duration(seconds=0.1)
                )
                
                for i, (sx, sy) in enumerate(sampled_points):
                    pt = PointStamped()
                    pt.point.x = float(sx)
                    pt.point.y = float(sy)
                    pt.point.z = 0.0
                    pt_transformed = tf2_geometry_msgs.do_transform_point(pt, transform)
                    # Normalize by dividing by 3.0
                    path_features[i*2] = pt_transformed.point.x / 3.0
                    path_features[i*2 + 1] = pt_transformed.point.y / 3.0
                    
                self.last_fallback_used = False
                
            except Exception as e:
                self.node.get_logger().warn(f"Path transformation failed, using fallback: {str(e)}")
                self.last_fallback_used = True
                
        if self.last_fallback_used:
            # Explicit fallback: pretend the "path" goes directly toward the final global goal
            self.node.get_logger().info(f"FALLBACK USED (stale={path_stale}, age={path_age:.2f}s). Generating dummy goal-directed path features.")
            target_distances = [0.5, 1.0, 1.5, 2.0, 2.5]
            
            # Goal in robot frame:
            goal_x_global = self.target_position[0]
            goal_y_global = self.target_position[1]
            
            # Simple rotation based on robot_yaw
            dx = goal_x_global - self.robot_pos[0]
            dy = goal_y_global - self.robot_pos[1]
            local_goal_x = dx * math.cos(-self.robot_yaw) - dy * math.sin(-self.robot_yaw)
            local_goal_y = dx * math.sin(-self.robot_yaw) + dy * math.cos(-self.robot_yaw)
            
            dist_to_goal = math.sqrt(local_goal_x**2 + local_goal_y**2)
            dir_x = local_goal_x / dist_to_goal if dist_to_goal > 0 else 1.0
            dir_y = local_goal_y / dist_to_goal if dist_to_goal > 0 else 0.0
            
            for i, target in enumerate(target_distances):
                actual_dist = min(target, dist_to_goal)
                path_features[i*2] = (dir_x * actual_dist) / 3.0
                path_features[i*2 + 1] = (dir_y * actual_dist) / 3.0
        
        # 3. Base Features
        norm_scans = self.preprocess_lidar(self.laser_ranges)
        norm_d_wp = float(np.clip(D_wp / 5.0, 0.0, 1.0))
        norm_heading = float(heading_error / np.pi)
        norm_linear = float(self.current_linear_vel / self.max_linear_vel)
        norm_angular = float(self.current_angular_vel / self.max_angular_vel)
        
        # Assemble 50D observation vector EXACTLY in the requested order
        obs = np.concatenate([
            norm_scans,                 # 36
            [norm_d_wp, norm_heading],  # 2
            [norm_linear, norm_angular],# 2
            path_features               # 10
        ]).astype(np.float32)
        
        return obs

    def scale_action(self, action):
        """Map normalized action [-1.0, 1.0] to physical robot velocities."""
        # Clamp action to [-1.0, 1.0] to ensure safety boundaries
        clamped_linear = float(np.clip(action[0], -1.0, 1.0))
        clamped_angular = float(np.clip(action[1], -1.0, 1.0))
        
        linear_vel = float((clamped_linear + 1.0) / 2.0 * (self.max_linear_vel - self.min_linear_vel) + self.min_linear_vel)
        angular_vel = float(clamped_angular * self.max_angular_vel)
        return linear_vel, angular_vel

    def step(self, action):
        self.step_count += 1
        
        # 1. Unpause physics to allow the robot to physically react in Gazebo
        self._call_service(self.unpause_physics_client)
        
        # 2. Spin once to ensure we have a fresh simulation time
        rclpy.spin_once(self.node, timeout_sec=0.005)
        start_sim_time = self.sim_time
        
        # 3. Scale the clamped action using the unified scale_action method
        linear_vel, angular_vel = self.scale_action(action)
        
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel
        cmd_vel.angular.z = angular_vel
        self.vel_pub.publish(cmd_vel)
        
        # 4. Spin the node until simulation time has advanced by exactly 0.05 seconds
        target_sim_time = start_sim_time + 0.05
        # Prevent infinite loops in case the clock fails to update
        loop_start_real = time.time()
        while self.sim_time < target_sim_time:
            rclpy.spin_once(self.node, timeout_sec=0.005)
            if time.time() - loop_start_real > 2.0:
                break
        
        # 5. Freeze physics so neural calculations don't cause the simulation clock to drift
        self._call_service(self.pause_physics_client)
        
        # Spin to process the last odom and scan messages published during the step
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
        self._update_robot_pose()
        
        step_dist = np.linalg.norm(self.robot_pos - self.prev_robot_pos)
        self.path_length += step_dist
        self.prev_robot_pos = np.copy(self.robot_pos)
        
        self.position_history.append(np.copy(self.robot_pos))
        if len(self.position_history) > 100:
            self.position_history.pop(0)
            
        # 6. Process state matrices and evaluate mathematical performance metrics
        obs = self._get_obs()
        
        # Transform laser scans to robot base_link frame to check rectangular footprint collisions
        is_collision = False
        is_near_obstacle = False
        angles = np.linspace(-3.14, 3.14, len(self.laser_ranges))
        for r, theta in zip(self.laser_ranges, angles):
            # Laser joint is at x=0.4 relative to base_link
            x_r = r * np.cos(theta) + 0.4
            y_r = r * np.sin(theta)
            
            # Check collision: exact box inflated by 0.03m (front: 0.53, rear: -0.13, sides: [-0.23, 0.23])
            if (-0.13 <= x_r <= 0.53) and (-0.23 <= y_r <= 0.23):
                is_collision = True
                break
            
            # Check inflation zone warning: box inflated by 0.15m (front: 0.65, rear: -0.25, sides: [-0.35, 0.35])
            if (-0.25 <= x_r <= 0.65) and (-0.35 <= y_r <= 0.35):
                is_near_obstacle = True
        
        # Calculate raw unnormalized distance to lookahead waypoint for progress reward math
        waypoint = self.target_position
        if self.current_path is not None and len(self.current_path.poses) > 0:
            closest_idx = 0
            min_dist = float('inf')
            for idx, pose_stamped in enumerate(self.current_path.poses):
                p = np.array([pose_stamped.pose.position.x, pose_stamped.pose.position.y])
                d = np.linalg.norm(p - self.robot_pos)
                if d < min_dist:
                    min_dist = d
                    closest_idx = idx
            for idx in range(closest_idx, len(self.current_path.poses)):
                pose_stamped = self.current_path.poses[idx]
                p = np.array([pose_stamped.pose.position.x, pose_stamped.pose.position.y])
                d = np.linalg.norm(p - self.robot_pos)
                if d >= 0.5:
                    waypoint = p
                    break
            else:
                last_pose = self.current_path.poses[-1].pose.position
                waypoint = np.array([last_pose.x, last_pose.y])
        raw_D_wp = np.linalg.norm(waypoint - self.robot_pos)
        
        terminated = False
        truncated = False
        
        # PART 1: Progress towards final goal
        distance_to_final_goal = np.linalg.norm(self.target_position - self.robot_pos)
        progress_change = self.prev_distance_to_goal - distance_to_final_goal
        progress_reward = progress_change * 15.0

        # PART 2: Obstacle clearance change with clipping
        current_min_laser = float(np.min(self.laser_ranges))
        clearance_change = current_min_laser - self.prev_min_laser
        self.prev_min_laser = current_min_laser
        safe_clearance_change = np.clip(clearance_change, -0.05, 0.05)
        clearance_reward = 5.0 * safe_clearance_change

        # PART 3: Turning penalty using ACTUAL angular velocity
        turning_penalty = -0.05 * (float(angular_vel) ** 2)

        # PART 4: Constant time/step penalty
        time_penalty = -0.05

        # PART 5: Reverse penalty
        reverse_penalty = -0.05 if linear_vel < -0.05 else 0.0

        # PART 6: Useful recovery reward
        reverse_recovery_reward = 0.0
        if is_near_obstacle:
            if linear_vel < -0.05:
                self._diag_reverse_attempts += 1
                if clearance_change > 0.02:
                    reverse_recovery_reward = 0.20
                    self._diag_reverse_rewarded += 1

        # PART 5 & 6: Inflation zone penalty
        inflation_penalty = -0.05 if is_near_obstacle else 0.0

        # PART NEW: Oscillation Penalty
        oscillation_penalty = 0.0
        ANGULAR_DEADBAND = 0.2
        OSCILLATION_WINDOW = 3.0
        MIN_REVERSALS = 3
        OSCILLATION_CLEARANCE = 0.5
        
        # 1. Determine direction of current turn
        if angular_vel > ANGULAR_DEADBAND:
            current_dir = 1
        elif angular_vel < -ANGULAR_DEADBAND:
            current_dir = -1
        else:
            current_dir = 0
            
        # 2. Add to history (timestamp, direction)
        if current_dir != 0:
            self.angular_history.append((self.sim_time, current_dir))
            
        # 3. Prune old history outside the window
        while len(self.angular_history) > 0 and (self.sim_time - self.angular_history[0][0]) > OSCILLATION_WINDOW:
            self.angular_history.pop(0)
            
        # 4. Count reversals in window
        self.reversals_in_window = 0
        if len(self.angular_history) >= 2:
            prev_dir = self.angular_history[0][1]
            for _, d in self.angular_history[1:]:
                if d != prev_dir:
                    self.reversals_in_window += 1
                    prev_dir = d
                    
        # 5. Apply penalty if thresholds met
        if self.reversals_in_window >= MIN_REVERSALS and current_min_laser < OSCILLATION_CLEARANCE:
            # Check progress context. Are we moving anywhere?
            # E.g., displacement over the window (max 30 steps ~ 1.5s real, sim time is 3.0s window)
            # Find the position 30 steps ago, or earliest available if < 30
            idx = max(0, len(self.position_history) - int(OSCILLATION_WINDOW / 0.05))
            if idx < len(self.position_history):
                progress_in_window = np.linalg.norm(self.position_history[idx] - self.target_position) - distance_to_final_goal
                if progress_in_window < 0.1:
                    oscillation_penalty = -0.25
                    self._diag_oscillation_events += 1

        # PART NEW: Safety Margin Penalty
        safety_margin_penalty = 0.0
        SAFETY_WARNING_DIST = 0.60
        SAFETY_CRITICAL_DIST = 0.35
        MAX_SAFETY_PENALTY = -2.0

        if current_min_laser < SAFETY_WARNING_DIST and linear_vel > 0.0:
            speed_factor = linear_vel / 0.4
            
            if current_min_laser > SAFETY_CRITICAL_DIST:
                proximity_factor = (SAFETY_WARNING_DIST - current_min_laser) / (SAFETY_WARNING_DIST - SAFETY_CRITICAL_DIST)
            else:
                proximity_factor = 1.0
                
            trend_factor = 1.0
            if clearance_change > 0:
                trend_factor = 0.0
                
            safety_margin_penalty = MAX_SAFETY_PENALTY * speed_factor * proximity_factor * trend_factor

        self.prev_distance_to_goal = distance_to_final_goal
        self.prev_distance_to_wp = raw_D_wp

        info = {
            'is_success': False,
            'is_collision': False,
            'path_length': self.path_length,
            'episode_time': self.sim_time
        }

        # PART 7: Success terminal reward
        success_reward = 0.0
        if distance_to_final_goal < 0.50:
            success_reward = 100.0
            terminated = True
            info['is_success'] = True

        # Stuck detection monitor
        if abs(linear_vel) > 0.05 and abs(self.current_linear_vel) < 0.02:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        if self.stuck_steps >= 30:
            is_collision = True

        # PART 8: Collision/stuck terminal penalty
        collision_penalty = 0.0
        if is_collision:
            collision_penalty = -100.0
            terminated = True
            info['is_collision'] = True

        # ── Aggregate reward ─────────────────────────────────────────────────
        reward = (
            progress_reward
            + clearance_reward
            + turning_penalty
            + time_penalty
            + inflation_penalty
            + reverse_penalty
            + reverse_recovery_reward
            + success_reward
            + collision_penalty
            + oscillation_penalty
            + safety_margin_penalty
        )

        # Pack reward components into info for diagnostics (PART 8)
        info["reward_components"] = {
            "progress": progress_reward,
            "clearance": clearance_reward,
            "turning": turning_penalty,
            "time": time_penalty,
            "inflation": inflation_penalty,
            "reverse_penalty": reverse_penalty,
            "reverse_recovery": reverse_recovery_reward,
            "oscillation_penalty": oscillation_penalty,
            "safety_margin": safety_margin_penalty,
            "success": success_reward,
            "collision": collision_penalty,
            "total": reward,
        }

        # Verify sum of all components matches total_reward exactly
        components_sum = (
            progress_reward
            + clearance_reward
            + turning_penalty
            + time_penalty
            + inflation_penalty
            + reverse_penalty
            + reverse_recovery_reward
            + oscillation_penalty
            + safety_margin_penalty
            + success_reward
            + collision_penalty
        )
        assert abs(components_sum - reward) < 1e-7, f"Reward mismatch: sum={components_sum}, total={reward}"

        # Track step movement statistics
        if linear_vel > 0.05:
            self._diag_forward_steps += 1
        elif linear_vel < -0.05:
            self._diag_reverse_steps += 1
        else:
            self._diag_stationary_steps += 1
            
        if abs(angular_vel) > 1.0:
            self._diag_high_rotation_steps += 1
            
        self._diag_sum_linear_vel += linear_vel
        self._diag_sum_angular_vel += angular_vel

        # Internal consistency check (debug builds only)
        if self.debug_mode:
            # Update episode accumulators
            self._diag_total_progress += progress_reward
            self._diag_total_clearance += clearance_reward
            self._diag_total_turning += turning_penalty
            self._diag_total_time += time_penalty
            self._diag_total_inflation += inflation_penalty
            self._diag_total_reverse += reverse_recovery_reward
            self._diag_total_reverse_penalty += reverse_penalty
            self._diag_total_oscillation_penalty += oscillation_penalty
            self._diag_total_safety_margin += safety_margin_penalty
            self._diag_total_success += success_reward
            self._diag_total_collision += collision_penalty
            self._diag_total_reward += reward

            if is_near_obstacle:
                self._diag_near_obstacle_steps += 1
            if abs(angular_vel) > 1.0:
                self._diag_high_angular_steps += 1
            if info['is_success']:
                self._diag_success_count += 1
            if info['is_collision']:
                self._diag_collision_count += 1

            # Per-step reward component print every debug_log_interval steps
            if self.step_count % self.debug_log_interval == 0:
                print(
                    f"\n[Step {self.step_count:04d}] Reward Components:"
                    f"\n  Progress:              {progress_reward:+.4f}"
                    f"\n  Clearance:             {clearance_reward:+.4f}"
                    f"\n  Turning Penalty:       {turning_penalty:+.4f}"
                    f"\n  Time Penalty:          {time_penalty:+.4f}"
                    f"\n  Inflation Penalty:     {inflation_penalty:+.4f}"
                    f"\n  Safety Margin Penalty: {safety_margin_penalty:+.4f}"
                    f"\n  Reverse Penalty:       {reverse_penalty:+.4f}"
                    f"\n  Reverse Recovery:      {reverse_recovery_reward:+.4f}"
                    f"\n  Avoidance Pressure:    {avoidance_pressure_penalty:+.4f}"
                    f"\n  Clearance Improvement: {clearance_improvement_reward:+.4f}"
                    f"\n  Critical Penalty:      {critical_zone_penalty:+.4f}"
                    f"\n  Stagnation Penalty:    {stagnation_penalty:+.4f}"
                    f"\n  Success:               {success_reward:+.4f}"
                    f"\n  Collision:             {collision_penalty:+.4f}"
                    f"\n  {'─'*32}"
                    f"\n  Total Reward:          {reward:+.4f}"
                    f"\n  State:"
                    f"\n    Linear Velocity:     {linear_vel:.4f} m/s"
                    f"\n    Angular Velocity:    {angular_vel:.4f} rad/s"
                    f"\n    Min LiDAR:           {current_min_laser:.4f} m"
                    f"\n    Clearance Change:    {clearance_change:+.4f} m"
                    f"\n    Near Obstacle:       {is_near_obstacle}"
                    f"\n    Goal Distance:       {distance_to_final_goal:.4f} m"
                    f"\n    Waypoint Distance:   {raw_D_wp:.4f} m"
                )

            if terminated or truncated:
                steps = max(self.step_count, 1)
                print(
                    f"\n{'='*46}"
                    f"\nREWARD DIAGNOSTIC SUMMARY"
                    f"\n{'='*46}"
                    f"\n  Reverse attempts:          {self._diag_reverse_attempts}"
                    f"\n  Reverse rewarded:          {self._diag_reverse_rewarded}"
                    f"\n  Warning zone steps:        {self._diag_warning_zone_steps}"
                    f"\n  Critical zone steps:       {self._diag_critical_zone_steps}"
                    f"\n  Stagnation steps:          {self._diag_stagnation_steps}"
                    f"\n  Avoidance penalized:       {self._diag_avoidance_penalized_steps}"
                    f"\n  Avoidance rewarded:        {self._diag_avoidance_rewarded_steps}"
                    f"\n  High angular vel steps:    {self._diag_high_angular_steps}"
                    f"\n  Near obstacle steps:       {self._diag_near_obstacle_steps}"
                    f"\n  Avg progress reward:       {self._diag_total_progress/steps:+.4f}"
                    f"\n  Avg clearance reward:      {self._diag_total_clearance/steps:+.4f}"
                    f"\n  Avg turning penalty:       {self._diag_total_turning/steps:+.4f}"
                    f"\n  Avg time penalty:          {self._diag_total_time/steps:+.4f}"
                    f"\n  Avg inflation penalty:     {self._diag_total_inflation/steps:+.4f}"
                    f"\n  Avg safety margin pen:     {self._diag_total_safety_margin/steps:+.4f}"
                    f"\n  Avg reverse penalty:       {self._diag_total_reverse_penalty/steps:+.4f}"
                    f"\n  Avg reverse reward:        {self._diag_total_reverse/steps:+.4f}"
                    f"\n  Avg avoidance penalty:     {self._diag_total_avoidance_penalty/steps:+.4f}"
                    f"\n  Avg clearance imprv rev:   {self._diag_total_improvement_reward/steps:+.4f}"
                    f"\n  Avg critical penalty:      {self._diag_total_critical_penalty/steps:+.4f}"
                    f"\n  Avg stagnation penalty:    {self._diag_total_stagnation_penalty/steps:+.4f}"
                    f"\n  Total episode reward:      {self._diag_total_reward:+.4f}"
                    f"\n  Success:                   {info['is_success']}"
                    f"\n  Collision:                 {info['is_collision']}"
                    f"\n{'='*46}"
                )
                
                # Print Movement Statistics (PART 9)
                f_pct = (self._diag_forward_steps / steps) * 100.0
                r_pct = (self._diag_reverse_steps / steps) * 100.0
                s_pct = (self._diag_stationary_steps / steps) * 100.0
                h_pct = (self._diag_high_rotation_steps / steps) * 100.0
                mean_l = self._diag_sum_linear_vel / steps
                mean_a = self._diag_sum_angular_vel / steps
                print(
                    f"\n===================================="
                    f"\nMovement Statistics:"
                    f"\nForward:"
                    f"\n    {f_pct:.1f} %"
                    f"\nReverse:"
                    f"\n    {r_pct:.1f} %"
                    f"\nStationary:"
                    f"\n    {s_pct:.1f} %"
                    f"\nHigh angular rotation:"
                    f"\n    {h_pct:.1f} %"
                    f"\nMean linear velocity:"
                    f"\n    {mean_l:.4f}"
                    f"\nMean angular velocity:"
                    f"\n    {mean_a:.4f}"
                    f"\n===================================="
                )
        # Episode length boundary timeout truncation
        if self.step_count >= self.max_steps_per_episode:
            truncated = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.stuck_steps = 0
        self.current_path = None
        self.path_length = 0.0
        self.prev_robot_pos = np.array([0.0, 0.0])
        self.position_history = []
        
        # Reset diagnostic counters for new episode
        self._diag_reverse_attempts = 0
        self._diag_reverse_rewarded = 0
        self._diag_high_angular_steps = 0
        self._diag_near_obstacle_steps = 0
        self._diag_collision_count = 0
        self._diag_success_count = 0
        self._diag_total_progress = 0.0
        self._diag_total_clearance = 0.0
        self._diag_total_safety_margin = 0.0
        self._diag_total_turning = 0.0
        self._diag_total_time = 0.0
        self._diag_total_inflation = 0.0
        self._diag_total_reverse = 0.0
        self._diag_total_reverse_penalty = 0.0
        self._diag_total_success = 0.0
        self._diag_total_collision = 0.0
        self._diag_total_reward = 0.0
        # Reset oscillation diagnostics
        self._diag_oscillation_events = 0
        self._diag_total_oscillation_penalty = 0.0
        
        self.angular_history = []
        self.last_meaningful_turn_direction = 0
        self.reversals_in_window = 0
        
        # Reset movement statistics
        self._diag_forward_steps = 0
        self._diag_reverse_steps = 0
        self._diag_stationary_steps = 0
        self._diag_high_rotation_steps = 0
        self._diag_sum_linear_vel = 0.0
        self._diag_sum_angular_vel = 0.0
        
        # Randomly choose a safe target position deep inside a room or kitchen
        if options is not None and "target_position" in options:
            self.target_position = np.array(options["target_position"], dtype=float)
        else:
            self.target_position = self.safe_goals[np.random.choice(len(self.safe_goals))]
        
        # Publish goal marker for visualization in RViz
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = 'goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.target_position[0])
        marker.pose.position.y = float(self.target_position[1])
        marker.pose.position.z = 0.2
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        self.marker_pub.publish(marker)
        
        # 1. Reset simulation (resets coordinates AND simulation clock, resetting odometry)
        self._call_service(self.reset_sim_client)
        self._call_service(self.unpause_physics_client)
        
        # 2. Wait for simulation clock to reset back to near 0.0
        start_wait = time.time()
        while self.sim_time > 1.0 and (time.time() - start_wait) < 2.0:
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
        # 3. Spin to let Gazebo physics run for a few ticks to publish new TF and AMCL transforms
        for _ in range(20):
            rclpy.spin_once(self.node, timeout_sec=0.02)
            
        # 4. Publish initialpose to AMCL to align localization at (0, 0)
        init_pose = PoseWithCovarianceStamped()
        init_pose.header.frame_id = 'map'
        init_pose.header.stamp = self.node.get_clock().now().to_msg()
        init_pose.pose.pose.position.x = 0.0
        init_pose.pose.pose.position.y = 0.0
        init_pose.pose.pose.orientation.w = 1.0
        self.initialpose_pub.publish(init_pose)
        
        # Spin to process initial pose
        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
        # 5. Publish target goal pose to Nav2 to trigger global planning from (0, 0)
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.position.x = float(self.target_position[0])
        goal_msg.pose.position.y = float(self.target_position[1])
        goal_msg.pose.orientation.w = 1.0
        self.goal_pub.publish(goal_msg)
        
        # 6. Spin and wait for the Nav2 path to arrive on /plan
        start_wait = time.time()
        while self.current_path is None and (time.time() - start_wait) < 4.0:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            
        # Spin to clear historical buffers
        for _ in range(10):
            rclpy.spin_once(self.node, timeout_sec=0.01)
            
        self._update_robot_pose()
        self._call_service(self.pause_physics_client)
        self.prev_robot_pos = np.copy(self.robot_pos)
        
        obs = self._get_obs()
        
        # Calculate raw unnormalized distance to lookahead waypoint for progress tracking
        waypoint = self.target_position
        if self.current_path is not None and len(self.current_path.poses) > 0:
            closest_idx = 0
            min_dist = float('inf')
            for idx, pose_stamped in enumerate(self.current_path.poses):
                p = np.array([pose_stamped.pose.position.x, pose_stamped.pose.position.y])
                d = np.linalg.norm(p - self.robot_pos)
                if d < min_dist:
                    min_dist = d
                    closest_idx = idx
            for idx in range(closest_idx, len(self.current_path.poses)):
                pose_stamped = self.current_path.poses[idx]
                p = np.array([pose_stamped.pose.position.x, pose_stamped.pose.position.y])
                d = np.linalg.norm(p - self.robot_pos)
                if d >= 0.5:
                    waypoint = p
                    break
            else:
                last_pose = self.current_path.poses[-1].pose.position
                waypoint = np.array([last_pose.x, last_pose.y])
        self.prev_distance_to_wp = np.linalg.norm(waypoint - self.robot_pos)
        self.prev_distance_to_goal = np.linalg.norm(self.target_position - self.robot_pos)
        self.prev_min_laser = float(np.min(self.laser_ranges))
        
        return obs, {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
