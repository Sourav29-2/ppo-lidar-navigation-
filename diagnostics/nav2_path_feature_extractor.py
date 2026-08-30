#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped, PoseWithCovarianceStamped
import tf2_ros
import tf2_geometry_msgs
import math
import time
import csv
import os

class Nav2FeatureExtractor(Node):
    def __init__(self):
        super().__init__('nav2_feature_extractor')
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.path_sub = self.create_subscription(Path, '/plan', self.path_callback, 10)
        
        self.current_test = "NONE"
        self.csv_file = open(os.path.join(os.path.dirname(__file__), 'nav2_path_features.csv'), 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "Test_Name", "Timestamp", "Robot_X", "Robot_Y", "Robot_Yaw",
            "Path_Length_Meters", "Num_Poses",
            "WP1_X", "WP1_Y", "WP2_X", "WP2_Y", "WP3_X", "WP3_Y", "WP4_X", "WP4_Y", "WP5_X", "WP5_Y"
        ])
        
        self.get_logger().info("Nav2 Feature Extractor Initialized.")

    def get_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            q = trans.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return x, y, yaw
        except Exception as e:
            return None, None, None

    def transform_point_to_robot_frame(self, x, y, from_frame='map', to_frame='base_footprint'):
        try:
            pt = PointStamped()
            pt.header.frame_id = from_frame
            pt.header.stamp = self.get_clock().now().to_msg()
            pt.point.x = x
            pt.point.y = y
            pt.point.z = 0.0
            
            # lookup_transform gets the transform at the latest available time
            transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
            pt_transformed = tf2_geometry_msgs.do_transform_point(pt, transform)
            return pt_transformed.point.x, pt_transformed.point.y
        except Exception as e:
            self.get_logger().warn(f"TF Error: {str(e)}")
            return None, None

    def path_callback(self, msg):
        if self.current_test == "NONE":
            return
            
        num_poses = len(msg.poses)
        if num_poses == 0:
            return
            
        # Calculate cumulative distance
        cumulative_distances = [0.0]
        for i in range(1, num_poses):
            p1 = msg.poses[i-1].pose.position
            p2 = msg.poses[i].pose.position
            dist = math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
            cumulative_distances.append(cumulative_distances[-1] + dist)
            
        total_length = cumulative_distances[-1]
        
        # Sample points at exactly 0.5, 1.0, 1.5, 2.0, 2.5 meters
        target_distances = [0.5, 1.0, 1.5, 2.0, 2.5]
        sampled_points = []
        
        for target in target_distances:
            if target >= total_length:
                # If path is shorter than target, use the last point
                sampled_points.append((msg.poses[-1].pose.position.x, msg.poses[-1].pose.position.y))
                continue
                
            # Find the segment where the target distance falls
            for i in range(1, num_poses):
                if cumulative_distances[i] >= target:
                    # Linear interpolation
                    d1 = cumulative_distances[i-1]
                    d2 = cumulative_distances[i]
                    ratio = (target - d1) / (d2 - d1) if (d2 - d1) > 1e-5 else 0.0
                    
                    p1 = msg.poses[i-1].pose.position
                    p2 = msg.poses[i].pose.position
                    
                    sx = p1.x + ratio * (p2.x - p1.x)
                    sy = p1.y + ratio * (p2.y - p1.y)
                    sampled_points.append((sx, sy))
                    break
                    
        # Transform to robot frame
        relative_points = []
        for sx, sy in sampled_points:
            rx, ry = self.transform_point_to_robot_frame(sx, sy, msg.header.frame_id, 'base_footprint')
            relative_points.append((rx, ry))
            
        # Log to CSV if valid
        if all(pt[0] is not None for pt in relative_points):
            rx, ry, ryaw = self.get_robot_pose()
            row = [
                self.current_test, self.get_clock().now().nanoseconds / 1e9,
                rx, ry, ryaw, total_length, num_poses
            ]
            for pt in relative_points:
                row.extend([pt[0], pt[1]])
                
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            
            # Also log a summary line to console
            self.get_logger().info(f"[{self.current_test}] Path: {num_poses} poses, {total_length:.2f}m. Lookaheads (x,y): " + 
                                   ", ".join([f"({p[0]:.2f}, {p[1]:.2f})" for p in relative_points]))
        else:
            self.get_logger().warn(f"[{self.current_test}] Failed to transform all points to robot frame.")

    def send_goal(self, x, y, theta=0.0):
        self.get_logger().info(f"Waiting for action server...")
        self.action_client.wait_for_server()
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)
        
        self.get_logger().info(f"Sending goal: x={x}, y={y}")
        return self.action_client.send_goal_async(goal_msg)


def run_test_suite():
    rclpy.init()
    node = Nav2FeatureExtractor()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    import threading
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    time.sleep(2.0)
    
    # We must reset the robot pose to (0,0) before tests to make sure we don't start from a failed aborted position
    # But since we can't easily reset Gazebo pose via python reliably without services, we'll just send the initial pose
    init_pose_pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
    time.sleep(1.0)
    
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.pose.position.x = 0.0
    msg.pose.pose.position.y = 0.0
    msg.pose.pose.orientation.w = 1.0
    init_pose_pub.publish(msg)
    time.sleep(2.0)
    
    tests = [
        ("TEST_3_WALL_BLOCKED", -5.0, 5.0)
    ]
    
    for name, x, y in tests:
        node.current_test = name
        node.get_logger().info(f"\nSTARTING {name}")
        future = node.send_goal(x, y)
        
        while not future.done():
            time.sleep(0.1)
            
        goal_handle = future.result()
        if not goal_handle.accepted:
            continue
            
        # Give it 15 seconds to plan and try to navigate so we can collect multiple path samples
        # as the controller tries to move.
        wait_start = time.time()
        result_future = goal_handle.get_result_async()
        
        while not result_future.done() and time.time() - wait_start < 15.0:
            time.sleep(0.5)
            
        if not result_future.done():
            # Cancel goal if it's still running after 15s to proceed to next test
            goal_handle.cancel_goal_async()
            time.sleep(1.0)
            
        time.sleep(3.0) # Pause before next
        
    node.current_test = "NONE"
    node.get_logger().info("Tests completed.")
    rclpy.shutdown()
    spin_thread.join()

if __name__ == '__main__':
    run_test_suite()
