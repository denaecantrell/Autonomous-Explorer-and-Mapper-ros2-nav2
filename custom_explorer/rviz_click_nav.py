import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose, ComputePathToPose


class RVizClickNav(Node):
    """
    Node that:
      - Listens for RViz 2D Nav Goals
      - Computes a path using Nav2
      - Publishes that path to RViz
      - Sends NavigateToPose to make the robot execute it
    """

    def __init__(self):
        super().__init__("rviz_click_nav")

        self.get_logger().info("RViz Click Navigation Node Started")

        # --- Subscribers ---
        # RViz default topic for "2D Nav Goal"
        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_callback,
            10
        )

        # Path publisher for visualization
        self.path_pub = self.create_publisher(
            Path,
            "/rviz_nav_path",
            10
        )

        # --- Nav2 Action Clients ---
        self.compute_path_client = ActionClient(
            self,
            ComputePathToPose,
            "compute_path_to_pose"
        )

        self.navigate_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            "navigate_to_pose"
        )

    # =====================================================================
    # CALLBACK: When RViz user clicks a goal
    # =====================================================================
    def goal_callback(self, msg: PoseStamped):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.get_logger().info(f"Received RViz goal at ({gx:.2f}, {gy:.2f})")

        self.compute_and_execute_path(msg)

    # =====================================================================
    # COMPUTE PATH WITH NAV2
    # =====================================================================
    def compute_and_execute_path(self, pose_msg):
        self.get_logger().info("Requesting ComputePathToPose...")

        goal = ComputePathToPose.Goal()
        goal.goal = pose_msg
        goal.use_start = False

        self.compute_path_client.wait_for_server()
        future = self.compute_path_client.send_goal_async(goal)
        future.add_done_callback(
            lambda f: self._compute_path_response(f, pose_msg)
        )

    def _compute_path_response(self, future, pose_msg):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("ComputePathToPose request was rejected!")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._compute_path_result(f, pose_msg)
        )

    def _compute_path_result(self, future, pose_msg):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(f"ComputePathToPose failed: {e}")
            return

        if result.error_code != 0 or len(result.path.poses) == 0:
            self.get_logger().warn(
                f"Path planning failed (error_code={result.error_code})"
            )
            return

        # Publish path for RViz
        path = result.path
        path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(path)
        self.get_logger().info(
            f"Path found with {len(path.poses)} poses — executing..."
        )

        # Send robot along the path with NavigateToPose
        self.send_nav_goal(pose_msg)

    # =====================================================================
    # EXECUTE PATH WITH NAV2
    # =====================================================================
    def send_nav_goal(self, pose_msg):
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = pose_msg

        self.navigate_to_pose_client.wait_for_server()
        send_future = self.navigate_to_pose_client.send_goal_async(nav_goal)
        send_future.add_done_callback(self._nav_goal_response)

    def _nav_goal_response(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn("NavigateToPose goal was rejected!")
            return

        self.get_logger().info("NavigateToPose goal accepted, executing...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result)

    def _nav_result(self, future):
        try:
            result = future.result().result
            self.get_logger().info(f"Navigation finished with result: {result}")
        except Exception as e:
            self.get_logger().error(f"Navigation failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RVizClickNav()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

