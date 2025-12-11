import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped


class DirectionToGoal(Node):
    """
    Converts a robot-relative direction command into a PoseStamped in map frame.
    
    Input topic:
        /desired_direction  (Twist)
            linear.x = direction x in base_link
            linear.y = direction y in base_link
            linear.z = distance in meters

    Output topic:
        /exploration_goal (PoseStamped, frame_id='map')
    """

    def __init__(self):
        super().__init__('direction_to_goal')

        self.robot_pose = None

        # Subscribe to AMCL pose (map frame)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        # Direction command (base_link frame)
        self.dir_sub = self.create_subscription(
            Twist,
            '/desired_direction',
            self.direction_callback,
            10
        )

        # Publish map-frame exploration goal
        self.goal_pub = self.create_publisher(
            PoseStamped,
            '/exploration_goal',
            10
        )

        self.get_logger().info("direction_to_goal node started")

    def pose_callback(self, msg):
        self.robot_pose = msg.pose.pose

    def direction_callback(self, msg):
        if self.robot_pose is None:
            self.get_logger().warn("No AMCL pose yet; cannot create goal.")
            return

        dx = msg.linear.x
        dy = msg.linear.y
        dist = msg.linear.z

        # direction must be non-zero
        mag = math.hypot(dx, dy)
        if mag < 1e-3 or dist <= 0.0:
            self.get_logger().warn("Invalid direction or distance.")
            return

        dx /= mag
        dy /= mag

        # Robot orientation (yaw) from quaternion
        q = self.robot_pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y*q.y + q.z*q.z)
        yaw = math.atan2(siny, cosy)

        # Rotate direction from base_link → map
        dx_map = dx * math.cos(yaw) - dy * math.sin(yaw)
        dy_map = dx * math.sin(yaw) + dy * math.cos(yaw)

        # Compute final goal in map frame
        goal_x = self.robot_pose.position.x + dx_map * dist
        goal_y = self.robot_pose.position.y + dy_map * dist

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = goal_x
        goal_msg.pose.position.y = goal_y
        goal_msg.pose.orientation.w = 1.0

        self.get_logger().info(
            f"Publishing exploration goal wrt robot: ({goal_x:.2f}, {goal_y:.2f})"
        )
        self.goal_pub.publish(goal_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DirectionToGoal()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

