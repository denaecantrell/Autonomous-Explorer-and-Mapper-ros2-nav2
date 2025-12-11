import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np


class FrontierVisualizer(Node):
    def __init__(self):
        super().__init__('frontier_visualizer')

        self.get_logger().info("Frontier Visualizer Node Started")

        # Subscribe to map
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        # Publish marker array
        self.marker_pub = self.create_publisher(
            MarkerArray, '/frontier_markers', 10
        )

        self.map_data = None

    def map_callback(self, msg):
        self.map_data = msg

        # convert the map into array
        grid = np.array(msg.data).reshape(
            msg.info.height, msg.info.width
        )

        # find frontiers
        frontiers = self.detect_frontiers(grid)

        # publish markers
        self.publish_markers(frontiers, msg)

    def detect_frontiers(self, grid):
        frontiers = []
        rows, cols = grid.shape

        # free cell = 0, unknown = -1
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if grid[r, c] == 0:  # free
                    neighbors = grid[r-1:r+2, c-1:c+2].flatten()
                    if -1 in neighbors:
                        frontiers.append((r, c))

        self.get_logger().info(f"Frontiers detected: {len(frontiers)}")
        return frontiers

    def publish_markers(self, frontiers, map_msg):
        marker_array = MarkerArray()
        resolution = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y

        marker_id = 0

        for (r, c) in frontiers:
            wx = c * resolution + origin_x
            wy = r * resolution + origin_y

            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "frontiers"
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD

            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = 0.05

            m.scale.x = 0.15
            m.scale.y = 0.15
            m.scale.z = 0.15

            # ORANGE
            m.color.r = 1.0
            m.color.g = 0.55
            m.color.b = 0.0
            m.color.a = 1.0

            marker_array.markers.append(m)
            marker_id += 1

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

