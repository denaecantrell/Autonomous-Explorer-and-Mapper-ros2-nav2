import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import numpy as np
from sklearn.cluster import DBSCAN


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer')
        self.get_logger().info("Explorer Node Started")

        # Subscriber to the map topic
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # Action client for navigation
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Visited frontiers set
        self.visited_frontiers = set()


        # Publish marker array
        self.marker_pub = self.create_publisher(
            MarkerArray, '/frontier_markers', 10
        )

        # Publish marker array
        self.filtered_marker_pub = self.create_publisher(
            MarkerArray, '/filtered_frontier_markers', 10
        )

        # Map and position data
        self.map_data = None
        self.robot_position = (0, 0)  # Placeholder, update from localization
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer for periodic exploration
        self.timer = self.create_timer(5.0, self.explore)

    def map_callback(self, msg):
        self.map_data = msg
        #self.get_logger().info("Map received")

    def navigate_to(self, x, y):
        """
        Send navigation goal to Nav2.
        """
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = 'map'
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.orientation.w = 1.0  # Facing forward

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = goal_msg

        self.get_logger().info(f"Navigating to goal: x={x}, y={y}")

        # Wait for the action server
        self.nav_to_pose_client.wait_for_server()

        # Send the goal and register a callback for the result
        send_goal_future = self.nav_to_pose_client.send_goal_async(nav_goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handle the goal response and attach a callback to the result.
        """
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warning("Goal rejected!")
            return

        self.get_logger().info("Goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_complete_callback)

    def navigation_complete_callback(self, future):
        """
        Callback to handle the result of the navigation action.
        """
        try:
            result = future.result().result
            self.get_logger().info(f"Navigation completed with result: {result}")
        except Exception as e:
            self.get_logger().error(f"Navigation failed: {e}")

    def find_frontiers(self, map_array):
        """
        Detect frontiers in the occupancy grid map.
        """
        frontiers = []
        rows, cols = map_array.shape

        # Iterate through each cell in the map
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if map_array[r, c] == 0:  # Free cell
                    # Check if any neighbors are unknown
                    neighbors = map_array[r-1:r+2, c-1:c+2].flatten()
                    if -1 in neighbors:
                        frontiers.append((r, c))

        self.get_logger().info(f"Found {len(frontiers)} frontiers")
        return frontiers

    def filter_frontiers_dbscan(self, frontiers, eps=2.0, min_samples=5, min_cluster_size=10):
        """
        Use DBSCAN to remove very small noisy frontier clusters.
        - eps: distance threshold in grid cells
        - min_samples: minimum density requirement
        - min_cluster_size: clusters smaller than this are removed
        """
        if len(frontiers) == 0:
            return []

        # Convert to numpy array
        points = np.array(frontiers)

        # Run DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = db.labels_

        # Count points per cluster
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique, counts))

        # Filter out noise (-1) and very small clusters
        filtered = [
            tuple(points[i])
            for i in range(len(points))
            if labels[i] != -1 and cluster_sizes[labels[i]] >= min_cluster_size
        ]

        self.get_logger().info(
            f"DBSCAN filtered {len(frontiers)} → {len(filtered)} frontiers "
            f"(clusters kept: {[c for c in cluster_sizes.values() if c >= min_cluster_size]})"
        )
        return filtered

    def choose_frontier(self, frontiers):
        """
        Choose the closest frontier to the robot.
        """
        #robot_row, robot_col = self.robot_position
        trans = self.tf_buffer.lookup_transform(
            "odom",   # target frame (where you want the result)
            "utlidar_lidar",            # source frame
            rclpy.time.Time()
        )
        x, y = trans.transform.translation.x, trans.transform.translation.y
        info = self.map_data.info
        res = info.resolution
        robot_col = int((x - info.origin.position.x) / res)
        robot_row = int((y - info.origin.position.y) / res)
        min_distance = float('inf')
        chosen_frontier = None
        

        for frontier in frontiers:
            if frontier in self.visited_frontiers:
                continue

            distance = np.sqrt((robot_row - frontier[0])**2 + (robot_col - frontier[1])**2)
            if distance < min_distance:
                min_distance = distance
                chosen_frontier = frontier

        if chosen_frontier:
            self.visited_frontiers.add(chosen_frontier)
            self.get_logger().info(f"Chosen frontier: {chosen_frontier}")
        else:
            self.get_logger().warning("No valid frontier found")

        return chosen_frontier
    
    # Check if world point is in/near an obstacle using /map
    def is_in_obstacle_on_map(self, row: int, col: int, inflation_cells: int = 1) -> bool:
        if self.map_data is None:
            return False  # if we don't know, don't block

        info = self.map_data.info
        data = self.map_data.data

        if row < 0 or col < 0 or row >= info.height or col >= info.width:
            return True  # outside map, treat as obstacle

        # look in a small neighborhood for occupied cells
        for r in range(max(0, row - inflation_cells), min(info.height, row + inflation_cells + 1)):
            for c in range(max(0, col - inflation_cells), min(info.width, col + inflation_cells + 1)):
                idx = r * info.width + c
                if data[idx] >= 50:
                    return True
        return False



    def explore(self):
        self.get_logger().info("HELLO")
        if self.map_data is None:
            self.get_logger().warning("No map data available")
            return

        # Convert map to numpy array
        map_array = np.array(self.map_data.data).reshape(
            (self.map_data.info.height, self.map_data.info.width))

        # Detect frontiers
        frontiers = self.find_frontiers(map_array)

        # Choose the closest frontier
        #filtered_frontiers = [point for point in frontiers if not self.is_in_obstacle_on_map(point[0], point[1], 5)]
        # First remove frontiers near obstacles (your existing logic)
        filtered_frontiers = [
            point for point in frontiers
            if not self.is_in_obstacle_on_map(point[0], point[1], 5)
        ]

        # Then apply DBSCAN noise removal
        filtered_frontiers = self.filter_frontiers_dbscan(
            filtered_frontiers,
            eps=2.5,           # distance in grid cells
            min_samples=5,     # density requirement
            min_cluster_size=10  # remove clusters smaller than 15 points
        )

        if not filtered_frontiers:
            self.get_logger().info("No frontiers found. Exploration complete!")
            # self.shutdown_robot()
            return

        chosen_frontier = self.choose_frontier(filtered_frontiers)

        if not chosen_frontier:
            self.get_logger().warning("No frontiers to explore")
            return

        # Filtering frontiers
        self.publish_markers(filtered_frontiers, self.map_data)

        # Convert the chosen frontier to world coordinates
        goal_x = chosen_frontier[1] * self.map_data.info.resolution + self.map_data.info.origin.position.x
        goal_y = chosen_frontier[0] * self.map_data.info.resolution + self.map_data.info.origin.position.y


        self.publish_goal_marker(goal_x, goal_y, self.map_data)
 
        # Navigate to the chosen frontier
        self.navigate_to(goal_x, goal_y)

    # def shudown_robot(self):
    #     
    #
    #
    #     self.get_logger().info("Shutting down robot exploration")
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

    def publish_goal_marker(self, x, y, map_msg):
        marker_array = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "user_goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.1

        m.scale.x = 0.25
        m.scale.y = 0.25
        m.scale.z = 0.25

        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0

        marker_array.markers.append(m)
        self.filtered_marker_pub.publish(marker_array)



def main(args=None):
    rclpy.init(args=args)
    explorer_node = ExplorerNode()

    try:
        explorer_node.get_logger().info("Starting exploration...")
        rclpy.spin(explorer_node)
    except KeyboardInterrupt:
        explorer_node.get_logger().info("Exploration stopped by user")
    finally:
        explorer_node.destroy_node()
        rclpy.shutdown()
