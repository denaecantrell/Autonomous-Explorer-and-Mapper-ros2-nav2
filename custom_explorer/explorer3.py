import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, ComputePathToPose  # <<< CHANGED
import numpy as np


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer')
        self.get_logger().info("Explorer Node Started")

        # Subscriber to the SLAM /map
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # <<< CHANGED: optional – use Nav2 global costmap to better reject obstacle goals
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, 10)

        # Action client for navigation (execute path)
        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        # <<< CHANGED: action client to compute path, used as a filter
        self.compute_path_client = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose')

        # Visited / tested frontiers
        self.visited_frontiers = set()

        # Map and costmap data
        self.map_data: OccupancyGrid | None = None
        self.costmap_data: OccupancyGrid | None = None  # <<< CHANGED

        # Robot position in map indices (row, col)
        # TODO: replace with pose from localization (e.g. /amcl_pose or TF)
        self.robot_position = (0, 0)

        # Timer for periodic exploration
        self.timer = self.create_timer(5.0, self.explore)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid):
        self.map_data = msg
        self.get_logger().info("Map received")

        # <<< CHANGED: crude fallback robot position at map center if unknown
        h = msg.info.height
        w = msg.info.width
        self.robot_position = (h // 2, w // 2)

    # <<< CHANGED: store Nav2 global costmap
    def costmap_callback(self, msg: OccupancyGrid):
        self.costmap_data = msg
        # self.get_logger().info("Costmap received")

    # ------------------------------------------------------------------
    # Helpers: occupancy / cost checks
    # ------------------------------------------------------------------

    # <<< CHANGED: check if world point is in/near an obstacle using /map
    def is_in_obstacle_on_map(self, x: float, y: float, inflation_cells: int = 1) -> bool:
        if self.map_data is None:
            return False  # if we don't know, don't block

        info = self.map_data.info
        res = info.resolution

        col = int((x - info.origin.position.x) / res)
        row = int((y - info.origin.position.y) / res)

        if row < 0 or col < 0 or row >= info.height or col >= info.width:
            return True  # outside map, treat as obstacle

        # look in a small neighborhood for occupied cells
        width = info.width
        data = self.map_data.data
        for r in range(max(0, row - inflation_cells), min(info.height, row + inflation_cells + 1)):
            for c in range(max(0, col - inflation_cells), min(info.width, col + inflation_cells + 1)):
                idx = r * width + c
                val = data[idx]
                if val >= 50:  # occupancy threshold; 100 is definitely an obstacle
                    return True
        return False

    # <<< CHANGED: check Nav2 global costmap (closer to what Nav2 will reject)
    def is_in_obstacle_on_costmap(self, x: float, y: float, inflation_cells: int = 1) -> bool:
        if self.costmap_data is None:
            # If we don't have the costmap yet, fall back to map-only check later
            return False

        info = self.costmap_data.info
        res = info.resolution

        col = int((x - info.origin.position.x) / res)
        row = int((y - info.origin.position.y) / res)

        if row < 0 or col < 0 or row >= info.height or col >= info.width:
            return True

        width = info.width
        data = self.costmap_data.data
        for r in range(max(0, row - inflation_cells), min(info.height, row + inflation_cells + 1)):
            for c in range(max(0, col - inflation_cells), min(info.width, col + inflation_cells + 1)):
                idx = r * width + c
                cost = data[idx]
                # Nav2 uses 0–255; >= 253 is lethal or unknown, treat as blocked
                if cost >= 253:
                    return True
        return False

    # ------------------------------------------------------------------
    # Frontier detection / selection
    # ------------------------------------------------------------------

    def find_frontiers(self, map_array: np.ndarray):
        """
        Detect frontiers in the occupancy grid map.
        Free cell (0) with at least one unknown (-1) neighbor.
        """
        frontiers = []
        rows, cols = map_array.shape

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if map_array[r, c] == 0:  # free cell
                    neighbors = map_array[r-1:r+2, c-1:c+2].flatten()
                    if -1 in neighbors:
                        frontiers.append((r, c))

        self.get_logger().info(f"Found {len(frontiers)} raw frontiers")
        return frontiers

    # <<< CHANGED: filter obviously bad frontiers before trying to plan
    def choose_frontier(self, frontiers, map_array):
        """
        Choose the closest frontier that:
        - wasn't tried before
        - is not inside / directly on top of an obstacle
        """
        if not frontiers:
            return None

        robot_row, robot_col = self.robot_position
        min_distance = float('inf')
        chosen_frontier = None

        for frontier in frontiers:
            if frontier in self.visited_frontiers:
                continue

            r, c = frontier

            # quick map-space obstacle check (extra safety)
            if map_array[r, c] != 0:
                continue

            # prefer frontier cells not immediately adjacent to occupied (buffer)
            neighborhood = map_array[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            if (neighborhood > 50).any():  # occupied neighbors -> skip
                continue

            distance = np.hypot(robot_row - r, robot_col - c)
            if distance < min_distance:
                min_distance = distance
                chosen_frontier = frontier

        if chosen_frontier is None:
            self.get_logger().warning("No valid frontier found after filtering")
        else:
            self.get_logger().info(f"Chosen frontier (grid): {chosen_frontier}")

        return chosen_frontier

    # ------------------------------------------------------------------
    # Nav2 interaction
    # ------------------------------------------------------------------

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

        self.get_logger().info(f"Navigating to goal: x={x:.2f}, y={y:.2f}")

        self.nav_to_pose_client.wait_for_server()
        send_goal_future = self.nav_to_pose_client.send_goal_async(nav_goal)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warning("NavigateToPose goal rejected!")
            return

        self.get_logger().info("NavigateToPose goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_complete_callback)

    def navigation_complete_callback(self, future):
        try:
            result = future.result().result
            self.get_logger().info(f"Navigation completed with result: {result}")
        except Exception as e:
            self.get_logger().error(f"Navigation failed: {e}")

    # <<< CHANGED: compute-path filter before we actually call navigate_to()
    def validate_and_navigate_to_frontier(self, frontier, goal_x, goal_y):
        """
        Use Nav2's ComputePathToPose action to make sure this frontier is reachable.
        Only if a path is successfully planned and the goal isn't in an obstacle
        do we send a NavigateToPose goal.
        """

        # mark this frontier as attempted so we don't keep trying it
        self.visited_frontiers.add(frontier)

        # reject goals obviously in obstacles even before calling Nav2
        if self.is_in_obstacle_on_costmap(goal_x, goal_y) or \
           self.is_in_obstacle_on_map(goal_x, goal_y):
            self.get_logger().info(
                f"Frontier {frontier} rejected: inside obstacle / inflated region")
            return

        # Prepare ComputePathToPose goal
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.orientation.w = 1.0

        cpp_goal = ComputePathToPose.Goal()
        cpp_goal.goal = goal_pose
        cpp_goal.use_start = False  # use current robot pose as start

        self.get_logger().info(
            f"Checking path to frontier {frontier} at x={goal_x:.2f}, y={goal_y:.2f}")

        self.compute_path_client.wait_for_server()
        send_future = self.compute_path_client.send_goal_async(cpp_goal)

        # pass coordinates and frontier into callback via default args
        send_future.add_done_callback(
            lambda f, gx=goal_x, gy=goal_y, fr=frontier:
            self.compute_path_goal_response_callback(f, gx, gy, fr)
        )

    # <<< CHANGED: handle ComputePathToPose goal acceptance
    def compute_path_goal_response_callback(self, future, goal_x, goal_y, frontier):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning(
                f"ComputePathToPose goal for frontier {frontier} was rejected")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f, gx=goal_x, gy=goal_y, fr=frontier:
            self.compute_path_result_callback(f, gx, gy, fr)
        )

    # <<< CHANGED: handle ComputePathToPose result and decide whether to navigate
    def compute_path_result_callback(self, future, goal_x, goal_y, frontier):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(
                f"ComputePathToPose failed for frontier {frontier}: {e}")
            return

        # error_code 0 == NONE (success)
        if result.error_code != 0 or len(result.path.poses) == 0:
            self.get_logger().info(
                f"Frontier {frontier} unreachable: "
                f"error_code={result.error_code}, path length={len(result.path.poses)}"
            )
            return

        self.get_logger().info(
            f"Frontier {frontier} reachable, path length {len(result.path.poses)}. "
            f"Sending NavigateToPose goal.")
        self.navigate_to(goal_x, goal_y)

    # ------------------------------------------------------------------
    # Main exploration loop
    # ------------------------------------------------------------------

    def explore(self):
        if self.map_data is None:
            self.get_logger().warning("No map data available")
            return

        # Convert map to numpy array
        map_array = np.array(self.map_data.data, dtype=np.int8).reshape(
            (self.map_data.info.height, self.map_data.info.width))

        # Detect frontiers
        frontiers = self.find_frontiers(map_array)

        if not frontiers:
            self.get_logger().info("No frontiers found. Exploration complete!")
            # self.shutdown_robot()
            return

        # Choose the closest filtered frontier
        chosen_frontier = self.choose_frontier(frontiers, map_array)  # <<< CHANGED

        if not chosen_frontier:
            self.get_logger().warning("No frontiers to explore after filtering")
            return

        # Convert grid frontier to world coordinates
        r, c = chosen_frontier
        goal_x = c * self.map_data.info.resolution + self.map_data.info.origin.position.x
        goal_y = r * self.map_data.info.resolution + self.map_data.info.origin.position.y

        # <<< CHANGED: check path first, then navigate if it’s valid
        self.validate_and_navigate_to_frontier(chosen_frontier, goal_x, goal_y)


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

