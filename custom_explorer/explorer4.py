import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav_msgs.msg import OccupancyGrid, Path  # <<< CHANGED (Path added)
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, ComputePathToPose
import numpy as np


class ExplorerNode(Node):
    def __init__(self):
        super().__init__('explorer')
        self.get_logger().info("Explorer Node Started")

        # Subscriber to the SLAM /map
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)

        # Use Nav2 global costmap to better reject obstacle goals
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap', self.costmap_callback, 10)

        # User exploration / navigation goal
        # The user publishes a PoseStamped in /exploration_goal.
        # If reachable, we go there (navigation mode).
        # If not reachable / outside map, we bias exploration in that direction.
        self.user_goal_sub = self.create_subscription(
            PoseStamped, '/exploration_goal', self.exploration_goal_callback, 10
        )

        # Action client for navigation (execute path)
        self.nav_to_pose_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose')

        # Action client to compute path, used as a filter
        self.compute_path_client = ActionClient(
            self, ComputePathToPose, 'compute_path_to_pose')

        # Path publisher so we can visualize planned path in RViz
        self.path_pub = self.create_publisher(Path, '/explorer/planned_path', 10)

        # Visited / tested frontiers
        self.visited_frontiers = set()

        # Map and costmap data
        self.map_data: OccupancyGrid | None = None
        self.costmap_data: OccupancyGrid | None = None  # <<< CHANGED

        # Last user goal (PoseStamped) and mode tracking
        self.user_goal: PoseStamped | None = None
        self.current_mode: str = "exploration"  # 'exploration' or 'navigation'

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

        # Crude fallback robot position at map center if unknown
        h = msg.info.height
        w = msg.info.width
        self.robot_position = (h // 2, w // 2)

    # <<< CHANGED: store Nav2 global costmap
    def costmap_callback(self, msg: OccupancyGrid):
        self.costmap_data = msg
        # self.get_logger().info("Costmap received")

    # <<< ADDED: store latest user goal (used for nav OR directional bias)
    def exploration_goal_callback(self, msg: PoseStamped):
        self.user_goal = msg
        self.get_logger().info(
            f"Received user exploration goal at "
            f"x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}"
        )

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
    def choose_frontier(self, frontiers, map_array, preferred_direction: np.ndarray | None = None):
        """
        Choose a frontier that:

        - wasn't tried before
        - is not inside / directly on top of an obstacle
        - is biased towards the preferred_direction (if given)
          while still respecting distance.
        """
        if not frontiers:
            return None

        robot_row, robot_col = self.robot_position
        min_cost = float('inf')
        chosen_frontier = None

        # <<< ADDED: if we want to bias in a direction, we need map info
        info = self.map_data.info if self.map_data is not None else None
        res = info.resolution if info is not None else 1.0

        # direction vs distance weighting
        direction_weight = 5.0   # bigger = more strongly follow user direction
        distance_weight = 1.0

        for frontier in frontiers:
            if frontier in self.visited_frontiers:
                continue

            r, c = frontier

            # quick map-space obstacle check (extra safety)
            if map_array[r, c] != 0:
                continue

            # prefer frontier cells not immediately adjacent to occupied (buffer)
            # <<< CHANGED: use slightly larger radius to avoid "frontiers in obstacles"
            rr_min = max(0, r - 3)
            rr_max = min(map_array.shape[0], r + 3)
            cc_min = max(0, c - 3)
            cc_max = min(map_array.shape[1], c + 3)
            neighborhood = map_array[rr_min:rr_max, cc_min:cc_max]
            if (neighborhood >= 50).any():  # occupied neighbors -> skip
                continue

            # base cost: grid-distance
            distance_cells = np.hypot(robot_row - r, robot_col - c)
            cost = distance_weight * distance_cells

            # directional bias
            if preferred_direction is not None and info is not None:
                # convert robot index position to world (roughly)
                robot_x = info.origin.position.x + robot_col * res
                robot_y = info.origin.position.y + robot_row * res
                frontier_x = info.origin.position.x + c * res
                frontier_y = info.origin.position.y + r * res

                f_vec = np.array(
                    [frontier_x - robot_x, frontier_y - robot_y],
                    dtype=float
                )
                f_norm = np.linalg.norm(f_vec)
                if f_norm > 1e-3:
                    f_dir = f_vec / f_norm
                    cos_sim = np.clip(np.dot(f_dir, preferred_direction), -1.0, 1.0)
                    # 0 when aligned, ~2 when opposite
                    angle_penalty = 1.0 - cos_sim
                    cost = distance_weight * distance_cells + direction_weight * angle_penalty

            if cost < min_cost:
                min_cost = cost
                chosen_frontier = frontier

        if chosen_frontier is None:
            self.get_logger().warning("No valid frontier found after filtering")
        else:
            self.get_logger().info(
                f"Chosen frontier (grid): {chosen_frontier}, "
                f"cost={min_cost:.2f}, mode={self.current_mode}"
            )

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

    # <<< ADDED: helper to publish Nav2 path so the dog’s path is highlighted in RViz
    def publish_path(self, nav2_path: Path):
        if nav2_path is None or self.path_pub is None:
            return
        # Make sure we have a fresh timestamp
        nav2_path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(nav2_path)

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
        # <<< CHANGED: use a slightly larger inflation radius to avoid "frontiers in obstacles"
        if self.is_in_obstacle_on_costmap(goal_x, goal_y, inflation_cells=2) or \
           self.is_in_obstacle_on_map(goal_x, goal_y, inflation_cells=2):
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

        # <<< ADDED: always publish planned path so it’s highlighted as the dog moves
        self.publish_path(result.path)

        self.get_logger().info(
            f"Frontier {frontier} reachable, path length {len(result.path.poses)}. "
            f"Sending NavigateToPose goal.")
        self.navigate_to(goal_x, goal_y)

    # <<< ADDED: planning for direct user goal (navigation mode)
    def plan_and_navigate_to_user_goal(self, goal_x: float, goal_y: float):
        # reject if obviously in obstacle
        if self.is_in_obstacle_on_costmap(goal_x, goal_y, inflation_cells=2) or \
           self.is_in_obstacle_on_map(goal_x, goal_y, inflation_cells=2):
            self.get_logger().info(
                "User goal in obstacle/inflation region; cannot navigate directly."
            )
            return False

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = goal_x
        goal_pose.pose.position.y = goal_y
        goal_pose.pose.orientation.w = 1.0

        cpp_goal = ComputePathToPose.Goal()
        cpp_goal.goal = goal_pose
        cpp_goal.use_start = False

        self.get_logger().info(
            f"[NAVIGATION MODE] Checking path to user goal at x={goal_x:.2f}, y={goal_y:.2f}"
        )

        self.compute_path_client.wait_for_server()
        send_future = self.compute_path_client.send_goal_async(cpp_goal)

        send_future.add_done_callback(
            lambda f, gx=goal_x, gy=goal_y:
            self.user_goal_compute_path_goal_response_callback(f, gx, gy)
        )
        return True

    def user_goal_compute_path_goal_response_callback(self, future, goal_x, goal_y):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning(
                "ComputePathToPose goal for user goal was rejected"
            )
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f, gx=goal_x, gy=goal_y:
            self.user_goal_compute_path_result_callback(f, gx, gy)
        )

    def user_goal_compute_path_result_callback(self, future, goal_x, goal_y):
        try:
            result = future.result().result
        except Exception as e:
            self.get_logger().error(
                f"ComputePathToPose failed for user goal: {e}"
            )
            return

        if result.error_code != 0 or len(result.path.poses) == 0:
            self.get_logger().info(
                f"User goal unreachable: "
                f"error_code={result.error_code}, path length={len(result.path.poses)}"
            )
            return

        # <<< ADDED: publish path for navigation mode too
        self.publish_path(result.path)

        self.get_logger().info(
            f"User goal reachable, path length {len(result.path.poses)}. "
            f"Sending NavigateToPose goal."
        )
        self.navigate_to(goal_x, goal_y)

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    # <<< ADDED: compute direction vector from robot to user goal (for biasing frontiers)
    def compute_preferred_direction_from_user_goal(self) -> np.ndarray | None:
        if self.user_goal is None or self.map_data is None:
            return None

        info = self.map_data.info
        res = info.resolution
        robot_row, robot_col = self.robot_position

        robot_x = info.origin.position.x + robot_col * res
        robot_y = info.origin.position.y + robot_row * res

        gx = self.user_goal.pose.position.x
        gy = self.user_goal.pose.position.y

        vec = np.array([gx - robot_x, gy - robot_y], dtype=float)
        norm = np.linalg.norm(vec)
        if norm < 1e-3:
            return None

        return vec / norm

    # <<< ADDED: check if user goal is inside current map frame
    def user_goal_in_map_bounds(self) -> bool:
        if self.user_goal is None or self.map_data is None:
            return False

        gx = self.user_goal.pose.position.x
        gy = self.user_goal.pose.position.y
        info = self.map_data.info
        res = info.resolution

        col = int((gx - info.origin.position.x) / res)
        row = int((gy - info.origin.position.y) / res)

        if row < 0 or col < 0 or row >= info.height or col >= info.width:
            return False
        return True

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

        # ------------------------------------------------------------------
        # 1) Navigation mode: if user goal is in frame AND reachable, go there.
        # ------------------------------------------------------------------
        used_navigation_mode = False
        if self.user_goal is not None and self.user_goal_in_map_bounds():
            self.current_mode = "navigation"
            gx = self.user_goal.pose.position.x
            gy = self.user_goal.pose.position.y
            self.get_logger().info(
                f"[NAVIGATION MODE] User goal is inside map bounds, trying direct nav."
            )
            used_navigation_mode = self.plan_and_navigate_to_user_goal(gx, gy)

        # If navigation mode succeeded in sending a goal, skip frontier exploration
        if used_navigation_mode:
            return

        # ------------------------------------------------------------------
        # 2) Exploration mode: expand map, biased by user direction if available.
        # ------------------------------------------------------------------
        self.current_mode = "exploration"

        # Detect frontiers
        frontiers = self.find_frontiers(map_array)

        if not frontiers:
            self.get_logger().info("No frontiers found. Exploration complete!")
            # self.shutdown_robot()
            return

        # Directional bias from user goal (if they gave one, even if unreachable)
        preferred_direction = self.compute_preferred_direction_from_user_goal()

        # Choose filtered frontier, biased towards desired direction
        chosen_frontier = self.choose_frontier(frontiers, map_array, preferred_direction)

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

