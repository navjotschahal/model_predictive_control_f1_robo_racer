from geometry_msgs.msg import Point
import rclpy
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration
from utils import quaternion_from_euler


def visualize_waypoints(waypoints, publisher, frame_id="map", ns="waypoints", 
                       color=[0.0, 0.8, 0.2], scale=0.1, lifetime=0.0):
    """
    Visualize waypoints in RViz with heading arrows
    
    Args:
        waypoints: Nx4 numpy array [x, y, yaw, v]
        publisher: ROS publisher for MarkerArray
        frame_id: Frame ID for the markers
        ns: Namespace for the markers
        color: RGB color values as list [r, g, b], range 0-1
        scale: Size of the markers
        lifetime: Lifetime of the markers in seconds (0 = forever)
    """    
    marker_array = MarkerArray()
    
    # First create a line strip to connect all waypoints
    line_strip = Marker()
    line_strip.header.frame_id = frame_id
    import rclpy
    line_strip.header.stamp = rclpy.clock.Clock().now().to_msg()
    line_strip.ns = ns + "_line"
    line_strip.id = 0
    line_strip.type = Marker.LINE_STRIP
    line_strip.action = Marker.ADD
    line_strip.pose.orientation.w = 1.0
    
    line_strip.scale.x = scale / 2.0  # Line width
    line_strip.color.r = color[0]
    line_strip.color.g = color[1]
    line_strip.color.b = color[2]
    line_strip.color.a = 1.0
    
    marker_array.markers.append(line_strip)
    
    # Add sphere markers for each waypoint
    waypoint_markers = Marker()
    waypoint_markers.header.frame_id = frame_id
    waypoint_markers.header.stamp = rclpy.clock.Clock().now().to_msg()
    waypoint_markers.ns = ns + "_points"
    waypoint_markers.id = 1
    waypoint_markers.type = Marker.SPHERE_LIST
    waypoint_markers.action = Marker.ADD
    waypoint_markers.pose.orientation.w = 1.0
    
    waypoint_markers.scale.x = scale
    waypoint_markers.scale.y = scale
    waypoint_markers.scale.z = scale
    waypoint_markers.color.r = color[0]
    waypoint_markers.color.g = color[1]
    waypoint_markers.color.b = color[2]
    waypoint_markers.color.a = 1.0
    
    # Add each waypoint
    for i in range(len(waypoints)):
        p = Point()
        p.x = float(waypoints[i, 0])
        p.y = float(waypoints[i, 1])
        p.z = 0.1  # Slightly above ground
        waypoint_markers.points.append(p)
        line_strip.points.append(p)
    
    marker_array.markers.append(waypoint_markers)
    
    # Add arrow markers to show yaw/heading at each waypoint
    for i in range(len(waypoints)):
        arrow_marker = Marker()
        arrow_marker.header.frame_id = frame_id
        arrow_marker.header.stamp = rclpy.clock.Clock().now().to_msg()
        arrow_marker.ns = ns + "_headings"
        arrow_marker.id = i + 100  # Offset to avoid ID collision
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # Position
        arrow_marker.pose.position.x = float(waypoints[i, 0])
        arrow_marker.pose.position.y = float(waypoints[i, 1])
        arrow_marker.pose.position.z = 0.1  # Slightly above ground
        
        # Orientation from yaw angle
        yaw = float(waypoints[i, 2])
        q = quaternion_from_euler(0, 0, yaw)
        arrow_marker.pose.orientation.x = q[0]
        arrow_marker.pose.orientation.y = q[1]
        arrow_marker.pose.orientation.z = q[2]
        arrow_marker.pose.orientation.w = q[3]
        
        # Scale - arrow length and width
        arrow_marker.scale.x = scale * 4.0  # Arrow length
        arrow_marker.scale.y = scale * 0.5  # Arrow width
        arrow_marker.scale.z = scale * 0.5  # Arrow height
        
        # Color - slightly different to distinguish from waypoints
        arrow_marker.color.r = color[0] * 0.8
        arrow_marker.color.g = color[1] * 0.8
        arrow_marker.color.b = color[2] * 1.2
        arrow_marker.color.a = 1.0
        
        # Set lifetime if specified
        if lifetime > 0:
            arrow_marker.lifetime = Duration(seconds=lifetime).to_msg()
            
        marker_array.markers.append(arrow_marker)
    
    publisher.publish(marker_array)

def visualize_optimised_path(x_path, y_path, yaw_path, v_path, publisher):
    """
    Visualize the predicted path from MPC
    
    Args:
        x_path: x positions along path
        y_path: y positions along path
        yaw_path: yaw angles along path
        v_path: velocities along path
        publisher: ROS publisher for MarkerArray
    """
    marker_array = MarkerArray()
    
    # Line showing the predicted path
    line_marker = Marker()
    line_marker.header.frame_id = "map"
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.scale.x = 0.03  # Line width
    line_marker.color.a = 1.0
    line_marker.color.r = 0.0
    line_marker.color.g = 1.0  # Green
    line_marker.color.b = 0.0
    line_marker.id = 0
    line_marker.lifetime = Duration(seconds=0.1).to_msg()
    
    # Add points to the line and create orientation arrows
    for i in range(len(x_path)):
        # Add point to line
        p = Point()
        p.x = x_path[i]
        p.y = y_path[i]
        p.z = 0.1  # Slightly above ground
        line_marker.points.append(p)
    
    marker_array.markers.append(line_marker)
    publisher.publish(marker_array)


def visualize_predicted_path(state_predict, publisher):
    """
    Visualize the predicted path from MPC
    """
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    import rclpy
    
    marker_array = MarkerArray()
    
    # Create line strip marker for the path
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rclpy.clock.Clock().now().to_msg()
    marker.ns = "predicted_path"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    
    # Set points
    for i in range(state_predict.shape[1]):
        p = Point()
        p.x = state_predict[0, i]
        p.y = state_predict[1, i]
        p.z = 0.0
        marker.points.append(p)
    
    # Set scale
    marker.scale.x = 0.05  # line width
    
    # Set color (blue for predicted path)
    marker.color.r = 0.0
    marker.color.g = 0.5
    marker.color.b = 1.0
    marker.color.a = 1.0
    
    marker_array.markers.append(marker)
    
    # Add velocity markers (spheres with size according to velocity)
    for i in range(0, state_predict.shape[1], 2):  # Add markers at every other point
        v_marker = Marker()
        v_marker.header.frame_id = "map"
        v_marker.header.stamp = rclpy.clock.Clock().now().to_msg()
        v_marker.ns = "predicted_velocities"
        v_marker.id = i + 100  # Offset ids to avoid collision
        v_marker.type = Marker.SPHERE
        v_marker.action = Marker.ADD
        
        # Position
        v_marker.pose.position.x = state_predict[0, i]
        v_marker.pose.position.y = state_predict[1, i]
        v_marker.pose.position.z = 0.05
        
        # Set orientation (identity quaternion)
        v_marker.pose.orientation.x = 0.0
        v_marker.pose.orientation.y = 0.0
        v_marker.pose.orientation.z = 0.0
        v_marker.pose.orientation.w = 1.0
        
        # Scale based on velocity
        vel = state_predict[2, i]
        scale_factor = 0.1 + (vel / 10.0) * 0.1  # Scale between 0.1 and 0.2 based on velocity
        v_marker.scale.x = scale_factor
        v_marker.scale.y = scale_factor
        v_marker.scale.z = scale_factor
        
        # Color based on velocity (green to red gradient)
        norm_vel = min(1.0, vel / 6.0)  # Normalize velocity to [0,1]
        v_marker.color.r = norm_vel
        v_marker.color.g = 1.0 - norm_vel
        v_marker.color.b = 0.0
        v_marker.color.a = 0.7
        
        marker_array.markers.append(v_marker)
    
    publisher.publish(marker_array)


def visualize_reference_trajectory(ref_traj, publisher):
    marker_array = MarkerArray()
    
    # Line strip for trajectory path
    line_marker = Marker()
    line_marker.header.frame_id = "map"
    line_marker.header.stamp = rclpy.clock.Clock().now().to_msg()
    line_marker.ns = "reference_trajectory"
    line_marker.id = 0
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.scale.x = 0.05  # Line width
    
    # Set pink color
    line_marker.color.r = 1.0
    line_marker.color.g = 0.75
    line_marker.color.b = 0.8
    line_marker.color.a = 1.0  # Full opacity
    
    for i in range(ref_traj.shape[1]):
        p = Point()
        p.x = ref_traj[0, i]
        p.y = ref_traj[1, i]
        p.z = 0.3
        line_marker.points.append(p)
    
    marker_array.markers.append(line_marker)
    
    # Add arrow markers to show yaw/heading at regular intervals
    arrow_interval = max(1, ref_traj.shape[1] // 15)  # Show ~15 arrows along the path
    
    for i in range(0, ref_traj.shape[1], arrow_interval):
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "map"
        arrow_marker.header.stamp = rclpy.clock.Clock().now().to_msg()
        arrow_marker.ns = "reference_trajectory_headings"
        arrow_marker.id = i + 100  # Offset to avoid ID collision
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # Position
        arrow_marker.pose.position.x = ref_traj[0, i]
        arrow_marker.pose.position.y = ref_traj[1, i]
        arrow_marker.pose.position.z = 0.3  # Same height as trajectory
        
        # Orientation from yaw angle
        yaw = ref_traj[2, i]
        q = quaternion_from_euler(0, 0, yaw)
        arrow_marker.pose.orientation.x = q[0]
        arrow_marker.pose.orientation.y = q[1]
        arrow_marker.pose.orientation.z = q[2]
        arrow_marker.pose.orientation.w = q[3]
        
        # Scale - arrow length and width
        arrow_marker.scale.x = 0.3  # Arrow length
        arrow_marker.scale.y = 0.05  # Arrow width
        arrow_marker.scale.z = 0.05  # Arrow height
        
        # Color - slightly different shade of pink
        arrow_marker.color.r = 1.0
        arrow_marker.color.g = 0.6
        arrow_marker.color.b = 0.9
        arrow_marker.color.a = 1.0
        
        marker_array.markers.append(arrow_marker)
    
    publisher.publish(marker_array)

def visualize_nearest_point(self, vehicle_state, ref_x, ref_y, ind):
    """
    Visualize the nearest point on the reference trajectory to current vehicle position
    """
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    
    if not hasattr(self, 'nearest_point_pub'):
        self.nearest_point_pub = self.create_publisher(MarkerArray, '/mpc/nearest_point', 10)
    
    marker_array = MarkerArray()
    
    # Nearest point marker
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = "nearest_point"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    
    # Set position
    marker.pose.position.x = ref_x[ind]
    marker.pose.position.y = ref_y[ind]
    marker.pose.position.z = 0.1  # Slightly above ground
    
    # Set orientation (identity quaternion)
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    
    # Set scale
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    
    # Set color (bright red)
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    
    marker_array.markers.append(marker)
    
    # Line connecting vehicle to nearest point
    line_marker = Marker()
    line_marker.header.frame_id = "map"
    line_marker.header.stamp = self.get_clock().now().to_msg()
    line_marker.ns = "nearest_point_line"
    line_marker.id = 1
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    
    # Add points for line
    p1 = Point()
    p1.x = vehicle_state.x
    p1.y = vehicle_state.y
    p1.z = 0.05
    
    p2 = Point()
    p2.x = ref_x[ind]
    p2.y = ref_y[ind]
    p2.z = 0.05
    
    line_marker.points = [p1, p2]
    
    # Set scale
    line_marker.scale.x = 0.05  # line width
    
    # Set color (yellow)
    line_marker.color.r = 1.0
    line_marker.color.g = 1.0
    line_marker.color.b = 0.0
    line_marker.color.a = 1.0
    
    marker_array.markers.append(line_marker)
    
    self.nearest_point_pub.publish(marker_array)

