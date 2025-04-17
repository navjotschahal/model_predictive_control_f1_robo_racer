#!/usr/bin/env python3
# MIT License

# Copyright (c) Hongrui Zheng, Johannes Betz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Utility functions for Kinematic Single Track MPC waypoint tracker

Author: Hongrui Zheng, Johannes Betz, Ahmad Amine
Last Modified: 12/27/22
"""
import math
import numpy as np
from numba import njit

# imports for vizualization
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Float32MultiArray
from geometry_msgs.msg import Point

@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    projections = trajectory[:-1,:] + (t*diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    print(quaternion)
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
  
def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return [x, y, z, w]

def smooth_waypoints(self, waypoints=None):
    """
    Smooth waypoints using B-spline interpolation to create a smoother trajectory.
    This function also calculates yaw angles as the heading towards the next waypoint.
    """
    from scipy.interpolate import splprep, splev
    
    if waypoints is None:
        waypoints = self.waypoints
    else:
        waypoints = np.array(waypoints)
    
    # Extract x and y coordinates only - ignore potentially faulty yaw values
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    
    # Create a 2D B-spline (periodic if track is closed)
    tck, u = splprep([x, y], s=2.0, k=3, per=True)  # k=3 for cubic spline
    
    # Evaluate the B-spline at evenly spaced points
    t_smooth = np.linspace(0, 1, 2200)
    x_smooth, y_smooth = splev(t_smooth, tck)
    
    # Create new waypoints array with smoothed x, y
    smoothed_waypoints = np.zeros((2200, 4))  # [x, y, heading, velocity]
    smoothed_waypoints[:, 0] = x_smooth
    smoothed_waypoints[:, 1] = y_smooth
    
    # Recalculate headings based on smoothed path
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    smoothed_waypoints[:, 2] = np.arctan2(dy, dx)
    
    # If velocity is included, recalculate based on curvature
    if waypoints.shape[1] >= 4:
        # Recalculate velocities based on new curvature
        velocities = calculate_velocity_profile(self, smoothed_waypoints[:, :3])
        smoothed_waypoints[:, 3] = velocities
    
    return smoothed_waypoints

def load_waypoints(self, file_path):
        """
        Load waypoints from CSV file and add velocity profile based on curvature
        
        CSV format should be: x,y,heading
        Returns: numpy array of [x, y, heading, velocity]
        """
        import csv
        import os
        from ament_index_python.packages import get_package_share_directory
        
        self.get_logger().info(f'Loading waypoints from: {file_path}')
        
        # Check if file exists with absolute path or relative to package
        if os.path.isfile(file_path):
            csv_path = file_path
        else:
            # Try to find relative to package
            package_share_directory = get_package_share_directory('mpc')
            self.get_logger().info(f'Looking for waypoints in package share directory: {package_share_directory}')
            csv_path = package_share_directory + file_path
            if not os.path.isfile(csv_path):
                self.get_logger().error(f'Could not find waypoint file: {csv_path}')
                raise FileNotFoundError(f"Waypoint file not found: {csv_path}")
        
        waypoints = []
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) >= 3:  # Ensure there are at least x, y, heading
                    waypoints.append([float(row[0]), float(row[1]), float(row[2])])
        
        if not waypoints:
            self.get_logger().error('No waypoints found in file')
            raise ValueError("No waypoints loaded from file")
        
        # Convert to numpy array
        waypoints = np.array(waypoints)
        
        # Calculate curvature and add velocity based on it
        velocities = calculate_velocity_profile(self, waypoints)
        
        # Add velocity as fourth column
        full_waypoints = np.column_stack((waypoints, velocities))
        
        self.get_logger().info(f'Loaded {len(full_waypoints)} waypoints')
        return full_waypoints

# testing may not use it need to see perf compared to linear speed calculator with curvature
def calculate_velocity_profile(self, waypoints):
        """
        Calculate velocity profile based on path curvature
        Higher curvature = lower speed, lower curvature = higher speed
        """
        # Extract x, y coordinates
        x = waypoints[:, 0]
        y = waypoints[:, 1]
        
        # Calculate the differences
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        # Calculate second derivatives
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # Calculate curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**(3/2)
        
        # Replace NaNs or Infs with zero curvature
        curvature = np.nan_to_num(curvature)
        
        # Map curvature to velocity: higher curvature -> lower velocity
        # You can adjust these values as needed for your track
        max_velocity = self.config.MAX_SPEED  # Use your configured max speed
        min_velocity = 1.0  # Minimum velocity at sharpest turns
        
        # Normalize curvature to [0, 1] range considering outliers
        # Use percentile to avoid extreme values affecting the scaling
        curvature_min = np.percentile(curvature, 5)  # 5th percentile
        curvature_max = np.percentile(curvature, 95)  # 95th percentile
        
        # Ensure we don't divide by zero
        if curvature_max - curvature_min < 1e-10:
            normalized_curvature = np.zeros_like(curvature)
        else:
            # Clip and normalize curvature
            clipped_curvature = np.clip(curvature, curvature_min, curvature_max)
            normalized_curvature = (clipped_curvature - curvature_min) / (curvature_max - curvature_min)
        
        # Calculate velocity: v = max_v - (max_v - min_v) * normalized_curvature
        velocity = max_velocity - (max_velocity - min_velocity) * normalized_curvature
        
        return velocity