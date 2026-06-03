#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
import transforms3d.euler as t3d_euler

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import math
import numpy as np
import time
import json

from obstacle_detector import ObstacleDetector
from global_planner import GlobalPlanner, SmoothPath
from safearea import SafeCorridor, SafeArea
from control import CoridorMPC, ReactiveFeedback


def transform_coord(p, pose, w2l=True):
    x, y, theta = pose
    c, s = math.cos(theta), math.sin(theta)
    px, py =  p
    if w2l:
        d = (px - x, py - y)
        px_n =  c*d[0] + s*d[1]
        py_n = -s*d[0] + c*d[1]
    else:
        px_n = c*px - s*py + x
        py_n = s*px + c*py + y
    return (px_n, py_n)


class NavigationNode:
    def __init__(self):
        rospy.init_node('navigation_node')
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        rospy.Subscriber('/front/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_callback)

        
        init_pos = rospy.get_param('~init_position')
        goal_pos = rospy.get_param('~goal_position')

        self.init_pose = init_pos
        self.goal = (goal_pos[0], goal_pos[1])
        
        # Placeholders for sensor data.
        self.marker_pub = rospy.Publisher('/obstacle_markers', MarkerArray, queue_size=1)
        self.maker_header_frame_id = "front_laser"

        self.lidar_ranges = None
        self.odom_data = None
        self.model_data = None
        
        self.obstacle_detector = None
        self.obstacles = None
        
        # r_safe = 0.2 #math.sqrt((0.420**2 + 0.310**2))/2
        self.global_planner = GlobalPlanner(r_safe=0.27, wp=1.0)
        self.smooth_path = SmoothPath(smoothing_distance=0.2, num_points=5)
        
        # self.safe_area = SafeArea(0.27, xlim=[-2,2], ylim=[-2,2])
        # self.recfeed = ReactiveFeedback(control_gain=2.5)

        self.safe_corr = SafeCorridor(r_safe=0.27, Lmax=0.4, h=2.5, ds=0.2)
        self.mpc = CoridorMPC(N=10, max_faces=50, max_poly=10, max_v=1.0, max_omega=1.5)
        
        
    def odom_callback(self, msg):
        self.odom_data = msg
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        q_ros = [orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z]
        roll, pitch, yaw = t3d_euler.quat2euler(q_ros, axes='sxyz')
        theta = yaw
        if theta < -math.pi:
            theta += 2*math.pi
        elif theta > math.pi:
            theta -= 2*math.pi
        self.pose = (x, y, theta)
    
    def model_callback(self, msg):
        self.model_data = msg

    def lidar_callback(self, msg):
        self.lidar_ranges = msg.ranges
        
        if self.obstacle_detector is None:
            # Use ROS message data instead of hardcoded degrees
            min_angle = msg.angle_min
            delta_angle = msg.angle_increment
            
            rospy.loginfo(f"Initializing ObstacleDetector with min_angle={min_angle:.2f}, delta_angle={delta_angle:.4f}")
            self.obstacle_detector = ObstacleDetector(
                delta_angle=delta_angle, min_angle=min_angle,
                eps_l=0.06, dk=5, s_num=6,  p_min=4,  l_min=0.05,
            )

        try:
            self.obstacles = self.obstacle_detector(self.lidar_ranges)
            # print(f"{len(self.obstacles['points'])}, Nl: {len(self.obstacles['lines'][0])}, Nc: {len(self.obstacles['circles'][0])}")
        except Exception as e:
            rospy.logwarn(f"Obstacle extraction failed this frame: {e}")
    
    def control(self):
        self.rel_goal = transform_coord(self.goal, self.pose, w2l=True)
        rel_pose = (0.0, 0.0, 0.0)


        self.path = self.global_planner(rel_pose[:2], self.rel_goal, self.obstacles)
        if not self.path:
            rospy.logwarn(f"Path generation failed")
            velocity, omega = 0.0, 0.0
        else:            
            # self.path = self.smooth_path(self.path)

            P, P_poly = self.safe_corr(self.path, self.obstacles)
            try:
                # st = time.time()
                opt_U, self.pred_traj, dt = self.mpc.solve(rel_pose, P, P_poly)
                # print(time.time() - st, self.mpc.solver.get_stats('time_tot'), 
                #       'iter=', self.mpc.solver.get_stats('nlp_iter'),'\t')

                velocity = opt_U[0,0]
                omega = opt_U[0,1]
            except:
                rospy.logwarn(f"MPC failed")
                self.pred_traj = np.array([])
                velocity = 0.0
                omega = 0.0

            # self.path = np.asarray(self.path)
            # poly = self.safe_area(rel_pose[:2], self.obstacles)
            # outside_poly = (poly.A @ self.path.T > poly.b[:,None]).any(0)
            # if any(outside_poly) and len(self.path) > 3:
            #     curr_rel_goal = self.path[outside_poly][0]
            # else:
            #     curr_rel_goal = self.path[1]
            # velocity, omega = self.recfeed(rel_pose, curr_rel_goal, poly)

        # print(f'v:{velocity}, w:{omega}')
        return velocity, omega

    def publish_markers(self):
        marker_array = MarkerArray()
        marker_id = 0
        
        points = [self.rel_goal]
        if len(points) > 0:
            pts_marker = Marker()
            pts_marker.header.frame_id = self.maker_header_frame_id
            pts_marker.header.stamp = rospy.Time.now()
            pts_marker.ns = "extracted_g" 
            pts_marker.id = marker_id
            pts_marker.type = Marker.SPHERE_LIST
            pts_marker.action = Marker.ADD
            pts_marker.pose.orientation.w = 1.0

            pts_marker.scale.x = 0.15 
            pts_marker.scale.y = 0.15
            pts_marker.scale.z = 0.15

            pts_marker.color.r = 1.0
            pts_marker.color.g = 0.8
            pts_marker.color.b = 0.0
            pts_marker.color.a = 1.0 

            for pt in points:
                if not (math.isinf(pt[0]) or math.isnan(pt[0])):
                    pts_marker.points.append(Point(x=pt[0], y=pt[1], z=0.0))

            marker_array.markers.append(pts_marker)
            marker_id += 1

        path_marker = Marker()
        path_marker.header.frame_id = self.maker_header_frame_id
        path_marker.header.stamp = rospy.Time.now()
        path_marker.ns = "global_path"
        path_marker.id = marker_id
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.pose.orientation.w = 1.0
        path_marker.scale.x = 0.05 
        
        path_marker.color.r = 0.0
        path_marker.color.g = 1.0
        path_marker.color.b = 1.0
        path_marker.color.a = 1.0

        if self.path is not None:
            for pt in self.path:
                if math.isfinite(pt[0]) and math.isfinite(pt[1]):
                    path_marker.points.append(Point(x=pt[0], y=pt[1], z=0.0))

        marker_array.markers.append(path_marker)
        marker_id += 1

        # path_marker = Marker()
        # path_marker.header.frame_id = self.maker_header_frame_id
        # path_marker.header.stamp = rospy.Time.now()
        # path_marker.ns = "global_path"
        # path_marker.id = marker_id
        # path_marker.type = Marker.LINE_STRIP
        # path_marker.action = Marker.ADD
        # path_marker.pose.orientation.w = 1.0
        # path_marker.scale.x = 0.05 
        
        # path_marker.color.r = 1.0
        # path_marker.color.g = 0.0
        # path_marker.color.b = 1.0
        # path_marker.color.a = 1.0

        # if self.pred_traj is not None:
        #     for pt in self.pred_traj:
        #         if math.isfinite(pt[0]) and math.isfinite(pt[1]):
        #             path_marker.points.append(Point(x=pt[0], y=pt[1], z=0.0))

        # marker_array.markers.append(path_marker)
        # marker_id += 1
        
        if self.obstacles.segments:
            line_marker = Marker()
            line_marker.header.frame_id = self.maker_header_frame_id
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "extracted_lines"
            line_marker.id = marker_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD

            line_marker.pose.orientation.w = 1.0 
            
            line_marker.scale.x = 0.05 
            
            line_marker.color.r = 0.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 1.0

            for segment in self.obstacles.segments:
                p1, p2 = segment
                line_marker.points.append(Point(x=p1[0], y=p1[1], z=0.0))
                line_marker.points.append(Point(x=p2[0], y=p2[1], z=0.0))

            marker_array.markers.append(line_marker)
            marker_id += 1

        for circle in self.obstacles.circles:
            a, b, r = circle 
            
            circle_marker = Marker()
            circle_marker.header.frame_id = self.maker_header_frame_id
            circle_marker.header.stamp = rospy.Time.now()
            circle_marker.ns = "extracted_circles"
            circle_marker.id = marker_id
            circle_marker.type = Marker.CYLINDER
            circle_marker.action = Marker.ADD

            # Position
            circle_marker.pose.position.x = a
            circle_marker.pose.position.y = b
            circle_marker.pose.position.z = 0.0            
            circle_marker.pose.orientation.w = 1.0

            circle_marker.scale.x = 2.0 * r
            circle_marker.scale.y = 2.0 * r
            circle_marker.scale.z = 0.1

            circle_marker.color.r = 0.0
            circle_marker.color.g = 0.5
            circle_marker.color.b = 1.0
            circle_marker.color.a = 0.6

            marker_array.markers.append(circle_marker)
            marker_id += 1

        # Publish the array
        self.marker_pub.publish(marker_array)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():

            if self.lidar_ranges is None or self.odom_data is None:
                rate.sleep()
                continue

            velocity, omega = self.control()
            
            msg = Twist()
            msg.linear.x = velocity
            msg.angular.z = omega

            self.pub.publish(msg)
            self.publish_markers()
            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.loginfo("====START NAVIGATION====")
        node = NavigationNode()
        node.run()
    except rospy.ROSInterruptException:
        pass