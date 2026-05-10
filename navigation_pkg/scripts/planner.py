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

from obstacle_detector import ObstacleDetector
from voronoi import Voronoi, nearest_point_on_polytope, point_on_polytope_given_direction
from global_planner import GlobalPlanner

class NavigationNode:
    def __init__(self):
        rospy.init_node('navigation_node')
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        rospy.Subscriber('/front/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_callback)

        
        init_pos = rospy.get_param('~init_position')
        goal_pos = rospy.get_param('~goal_position')

        self.init_pos = np.array(init_pos[:2])
        self.init_theta = init_pos[2]
        self.goal = self.init_pos + np.array(goal_pos)
        
        # Other parameters.
        self.max_range = 4.0

        # Placeholders for sensor data.
        self.lidar_ranges = None
        self.odom_data = None
        self.model_data = None

        # Others
        self.marker_pub = rospy.Publisher('/obstacle_markers', MarkerArray, queue_size=1)
        self.lidar_frame_id = "front_laser"

        self.obstacle_detector = None
        self.obstacles = None
        
        safety_radius = 0.3 #math.sqrt((0.420**2 + 0.310**2))/2
        self.voronoi = Voronoi(pos=np.zeros(2), safety_radius=safety_radius, xlim=[-5,5], ylim=[-5,5])
        self.global_planner = GlobalPlanner(r_safe=safety_radius+0.1)
    
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
        self.pos = self.init_pos + np.array([x, y])
        self.theta = theta
    
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

    def control(self, k=0.5):
        polytope = self.voronoi.cell.poly
        pos, theta = np.zeros(2), 0
        R = np.array([
            [math.cos(self.theta), math.sin(self.theta)],
            [-math.sin(self.theta), math.cos(self.theta)]
        ])
        self.rel_goal = R @ (self.goal - self.pos)
        path = self.global_planner((0, 0), (self.rel_goal[0], self.rel_goal[1]), self.obstacles)
        self.curr_goal = np.array(path[1])

        g_dir = self.curr_goal - pos
        h_dir = np.array([math.cos(theta), math.sin(theta)])
        self._g = nearest_point_on_polytope(self.curr_goal, polytope, pos)
        self._gw = point_on_polytope_given_direction(pos, g_dir, polytope)
        self._gv = point_on_polytope_given_direction(pos, h_dir, polytope)
        
        hp_dir = np.array([-math.sin(theta), math.cos(theta)])
        q = pos - (self._g + self._gw)/2
        velocity = -k * h_dir @ (pos - self._gv)
        omega = k * math.atan((hp_dir @ q)/(h_dir @ q))
        # print(f'v:{velocity}, w:{omega}')
        return velocity, omega

    def publish_markers(self):
        marker_array = MarkerArray()
        marker_id = 0

        vertices = self.voronoi.cell.poly.vertices            
        if len(vertices) >= 3: # A polygon needs at least 3 points
            voro_marker = Marker()
            voro_marker.header.frame_id = self.lidar_frame_id
            voro_marker.header.stamp = rospy.Time.now()
            voro_marker.ns = "extracted_polytope"
            voro_marker.id = marker_id
            voro_marker.type = Marker.LINE_STRIP  # Connects points into a shape
            voro_marker.action = Marker.ADD
            voro_marker.pose.orientation.w = 1.0

            voro_marker.scale.x = 0.05 # Thickness of the polygon boundary

            # Color: Bright Magenta/Purple to stand out from obstacles
            voro_marker.color.r = 1.0
            voro_marker.color.g = 0.0
            voro_marker.color.b = 1.0
            voro_marker.color.a = 0.8 

            # Add all vertices to the marker
            for v in vertices:
                # Safety check against inf/NaN
                if not (math.isinf(v[0]) or math.isnan(v[0])):
                    voro_marker.points.append(Point(x=v[0], y=v[1], z=0.0))

            # Close the polygon loop by adding the first vertex at the end
            if len(voro_marker.points) > 0:
                first_point = voro_marker.points[0]
                voro_marker.points.append(first_point)

            marker_array.markers.append(voro_marker)
            marker_id += 1
        
        points = [self.rel_goal, self.curr_goal, self._g, self._gw, self._gv]
        if len(points) > 0:
            pts_marker = Marker()
            pts_marker.header.frame_id = self.lidar_frame_id
            pts_marker.header.stamp = rospy.Time.now()
            pts_marker.ns = "extracted_g" 
            pts_marker.id = marker_id
            pts_marker.type = Marker.SPHERE_LIST
            pts_marker.action = Marker.ADD
            pts_marker.pose.orientation.w = 1.0

            # Scale sets the diameter of the spheres in meters
            pts_marker.scale.x = 0.15 
            pts_marker.scale.y = 0.15
            pts_marker.scale.z = 0.15

            # Color: Bright Yellow / Gold
            pts_marker.color.r = 1.0
            pts_marker.color.g = 0.8
            pts_marker.color.b = 0.0
            pts_marker.color.a = 1.0 

            # Add all 4 points to the marker
            for pt in points:
                # Safety check against inf/NaN
                if not (math.isinf(pt[0]) or math.isnan(pt[0])):
                    pts_marker.points.append(Point(x=pt[0], y=pt[1], z=0.0))

            marker_array.markers.append(pts_marker)
            marker_id += 1

        
        Fl, Idl, BPl = self.obstacles.get('lines', ([], [], []))
        Fc, Idc = self.obstacles.get('circles', ([], []))
        if BPl:
            line_marker = Marker()
            line_marker.header.frame_id = self.lidar_frame_id
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "extracted_lines"
            line_marker.id = marker_id
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD

            line_marker.pose.orientation.w = 1.0 
            
            # Line thickness
            line_marker.scale.x = 0.05 
            
            # Color: Green
            line_marker.color.r = 0.0
            line_marker.color.g = 1.0
            line_marker.color.b = 0.0
            line_marker.color.a = 1.0

            # BPl contains lists of endpoints for each segment: [(x1,y1), (x2,y2)]
            for segment in BPl:
                p1, p2 = segment
                line_marker.points.append(Point(x=p1[0], y=p1[1], z=0.0))
                line_marker.points.append(Point(x=p2[0], y=p2[1], z=0.0))

            marker_array.markers.append(line_marker)
            marker_id += 1

        # 3. CREATE CIRCLE MARKERS
        for circle in Fc:
            a, b, r = circle # x-center, y-center, radius
            
            circle_marker = Marker()
            circle_marker.header.frame_id = self.lidar_frame_id
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

            # Scale (Cylinder diameter is 2 * radius)
            circle_marker.scale.x = 2.0 * r
            circle_marker.scale.y = 2.0 * r
            circle_marker.scale.z = 0.1 # Flat disc height

            # Color: Blue
            circle_marker.color.r = 0.0
            circle_marker.color.g = 0.5
            circle_marker.color.b = 1.0
            circle_marker.color.a = 0.6 # Slightly transparent

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
            
            self.voronoi(self.obstacles)

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