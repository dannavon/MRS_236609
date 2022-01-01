#! /usr/bin/env python
import rospy
import time
from geometry_msgs.msg import Twist
import math

def move_line():
    rospy.init_node('twist_publisher')
    twist = Twist()
    twist.angular.z = 0
    twist.linear.x = 0.5
    pub = rospy.Publisher('/cmd_vel',Twist , queue_size=1)
    r = rospy.Rate(2)
    while not rospy.is_shutdown(): 
        pub.publish(twist)
        r.sleep()


def move_circle():
    rospy.init_node('twist_publisher')
    twist = Twist()
    twist.angular.z = 1
    twist.linear.x = 0.6
    pub = rospy.Publisher('/cmd_vel',Twist , queue_size=1)
    
    r = rospy.Rate(10)
    t1 = time.time()
    t2 = t1
    while t2-t1 < 13:
        t2 = time.time()
        pub.publish(twist)
        r.sleep()
    print('COMPLETE')

    stop(twist,pub)
    stop(twist,pub)
    stop(twist,pub)


def stop(twist,pub):
    twist.linear.x = 0
    twist.angular.z = 0
    pub.publish(twist)     
    r = rospy.Rate(10)

    t1 = time.time()
    t2 = t1
    while t2-t1 < 0.2:
        t2 = time.time()
        pub.publish(twist)
        r.sleep()
        
def move_square():
    rospy.init_node('twist_publisher')
    twist = Twist()
    pub = rospy.Publisher('/cmd_vel',Twist , queue_size=1)
    r = rospy.Rate(10)

    for i in range (4):
        twist.linear.x = 0.5
        twist.angular.z = 0
        t1 = time.time()
        t2 = t1
        #head straight
        while t2-t1<4: 
            t2 = time.time()
            pub.publish(twist)
            r.sleep()
        stop(twist,pub)
        
        #turn
        twist.linear.x = 0
        twist.angular.z = math.pi/2
        t1 = time.time()
        t2 = t1          
        while t2-t1<0.8: 
            t2 = time.time()
            pub.publish(twist)
            r.sleep()
        stop(twist,pub)


    twist.linear.x = 0
    twist.angular.z = 0
    pub.publish(twist)
    print('COMPLETE')


if __name__ == "__main__":

    move_circle()
    move_square()
