#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import TwistStamped

class RobotController(Node):
    def __init__(self):
        # Initialize the ROS 2 node named 'robot_controller'
        super().__init__('robot_controller')
        
        # Publisher: Sends movement commands to the robot
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        # Subscriber: Listens for the vision node's detection results
        self.create_subscription(String, '/detected_sign', self.sign_cb, 10)
        
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.get_logger().info('🤖 Robot Controller initialized and waiting for signs...')

    def sign_cb(self, msg):
        sign_data = msg.data.lower()
        
        # Parse sign name and confidence
        try:
            sign_name = sign_data.split('(')[0].strip()
        except:
            sign_name = sign_data

        # Logic Mapping
        if 'stop' in sign_name:
            self.target_speed = 0.0
            self.get_logger().warn('🛑 STOP SIGN: Emergency Brake Engaged!')
        elif 'speed 30' in sign_name:
            self.target_speed = 0.2
            self.get_logger().info('🟡 Speed 30: Moving cautiously.')
        elif 'speed 60' in sign_name:
            self.target_speed = 0.4
            self.get_logger().info('🟢 Speed 60: Moving at cruising speed.')
        elif 'speed 100' in sign_name:
            self.target_speed = 0.6
            self.get_logger().info('⚡ Speed 100: Maximum safe speed!')
        else:
            # Default behavior for unknown but detected red signs
            self.target_speed = 0.2
            self.get_logger().info(f'❓ Detected {sign_name}: Maintaining safety speed.')

        self.publish_twist()

    def publish_twist(self):
        # Create a TwistStamped message (Standard for ROS 2 robot motion)
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_link'
        
        # Linear.x controls forward/backward movement
        # Angular.z controls rotation (0.0 means move straight)
        twist.twist.linear.x = self.target_speed
        twist.twist.angular.z = 0.0
        
        # Send the command to the robot actuators
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
