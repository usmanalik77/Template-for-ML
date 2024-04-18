import rclpy
import os
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped  # Adjust message types based on observations
from sensor_msgs.msg import LaserScan  # Adjust message types based on observations
from geometry_msgs.msg import Twist, PoseStamped  # Adjust message types for actions and navigation completion
import numpy as np
import time

class DataCollectorNode(Node):

    def __init__(self, node_name):
        super().__init__(node_name)
        self.observations = []
        self.actions = []

        # subscriptions for observations and actions 
        self.observation_sub = self.create_subscription(
            PoseStamped,  #  message type 
            "/topic/observation",  # observation topic
            self.collect_data,
            10)  # que size

        self.action_sub = self.create_subscription(
            Twist,  #  message type for action
            "/topic/action",  # action topic
            self.collect_data,
            10)  

        ###: usman: subscription for navigation completion 
        self.nav_completion_sub = self.create_subscription(
            PoseStamped,  #  PoseStamped as completion signal
            "/topic/navigation_completion",  # actual topic
            self.navigation_completion_callback,
            10)  

    def collect_data(self, observation_msg, action_msg):
        #  relevant data from observation and action messages 
        observation = [observation_msg.pose.position.x, observation_msg.pose.position.y]
        action = [action_msg.linear.x, action_msg.angular.z]

        self.observations.append(observation)
        self.actions.append(action)

    def save_data_to_npz(self, data_dir, filename):
        #  observations and actions to numpyu arrays
        observations_array = np.array(self.observations)
        actions_array = np.array(self.actions)

        #  .npz file
        np.savez(os.path.join(data_dir, filename), observations=observations_array, actions=actions_array)

    def navigation_completion_callback(self, msg):
        # upon navigation completion
        self.save_data_to_npz("data", "navigation_data.npz")  # Example filename

def main():
    rclpy.init()
    node = DataCollectorNode("data_collector_node")

    save_period = 5.0  #every 5 seconds

    while rclpy.ok():
        rclpy.spin_once(node)
        time.sleep(save_period)

        # data periodically
        node.save_data_to_npz("data", f"data_{time.time()}.npz")  # Filename with timestamp

    rclpy.shutdown()

if __name__ == "__main__":
    main()