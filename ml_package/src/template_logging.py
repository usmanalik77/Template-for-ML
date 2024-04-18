import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped  # message type for sim data
import curses  # user interface
import cv2  # OpenCV for image processing  
from sensor_msgs.msg import Image  # if image message type


class GenericLoggerNode(Node):

    def __init__(self, node_name):
        super().__init__(node_name)
        self.observations = []
        self.actions = []
        self.sim_data_sub = None  #  data subscription

    
        self.declare_parameter("observation_topic", "/topic/observations")  # Adjust topic name
        self.declare_parameter("action_topic", "/topic/actions")  # Adjust topic name

    def sim_data_callback(self, msg):
        #  data message is LaserScan
        observation = []

        if isinstance(msg, sensor_msgs.msg.LaserScan):
            #  relevant data 
            observation.append(msg.ranges[0])  # range measurement
            observation.append(len(msg.ranges) / 2)  # average range measurement

        #  data from other sensor messages ???

        # receive actions from source LIKE user input
        action = [action_data_1, action_data_2]  

        self.observations.append(observation)
        self.actions.append(action)

        # saving data to desired format periodically

    def get_user_action():
        stdscr = curses.initscr()  #  curses screen
        curses.cbreak()  # Disable line buffering
        stdscr.keypad(True)  #special key handling

        key = stdscr.getch()  # wait for a key press

        curses.endwin()  
        # Map keys modify as needed
        if key == curses.KEY_UP:
            action = [1, 0]  # Move forward
        elif key == curses.KEY_DOWN:
            action = [-1, 0]  # Move backward
        elif key == curses.KEY_LEFT:
            action = [0, -1]  # Turn left
        elif key == curses.KEY_RIGHT:
            action = [0, 1]  # Turn right
        else:
            action = [0, 0]  # No action

        return action

        
    def preprocess_image(self, image_msg):
        #  ROS image to OpenCV format
        cv_image = cv2.bridge.img_to_cv2(image_msg, desired_encoding='bgr8')

        #  preprocessing steps (e.g., resizing, filtering)
        # ... (your image processing logic)

        ### IS IT BGR OR RGB?
        
        return cv_image
    

    # def sim_data_callback(self, msg):
        # Assuming Gazebo image message (sensor_msgs.msg.Image)
        observation = []

    # def get_feature_vector(image):

        #  the image for the chosen model (
        image = cv2.resize(image, (224, 224))  # Resize 
        image = image.astype('float32') / 255.0  # pixel values

        # Expand dimensions 
        image = np.expand_dims(image, axis=0)

        # Extract features 
        features = model.predict(image)

        # Flatten to 1D vector
        feature_vector = features.flatten()

        return feature_vector


        if isinstance(msg, Image):
            image = self.preprocess_image(msg)  
            observation.append(extracted_feature_1)
            observation.append(extracted_feature_2)

        # receive actions  another source 
        action = [action_data_1, action_data_2]  
        self.observations.append(observation)
        self.actions.append(action)



    def main():
        rclpy.init()
        node = GenericLoggerNode("generic_logger_node")

        # node.set_sim_data_topic("/custom/sim/data")  
        while rclpy.ok():
            rclpy.spin_once(node)
            time.sleep(save_period)

            # data periodically
            node.save_data_to_npz("data", f"data_{time.time()}.npz")  # Filename with timestamp

        rclpy.spin(node)
        rclpy.shutdown()

if __name__ == "__main__":
    main()