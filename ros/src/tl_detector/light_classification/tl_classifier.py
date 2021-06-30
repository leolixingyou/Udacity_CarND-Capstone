from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime
import rospy
import cv2

class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        self.sess = None

        self.tf_graph = tf.Graph()
        self.tf_graphdef = tf.GraphDef()
        
        self.tf_conf = tf.ConfigProto()
        self.tf_conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        with tf.Session(config=self.tf_conf, graph=self.tf_graph) as sess:
            self.sess = sess
            self.tf_graphdef.ParseFromString(tf.gfile.GFile('/home/workspace/CarND-Capstone/ros/src/tl_detector/weight/sim/tf_weight.pb', 'rb').read())
            tf.import_graph_def(self.tf_graphdef, name='')
        
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction

        # with self.detection_graph.as_default():
        image = cv2.resize(image, (300, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = [-1, TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN, TrafficLight.UNKNOWN]

        (bb, conf, answer) = self.sess.run(
            [self.tf_graph.get_tensor_by_name('detection_boxes:0'), 
            self.tf_graph.get_tensor_by_name('detection_scores:0'),
            self.tf_graph.get_tensor_by_name('detection_classes:0')], 
            feed_dict={self.tf_graph.get_tensor_by_name('image_tensor:0'): np.expand_dims(image, axis=0)})

        conf = np.squeeze(conf)
        answer = np.squeeze(answer)
        bb = np.squeeze(bb)
        
        high_conf = 0
        label = None

        for i in range(len(conf)):
            if conf[i] > high_conf:
                high_conf = conf[i]
                label = labels[int(answer[i])]
                
        return label