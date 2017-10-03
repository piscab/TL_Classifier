# 
from styx_msgs.msg import TrafficLight
from keras.models import model_from_json
from keras.models import load_model
from keras.utils  import np_utils
from keras.optimizers  import Adam
import json
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        with open("light_classification/model.json", 'r') as jfile:
          self.model = model_from_json(json.load(jfile))

        #Load the weights
        weights_file = "light_classification/model.h5"
        self.model.load_weights(weights_file)

        #Compile the model
        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=Adam(lr=0.0005),
                           metrics=['accuracy'])


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction

        # Convert rgb in gbr
        # consider also to self.bridge ros images to "rgb8" instead of "bgr8"
        dim = 200
        img  = image[...,::-1]

        img  = np.reshape(img,[1,dim,dim,3])
        y_predicted = self.model.predict(img)
        y_pred      = np_utils.probas_to_classes(y_predicted)
        if y_pred == 0:
            return TrafficLight.RED
        elif y_pred == 1:
            return TrafficLight.YELLOW
        elif y_pred == 2:
            return TrafficLight.GREEN

        return TrafficLight.UNKNOWN
