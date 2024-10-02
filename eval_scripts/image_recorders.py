import numpy as np
import time
import time
import threading
import pyzed.sl as sl 
from collections import deque
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import threading
import cv2
from functools import partial

import IPython
e = IPython.embed

class ZEDImageRecorder:
    def __init__(self, auto_start=True):
        self.zed = None
        self.image = None
        self.is_running = False
        self.thread = None
        if auto_start:
            self.start()

    def __del__(self):
        self.stop()

    def start(self):
        self.thread = threading.Thread(target=self.record_images)
        self.thread.start()

        # try to see if we can get the first image timeout after 5 seconds
        start_time = time.time()
        while self.image is None:
            if time.time() - start_time > 5:
                raise Exception("Timeout, failed to get image from ZED camera")
            time.sleep(0.1)
    
    def stop(self):
        if self.thread.is_alive():
            self.is_running = False
            self.thread.join()

    def record_images(self):
        # Create a ZED camera object
        self.zed = sl.Camera()

        # Set configuration parameters
        input_type = sl.InputType()
        init = sl.InitParameters(input_t=input_type)
        init.camera_resolution = sl.RESOLUTION.HD720
        
        # Open the camera and timeout if not opened
        timeout_duration = 5  # [s]
        start_time = time.time()
        err = self.zed.open(init)
        while err != sl.ERROR_CODE.SUCCESS:
            if time.time() - start_time > timeout_duration:
                raise Exception(f"Failed to open ZED camera: {err}")
            err = self.zed.open(init)
            
        # Set runtime parameters after opening the camera
        self.runtime = sl.RuntimeParameters()

        # Prepare new image size to retrieve half-resolution images
        self.image_size = self.zed.get_camera_information().camera_configuration.resolution
        # Declare sl.Mat matrices
        self.left_image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C3)
        self.right_image_zed = sl.Mat(self.image_size.width, self.image_size.height, sl.MAT_TYPE.U8_C3)

        self.is_running = True
        while self.is_running and not rospy.is_shutdown():
            if self.zed.grab(self.runtime) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.left_image_zed, sl.VIEW.LEFT, sl.MEM.CPU, self.image_size)
                self.zed.retrieve_image(self.right_image_zed, sl.VIEW.RIGHT, sl.MEM.CPU, self.image_size)
                left_image_ocv = self.left_image_zed.get_data()
                right_image_ocv = self.right_image_zed.get_data()

                # convert to RGB
                left_image_ocv = cv2.cvtColor(left_image_ocv, cv2.COLOR_BGR2RGB)
                right_image_ocv = cv2.cvtColor(right_image_ocv, cv2.COLOR_BGR2RGB)

                frame = np.concatenate((left_image_ocv, right_image_ocv), axis=1)

                self.image = frame

        self.zed.close()

    def get_image(self):
        if self.is_running and self.image is not None:
            return self.image
        else:
            raise Exception("ZED camera is not running or image is not available")

class ROSImageRecorder:
    def __init__(self, 
                 camera_names=['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
                 init_node=True, 
                 is_debug=False):
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = camera_names
        if init_node:
            rospy.init_node('image_recorder', anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f'{cam_name}_image', None)
            setattr(self, f'{cam_name}_secs', None)
            setattr(self, f'{cam_name}_nsecs', None)
            callback_func = partial(self.image_cb, cam_name)
            rospy.Subscriber(f"/{cam_name}/color/image_raw", Image, callback_func) 
            if self.is_debug:
                setattr(self, f'{cam_name}_timestamps', deque(maxlen=50))

            rospy.wait_for_message(f"/{cam_name}/color/image_raw", Image, timeout=1.0)

    def image_cb(self, cam_name, data):
        setattr(self, f'{cam_name}_image', self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough'))
        setattr(self, f'{cam_name}_secs', data.header.stamp.secs)
        setattr(self, f'{cam_name}_nsecs', data.header.stamp.nsecs)
        if self.is_debug:
            getattr(self, f'{cam_name}_timestamps').append(data.header.stamp.secs + data.header.stamp.secs * 1e-9)

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}_image')
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)
        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f'{cam_name}_timestamps'))
            print(f'{cam_name} {image_freq=:.2f}')
        print()