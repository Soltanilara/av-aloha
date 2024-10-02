import sys
sys.path.append('..')

from webrtc_headset import WebRTCHeadset
from headset_utils import HeadsetFeedback
import os
import cv2
import numpy as np

def main():
    headset = WebRTCHeadset(serviceAccountKeyFile='../serviceAccountKey.json', signalingSettingsFile='../signalingSettings.json')
    headset.run_in_thread()


    # Open a connection to the webcam (usually 0 is the default camera)
    cap = cv2.VideoCapture(0)

    # set the resolution to `720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # set to 1080p
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()


        if not ret:
            print("Error: Failed to capture image.")
            break

        left_frame = frame.copy()
        right_frame = frame.copy()

        # draw "left" in the left frame (put in center of picture)
        cv2.putText(left_frame, "Left", (int(left_frame.shape[1]/2)-50, int(left_frame.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(right_frame, "Right", (int(right_frame.shape[1]/2)-50, int(right_frame.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        concat_image = np.concatenate((left_frame, right_frame), axis=1)

        cv2.imshow('Robot Side', concat_image)

        # convert to greyscale bgr
        # left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_RGB2BGR)
        # right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_RGB2BGR)

        headset.send_images(left_frame, right_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
        os._exit(42)

