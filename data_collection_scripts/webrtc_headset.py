from google.cloud import firestore
import json
import asyncio
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCRtpSender
)
from aiortc import VideoStreamTrack
from av import VideoFrame
import numpy as np
import queue
import time
from headset_utils import HeadsetData, HeadsetFeedback, convert_left_to_right_coordinates
import os
import threading
import cv2

def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    transceiver.setCodecPreferences(
        [codec for codec in codecs if codec.mimeType == forced_codec]
    )

class BufferVideoStreamTrack(VideoStreamTrack):
    def __init__(self, buffer_size=1, image_format="rgb24", max_fps=60):
        super().__init__()
        self.queue = queue.Queue(maxsize=buffer_size)
        self.image_format = image_format
        self.last_frame = None
        self.max_fps = max_fps
        self.last_send_time = time.time()


    async def get_frame(self) -> np.ndarray:
        while True:
            try:
                frame = self.queue.get_nowait()
                self.last_frame = frame
                return frame
            except queue.Empty:
                if self.last_frame is not None:
                    return self.last_frame.copy()
                await asyncio.sleep(0)
        

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = await self.get_frame()
        # convert to gray scale
        frame = VideoFrame.from_ndarray(frame, format=self.image_format)
        frame.pts = pts
        frame.time_base = time_base

        # limit fps
        elapsed_time = time.time() - self.last_send_time
        await asyncio.sleep(max(1/self.max_fps - elapsed_time, 0))
        self.last_send_time = time.time()

        return frame

    def add_frame(self, frame):
        # try to put but if pull pop the oldest frame
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

class WebRTCHeadset:
    def __init__(
        self,
        serviceAccountKeyFile='serviceAccountKey.json',
        signalingSettingsFile='signalingSettings.json',
        video_buffer_size=1,
        data_buffer_size=1,
        send_data_freq=10,
    ):        
        # create firestore client
        with open(serviceAccountKeyFile) as f:
            serviceAccountKey = json.load(f)
        self.db = firestore.Client.from_service_account_info(serviceAccountKey)

        # load signaling settings
        with open(signalingSettingsFile) as f:
            signalingSettings = json.load(f)
        self.robotId = signalingSettings['robotID']
        self.password = signalingSettings['password']
        self.turn_server_url = signalingSettings['turn_server_url']
        self.turn_server_username = signalingSettings['turn_server_username']
        self.turn_server_password = signalingSettings['turn_server_password']

        # create peer connection
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                RTCIceServer("stun:stun1.l.google.com:19302"),
                RTCIceServer("stun:stun2.l.google.com:19302"),
                RTCIceServer(self.turn_server_url, self.turn_server_username, self.turn_server_password)
            ])
        )

        # vars for video and data
        self.channel = None
        self.left_video_track = None
        self.right_video_track = None
        self.video_buffer_size = video_buffer_size
        self.data_buffer_size = data_buffer_size
        self.send_data_freq = send_data_freq
        self.receive_data_queue = queue.Queue(maxsize=data_buffer_size)
        self.send_data_queue = queue.Queue(maxsize=data_buffer_size)

        self.thread = None
        self.event_loop = None

    async def channel_send_loop(self):
        last_data = None
        while True:
            start_time = time.time()
            
            try:
                if self.channel is not None and self.channel.readyState == "open":
                    data = self.send_data_queue.get_nowait()
                    data = json.dumps(data)
                    last_data = data
                    self.channel.send(data)
            except Exception as e:
                try:
                    if last_data is not None:
                        self.channel.send(last_data)
                except Exception as e:
                    print(f"Failed to send data: {e}")

            elapsed_time = time.time() - start_time
            await asyncio.sleep(max(1/self.send_data_freq - elapsed_time, 0))

    def receive_data(self) -> HeadsetData:
        try:
            data = self.receive_data_queue.get_nowait()
            return data
        except queue.Empty:
            return None
    
    def send_images(self, left_image: np.ndarray, right_image: np.ndarray):
        try:
            if self.left_video_track is not None:
                self.left_video_track.add_frame(left_image) #TODO copy image?
            if self.right_video_track is not None:
                self.right_video_track.add_frame(right_image) #TODO copy image?
        except Exception as e:
            print(f"Failed to send image: {e}")

    def send_feedback(self, data: HeadsetFeedback):
        data = {
            'headOutOfSync': data.head_out_of_sync,
            'leftOutOfSync': data.left_out_of_sync,
            'rightOutOfSync': data.right_out_of_sync,
            'info': data.info,
            'leftArmPosition': data.left_arm_position.tolist(),
            'leftArmRotation': data.left_arm_rotation.tolist(),
            'rightArmPosition': data.right_arm_position.tolist(),
            'rightArmRotation': data.right_arm_rotation.tolist(),
            'middleArmPosition': data.middle_arm_position.tolist(),
            'middleArmRotation': data.middle_arm_rotation.tolist(),
        }
        try:
            self.send_data_queue.put_nowait(data)
        except queue.Full:
            try:
                self.send_data_queue.get_nowait()
                self.send_data_queue.put_nowait(data)
            except (queue.Empty, queue.Full):
                pass

    def on_message(self, message):
        try:
            headset_data = HeadsetData()
            data = json.loads(message)
        except json.JSONDecodeError:
            print("WebRTC: JSON decode error")
            return

        try:
            headset_data.h_pos[0] = data['HPosition']['x']
            headset_data.h_pos[1] = data['HPosition']['y']
            headset_data.h_pos[2] = data['HPosition']['z']
            headset_data.h_quat[0] = data['HRotation']['x']
            headset_data.h_quat[1] = data['HRotation']['y']
            headset_data.h_quat[2] = data['HRotation']['z']
            headset_data.h_quat[3] = data['HRotation']['w']
            headset_data.l_pos[0] = data['LPosition']['x']
            headset_data.l_pos[1] = data['LPosition']['y']
            headset_data.l_pos[2] = data['LPosition']['z']
            headset_data.l_quat[0] = data['LRotation']['x']
            headset_data.l_quat[1] = data['LRotation']['y']
            headset_data.l_quat[2] = data['LRotation']['z']
            headset_data.l_quat[3] = data['LRotation']['w']
            headset_data.l_thumbstick_x = data['LThumbstick']['x']
            headset_data.l_thumbstick_y = data['LThumbstick']['y']
            headset_data.l_index_trigger = data['LIndexTrigger']
            headset_data.l_hand_trigger = data['LHandTrigger']
            headset_data.l_button_one = data['LButtonOne']
            headset_data.l_button_two = data['LButtonTwo']
            headset_data.l_button_thumbstick = data['LButtonThumbstick']
            headset_data.r_pos[0] = data['RPosition']['x']
            headset_data.r_pos[1] = data['RPosition']['y']
            headset_data.r_pos[2] = data['RPosition']['z']
            headset_data.r_quat[0] = data['RRotation']['x']
            headset_data.r_quat[1] = data['RRotation']['y']
            headset_data.r_quat[2] = data['RRotation']['z']
            headset_data.r_quat[3] = data['RRotation']['w']
            headset_data.r_thumbstick_x = data['RThumbstick']['x']
            headset_data.r_thumbstick_y = data['RThumbstick']['y']
            headset_data.r_index_trigger = data['RIndexTrigger']
            headset_data.r_hand_trigger = data['RHandTrigger']
            headset_data.r_button_one = data['RButtonOne']
            headset_data.r_button_two = data['RButtonTwo']
            headset_data.r_button_thumbstick = data['RButtonThumbstick']
            headset_data.h_pos, headset_data.h_quat = convert_left_to_right_coordinates(headset_data.h_pos, headset_data.h_quat)
            headset_data.l_pos, headset_data.l_quat = convert_left_to_right_coordinates(headset_data.l_pos, headset_data.l_quat)
            headset_data.r_pos, headset_data.r_quat = convert_left_to_right_coordinates(headset_data.r_pos, headset_data.r_quat)
        except KeyError:
            print("[RobotWebRTC] Key error") 
            return

        try:
            self.receive_data_queue.put_nowait(headset_data)
        except queue.Full:
            try:
                self.receive_data_queue.get_nowait()
                self.receive_data_queue.put_nowait(headset_data)
            except (queue.Empty, queue.Full):
                pass

    async def run_offer(self):
        # create data channel
        self.channel = self.pc.createDataChannel("control")
        @self.channel.on("open")
        def on_open():
            print("Data channel is open.")
        self.channel.on("message", self.on_message)       

        # create video track
        self.left_video_track = BufferVideoStreamTrack(buffer_size=self.video_buffer_size)
        self.left_video_sender = self.pc.addTrack(self.left_video_track)
        force_codec(self.pc, self.left_video_sender, 'video/VP8')

        # create video track
        self.right_video_track = BufferVideoStreamTrack(buffer_size=self.video_buffer_size)
        self.right_video_sender = self.pc.addTrack(self.right_video_track)
        force_codec(self.pc, self.right_video_sender, 'video/VP8')


        # create offer and place in firestore     
        print("WebRTC: Running offer...")  
        await self.pc.setLocalDescription(await self.pc.createOffer())
        call_doc = self.db.collection(self.password).document(self.robotId)
        call_doc.set(
            {
                'sdp': self.pc.localDescription.sdp,
                'type': self.pc.localDescription.type
            }
        )

        # wait for answer from firestore
        data = None
        def answer_callback(doc_snapshot, changes, read_time):
            nonlocal data
            for doc in doc_snapshot:
                if self.pc.remoteDescription is None and doc.to_dict()['type'] == 'answer':
                    data = doc.to_dict()
        doc_watch = call_doc.on_snapshot(answer_callback)
        print('WebRTC: Waiting for answer...')
        while data is None:
            await asyncio.sleep(1/30)
        print('WebRTC: Answer received.')
        doc_watch.unsubscribe()

        # set remote description from answer
        await self.pc.setRemoteDescription(RTCSessionDescription(
            sdp=data['sdp'],
            type=data['type']
        ))

        # delete firestore call document
        call_doc = self.db.collection(self.password).document(self.robotId)
        call_doc.delete()

        # add event listener for connection close
        @self.pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            if self.pc.iceConnectionState == "closed":
                print("WebRTC: Connection closed, restarting...")
                await self.restart_connection()

    async def restart_connection(self):
        # close current peer connection
        await self.pc.close()

        # create new peer connection
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration([
                RTCIceServer("stun:stun1.l.google.com:19302"),
                RTCIceServer("stun:stun2.l.google.com:19302"),
                RTCIceServer(self.turn_server_url, self.turn_server_username, self.turn_server_password)
            ])
        )

        # run offer again
        await self.run_offer() 

    def run_in_thread(self):
        def run(loop: asyncio.AbstractEventLoop):  
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.run_offer())  
            loop.create_task(self.channel_send_loop())
            loop.run_forever()

        self.event_loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=run, args=(self.event_loop,))
        self.thread.start()

    def close(self):
        if self.thread is not None and self.thread.is_alive():
            self.event_loop.stop()
            self.thread.join()

if __name__ == "__main__":
    try:
        headset = WebRTCHeadset()
        headset.run_in_thread()
        
        while True:
            data = headset.receive_data()
            if data is not None:
                print(f"Received data: {data.h_pos}, {data.h_quat}")

            feedback = HeadsetFeedback()
            feedback.info = f"Hello from python: {time.time()}"
            headset.send_feedback(feedback)

            headset.send_image(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))

    except KeyboardInterrupt:
        print("Shutting down...")
        os._exit(42)
