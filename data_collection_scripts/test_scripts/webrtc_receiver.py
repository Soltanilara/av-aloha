from google.cloud import firestore
import json
import asyncio
import cv2
import numpy as np
from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.contrib.signaling import BYE, add_signaling_arguments, create_signaling
from aiortc.sdp import candidate_from_sdp

class VideoDisplayTrack(VideoStreamTrack):
    def __init__(self, track, name="video_display_track"):
        super().__init__()
        self.track = track
        self.name = name
        self.image = None

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="rgb24")  # Convert frame to OpenCV format

        self.image = img

        return frame
    
    def get_image(self):
        return self.image
    
left_video_track = None
right_video_track = None
counter =0

async def run_answer(pc, db, signalingSettings):
    global left_video_track, right_video_track, counter

    @pc.on("datachannel")
    def on_datachannel(channel):
        print('Data channel opened')
        # @channel.on("message")
        # def on_message(message):
        #     print('Message received:', message) 

    @pc.on("track")
    def on_track(track):
        global left_video_track, right_video_track, counter
        print("Receiving %s" % track.kind)
        if track.kind == "video":
            if counter == 0:
                print('Left video track')
                left_video_track = VideoDisplayTrack(track, name="left_video_track")
                pc.addTrack(left_video_track)
                counter += 1
            else:
                print('Right video track')
                right_video_track = VideoDisplayTrack(track, name="right_video_track")
                pc.addTrack(right_video_track)

    call_doc = db.collection(signalingSettings['password']).document(signalingSettings['robotID'])

    data = call_doc.get().to_dict()

    await pc.setRemoteDescription(RTCSessionDescription(
        sdp=data['sdp'],
        type=data['type']
    ))

    await pc.setLocalDescription(await pc.createAnswer())

    call_doc.set(
        {
            'sdp': pc.localDescription.sdp,
            'type': pc.localDescription.type
        }
    )

if __name__ == "__main__":
    # Read firebase-creds.json
    with open('../serviceAccountKey.json') as f:
        serviceAccountKey = json.load(f)

    with open('../signalingSettings.json') as f:
        signalingSettings = json.load(f)

    db = firestore.Client.from_service_account_info(serviceAccountKey)

    pc = RTCPeerConnection(
        configuration=RTCConfiguration([
            RTCIceServer("stun:stun1.l.google.com:19302"),
            RTCIceServer("stun:stun2.l.google.com:19302"),
        ])
    )


    import threading

    def run(loop: asyncio.AbstractEventLoop):  
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_answer(pc, db, signalingSettings))  
        loop.run_forever()

    event_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=run, args=(event_loop,))
    thread.start()


    




    try:
        while True:
            if left_video_track is not None and right_video_track is not None:
                left_frame = left_video_track.get_image()
                right_frame = right_video_track.get_image()



                if left_frame is None or right_frame is None:
                    continue
                

                concat_image = np.concatenate((left_frame, right_frame), axis=1)
                cv2.imshow('Live Stream', concat_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    
                    break

    except KeyboardInterrupt:
        pass
    finally:

        # kill the thread

        import os
        os._exit(42)
