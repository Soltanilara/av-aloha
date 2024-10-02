```python
ENVS = {
     # InsertPeg
    "gym_guided_vision/InsertPeg-3Arms-v0": {
        "env": "InsertPegEnv",
        "num_arms": 3,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
    },
    "gym_guided_vision/InsertPeg-2Arms-v0": {
        "env": "InsertPegEnv",
        "num_arms": 2,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
    },

    # SlotInsertion
    "gym_guided_vision/SlotInsertion-3Arms-v0": {
        "env": "SlotInsertion",
        "num_arms": 3,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
    },
    "gym_guided_vision/SlotInsertion-2Arms-v0": {
        "env": "SlotInsertion",
        "num_arms": 2,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
    },

    # SewNeedle
    "gym_guided_vision/SewNeedle-3Arms-v0": {
        "env": "SewNeedle",
        "num_arms": 3,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
    },
    "gym_guided_vision/SewNeedle-2Arms-v0": {
        "env": "SewNeedle",
        "num_arms": 2,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
    },
   
    # TubeTransfer
    "gym_guided_vision/TubeTransfer-3Arms-v0": {
        "env": "TubeTransfer",
        "num_arms": 3,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
    },
    "gym_guided_vision/TubeTransfer-2Arms-v0": {
        "env": "TubeTransfer",
        "num_arms": 2,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
    },
  

    # HookPackage
    "gym_guided_vision/HookPackage-3Arms-v0": {
        "env": "HookPackage",
        "num_arms": 3,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
    },
    "gym_guided_vision/HookPackage-2Arms-v0": {
        "env": "HookPackage",
        "num_arms": 2,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
    },
  
}
```