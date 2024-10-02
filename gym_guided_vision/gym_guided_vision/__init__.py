from gymnasium.envs.registration import register


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
        "env": "SlotInsertionEnv",
        "num_arms": 3,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
    },
    "gym_guided_vision/SlotInsertion-2Arms-v0": {
        "env": "SlotInsertionEnv",
        "num_arms": 2,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
    },

    # SewNeedle
    "gym_guided_vision/SewNeedle-3Arms-v0": {
        "env": "SewNeedleEnv",
        "num_arms": 3,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
    },
    "gym_guided_vision/SewNeedle-2Arms-v0": {
        "env": "SewNeedleEnv",
        "num_arms": 2,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
    },
   
    # TubeTransfer
    "gym_guided_vision/TubeTransfer-3Arms-v0": {
        "env": "TubeTransferEnv",
        "num_arms": 3,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
    },
    "gym_guided_vision/TubeTransfer-2Arms-v0": {
        "env": "TubeTransferEnv",
        "num_arms": 2,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
    },
  

    # HookPackage
    "gym_guided_vision/HookPackage-3Arms-v0": {
        "env": "HookPackageEnv",
        "num_arms": 3,
        "cameras": ["zed_cam_left", "zed_cam_right", "wrist_cam_left", "wrist_cam_right", "overhead_cam", "worms_eye_cam"],
        "observation_height": 480,
        "observation_width": 640,
    },
    "gym_guided_vision/HookPackage-2Arms-v0": {
        "env": "HookPackageEnv",
        "num_arms": 2,
        "cameras": ["overhead_cam", "worms_eye_cam", "wrist_cam_left", "wrist_cam_right"],
        "observation_height": 480,
        "observation_width": 640,
    },
  
}

for env_id, env_kwargs in ENVS.items():
    register(
        id=env_id,
        entry_point=f"gym_guided_vision.env:{env_kwargs['env']}",
        # Even after seeding, the rendered observations are slightly different,
        # so we set `nondeterministic=True` to pass `check_env` tests
        nondeterministic=True,
        kwargs={
            "num_arms": env_kwargs["num_arms"],
            "cameras": env_kwargs["cameras"],
            "observation_height": env_kwargs["observation_height"],
            "observation_width": env_kwargs["observation_width"],
        }
    )
