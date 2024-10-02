from pathlib import Path
import gym_guided_vision.constants
from tqdm import tqdm
import einops
import numpy as np
import torch
import imageio
import os
from torch import Tensor
import torchvision.transforms as v2
from huggingface_hub import snapshot_download
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.envs.utils import preprocess_observation

import gym_guided_vision
import gymnasium as gym

from real_env import RealEnv
from constants import REAL_DT

RESIZE = v2.Resize((480, 640))

def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, Tensor]:
    """Convert environment observation to LeRobot format observation.
    Args:
        observation: Dictionary of observation batches from a Gym vector environment.
    Returns:
        Dictionary of observation batches with keys renamed to LeRobot format and values as tensors.
    """
    # map to expected inputs for the policy
    return_observations = {}
    if "pixels" in observations:
        if isinstance(observations["pixels"], dict):
            imgs = {f"observation.images.{key}": img for key, img in observations["pixels"].items()}
        else:
            imgs = {"observation.image": observations["pixels"]}

        for imgkey, img in imgs.items():
            img = torch.from_numpy(img.copy()).unsqueeze(0)

            # sanity check that images are channel last
            _, h, w, c = img.shape
            assert c < h and c < w, f"expect channel last images, but instead got {img.shape=}"

            # sanity check that images are uint8
            assert img.dtype == torch.uint8, f"expect torch.uint8, but instead {img.dtype=}"

            # convert to channel first of type float32 in range [0,1]
            img = einops.rearrange(img, "b h w c -> b c h w").contiguous()
            img = img.type(torch.float32)
            img /= 255

            # resive with v2.Resize(resize) to 480x640
            img = RESIZE(img)

            return_observations[imgkey] = img

    if "environment_state" in observations:
        return_observations["observation.environment_state"] = torch.from_numpy(
            observations["environment_state"]
        ).float()

    # TODO(rcadene): enable pixels only baseline with `obs_type="pixels"` in environment by removing
    # requirement for "agent_pos"
    return_observations["observation.state"] = torch.from_numpy(observations["agent_pos"]).float().unsqueeze(0)
    return return_observations

def main(args):
    policy_path = args['policy']
    episode_len = args['episode_len']
    num_episodes = args['num_episodes']
    sim_env = args['sim_env']

    # Download the diffusion policy for pusht environment
    pretrained_policy_path = Path(snapshot_download(policy_path))

    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
    policy.eval()
    
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available. Device set to:", device)
    else:
        raise Exception("GPU not available")
    
    policy.to(device)

    # setup the environment
    if sim_env == '':
        env = RealEnv(init_node=True)
        fps = REAL_DT
    else:
        env = gym.make(sim_env)
        fps = env.metadata["render_fps"]

    # run the policy for the specified number of steps
    for i in range(num_episodes):

        print("Resetting environment...")
        # reset the environment
        policy.reset()
        observation, info = env.reset()

        print(f"Running episode {i+1}/{num_episodes}")

        frames = []
        for _ in tqdm(range(episode_len)):
            observation = preprocess_observation(observation)
            observation = {key: observation[key].to(device, non_blocking=True) for key in observation}

            with torch.inference_mode():
                action = policy.select_action(observation)

            # Convert to CPU / numpy.
            action = action.to("cpu").numpy()
            assert action.ndim == 2, "Action dimensions should be (batch, action_dim)"

            # Apply the next action.
            observation, reward, terminated, truncated, info = env.step(action[0])

            if "pixels" in observation and "zed_cam_left" in observation["pixels"]:
                frames.append(observation["pixels"]["zed_cam_left"])

        # Encode all frames into a mp4 video.
        video_path = os.path.join("outputs", policy_path.split("/")[-1], f"rollout_{i}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(str(video_path), np.stack(frames), fps=50)

        input("Press Enter to continue...")

if __name__ == "__main__":
    import rospy
    import os
    import argparse

    # add arg for policy
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True, help='Path to the policy')
    parser.add_argument('--episode_len', type=int, required=True, help='Length of the episode')
    parser.add_argument('--num_episodes', type=int, default=50, help='Number of episodes to run')
    parser.add_argument('--sim_env', type=str, default='', help='Environment to run like gym_guided_vision/SewNeedle-3Arms-v0')
    args = parser.parse_args()

    def shutdown():
        print("Shutting down...")
        os._exit(42)
    rospy.on_shutdown(shutdown)

    try:
        main(vars(parser.parse_args()))
    except KeyboardInterrupt:
        print("Shutting down...")
        os._exit(42)

"""
python eval.py --policy iantc104/sim_slot_insertion_3arms_zed_wrist_act --episode_len 300 --num_episodes 50 --sim_env gym_guided_vision/SlotInsertion-3Arms-v0

python lerobot/lerobot/scripts/eval.py \
    -p iantc104/gv_sim_sew_needle_3arms_zed_wrist_act \
    eval.n_episodes=10 \
    eval.batch_size=10
"""