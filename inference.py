import os 
from huggingface_hub import snapshot_download

#checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_panda_lora/pi_experiment/5000")
checkpoint_root = snapshot_download(
    repo_id="eyerisshe/vla-checkpoints",
    repo_type="dataset",
    allow_patterns=["pi0_panda_lora/5000/**"],
    local_dir="./maniskill-checkpoints",
    local_dir_use_symlinks=False
)

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np 
import torch
from mani_skill.utils.wrappers.record import RecordEpisode
import gymnasium as gym

config = _config.get_config("pi0_panda_lora")
checkpoint_dir = os.path.join(checkpoint_root, "pi0_panda_lora", "5000")
#checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
policy = policy_config.create_trained_policy(config, checkpoint_dir)


env = gym.make(
    "StackCube-v1", # Other benchmarks include "PushCube-v1", "PegInsertionSide-v1" and others
    
    render_mode = "human",
    robot_uids="panda",
    num_envs=1,
    obs_mode="rgb", 
)

env = RecordEpisode(
    env,
    output_dir="videos", 
    video_fps=30,
    save_video = True,
    max_steps_per_video=300 # Save video every X steps
)  

obs, _ = env.reset()

for _ in range(10):
    action = policy.infer(obs)["actions"] # Sample action from π0
    count = 0
    while (count < 30):
        for i in action:
            obs, rew, terminated, truncated, info = env.step(i)
            count = count + 1
