import os 
from huggingface_hub import snapshot_download

# UNCOMMENT if use (lightly) fine-tuned Chkpt
# checkpoint_root = snapshot_download(
#     repo_id="eyerisshe/vla-checkpoints",
#     repo_type="dataset",
#     allow_patterns=["pi0_panda_lora/5000/**"],
#     local_dir="./maniskill-checkpoints",
#     local_dir_use_symlinks=False
# )

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np 
import torch
from mani_skill.utils.wrappers.record import RecordEpisode
import gymnasium as gym

checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
config = _config.get_config("pi0_panda")
# UNCOMMENT if use (lightly) fine-tuned Chkpt
# checkpoint_dir = os.path.join(checkpoint_root, "pi0_panda_lora", "5000")
# config = _config.get_config("pi0_panda_lora")

policy = policy_config.create_trained_policy(config, checkpoint_dir)

env = gym.make(
    "StackCube-v1", # Other benchmarks include "PushCube-v1", "PegInsertionSide-v1" and others
    robot_uids="panda",
    num_envs=1,
    render_mode="rgb_array", 
    obs_mode = "rgb"
)

VIDEO_STEPS = 16
EPISODE = 1
H = 16

env = RecordEpisode(
    env,
    output_dir="videos", 
    video_fps=30,
    save_video = True,
    max_steps_per_video=VIDEO_STEPS # Save video every X steps
)  

obs, _ = env.reset()

for _ in range(EPISODE):
    action = policy.infer(obs)["actions"] # Sample action from π0
    count = 0
    while (count < H):
        for i in action: # Sample actions from chunk
            obs, rew, terminated, truncated, info = env.step(i)
            count = count + 1
