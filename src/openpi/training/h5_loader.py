"""
SCRIPT COURTESY OF RDT-1B TEAM
Copied from [https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/main/data/hdf5_maniskill_dataset.py]
Adapted to fit π0
"""

import os
import h5py
import yaml
import numpy as np
# Assuming STATE_VEC_IDX_MAPPING is a dictionary mapping state variable names to indices
import openpi.training.state_vec as STATE_VEC_IDX_MAPPING
import glob
from scipy.interpolate import interp1d
from PIL import Image
from datasets import load_dataset

# Loading in h5 and .png
from huggingface_hub import snapshot_download

def interpolate_action_sequence(action_sequence, target_size):
    """
    Extend the action sequece to `target_size` by linear interpolation.
    
    Args:
        action_sequence (np.ndarray): original action sequence, shape (N, D).
        target_size (int): target sequence length.
    
    Returns:
        extended_sequence (np.ndarray): extended action sequence, shape (target_size, D).
    """
    N, D = action_sequence.shape
    indices_old = np.arange(N)
    indices_new = np.linspace(0, N - 1, target_size)

    interp_func = interp1d(indices_old, action_sequence, 
                           kind='linear', axis=0, assume_sorted=True)
    action_sequence_new = interp_func(indices_new)

    return action_sequence_new


class HDF5VLADataset:
    """
    This class is used to sample episodes from the embodiment dataset
    stored in HDF5 files.
    """
    def __init__(self):
    
        # Multiple tasks
        self.tasks = ['PickCube-v1', 'StackCube-v1']  # RDT-1 Team also included PlugCharger-v1 and PegInsertionSide-v1 and PushCube-v1
        # Load configuration from YAML file
        self.CHUNK_SIZE = 30
        self.IMG_HISTORY_SIZE = 2
        self.STATE_DIM = 128

        self.num_episode_per_task = 1000
        self.img = []
        self.state = []
        self.action = []

        # open the hdf5 files in memory to speed up the data loading
        for task in self.tasks:
    
            file_path = os.path.join(
                "src",
                "openpi",
                "training",
                "maniskill_data",
                "demo_1k",
                task,
                "motionplanning",
                f"{task}.h5"
            )

            with h5py.File(file_path, "r") as f:
                trajs = f.keys() #  traj_0, traj_1,
                # sort by the traj number
                trajs = sorted(trajs, key=lambda x: int(x.split('_')[-1]))
                for traj in trajs:
                    # images = f[traj]['obs']['sensor_data']['base_camera']['rgb'][:]
                    states = f[traj]['obs']['agent']['qpos'][:]
                    actions = f[traj]['actions'][:]

                    self.state.append(states)
                    self.action.append(actions)
                    # self.img.append(images)
        
        self.state_min = np.concatenate(self.state).min(axis=0)
        self.state_max = np.concatenate(self.state).max(axis=0)
        self.action_min = np.concatenate(self.action).min(axis=0)
        self.action_max = np.concatenate(self.action).max(axis=0)
        self.action_std = np.concatenate(self.action).std(axis=0)
        self.action_mean = np.concatenate(self.action).mean(axis=0)
                    
        self.task2lang = {
          #  "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
            "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
            "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
          #  "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
          #  "PushCube-v1": "Push and move a cube to a goal region in front of it."
        }

    def __len__(self):
        # Assume each file contains 100 episodes
        return len(self.tasks) * self.num_episode_per_task

    def __getitem__(self, index):
        """
        Fetch a training sample at the given index.

        Args:
            index (int): The index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing the state, action, image, etc. for the sample.
        """
        num_steps = len(self.action[index])
        step_index = np.random.randint(0, num_steps)
        task_index = index // self.num_episode_per_task
        language = self.task2lang[self.tasks[task_index]]
        task_inner_index = index % self.num_episode_per_task

        # Revised dataset does not have PegInsertionSide-v1
        # Skip these episodes since in the eef version dataset they are invalid.
        # if self.tasks[task_index] == 'PegInsertionSide-v1' and task_inner_index > 400:
        #     return False, None

        proc_index = task_inner_index // 100
        episode_index = task_inner_index % 100
        #  images0 = self.img[index]
        #   normalize to -1, 1
        
        states = (self.state[index] - self.state_min) / (self.state_max - self.state_min) * 2 - 1
        states = states[:, :-1]  # remove the last state as it is replicate of the -2 state
        actions = (self.action[index] - self.action_min) / (self.action_max - self.action_min) * 2 - 1
        

        # Create the image history (as done in `parse_hdf5_file`)
    
        img_history = []
        start_img_idx = max(0, step_index - self.IMG_HISTORY_SIZE + 1)
        end_img_idx = step_index + 1
        
        for i in range(start_img_idx, end_img_idx):
            image_path = os.path.join(
                "src",
                "openpi",
                "training",
                "maniskill_data",
                "demo_1k",
                self.tasks[task_index],
                "motionplanning",
                str(proc_index),
                str(episode_index),
                f"{i+1}.png"
            )

            if os.path.exists(image_path):
                img_history = np.array(Image.open(image_path))
            else:
            # If no episode found, skip this one
                img_history = np.zeros((64, 64, 3), dtype=np.uint8)  # or whatever shape you expect


        ############## Batching works differently in π0
            #img = np.array(Image.open(image_path))
          #  img_history.append(img)
        #img_history = np.array(img_history)
       # img_history = img  
        
        # img_history = images0[start_img_idx:end_img_idx]
        #  img_valid_len = img_history.shape[0]

        # Pad images if necessary
        # if img_valid_len < self.IMG_HISTORY_SIZE:
        #     padding = np.tile(img_history[0:1], (self.IMG_HISTORY_SIZE - img_valid_len, 1, 1, 1))
        #     img_history = np.concatenate([padding, img_history], axis=0)
      #  img_history_mask = np.array([True] * img_valid_len)
      

        # Compute state statistics
        state_std = np.std(states, axis=0)
        state_mean = np.mean(states, axis=0)
        state_norm = np.sqrt(np.mean(states ** 2, axis=0))

        # Get state and action at the specified timestep
       # state = states[step_index: step_index + 1]
        state = states[step_index] # Batching works differently in π0
        runtime_chunksize = self.CHUNK_SIZE // 4
        action_sequence = actions[step_index: step_index + runtime_chunksize]
        # we use linear interpolation to pad the action sequence

        # Pad action sequence if necessary
        if action_sequence.shape[0] < runtime_chunksize:
            padding = np.tile(action_sequence[-1:], (runtime_chunksize - action_sequence.shape[0], 1))
            action_sequence = np.concatenate([action_sequence, padding], axis=0)

        action_sequence = interpolate_action_sequence(action_sequence, self.CHUNK_SIZE)


        return {
            "state": state,
            "actions": action_sequence,
            "image": img_history,
            "wrist_image": np.zeros((self.IMG_HISTORY_SIZE, 0, 0, 0)),
            "task": language
        }

# if __name__ == "__main__":
#     from PIL import Image
    
  #  ds = HDF5VLADataset()

    # json_data = {
    #     'state_min': ds.state_min.tolist(),
    #     'state_max': ds.state_max.tolist(),
    #     'action_min': ds.action_min.tolist(),
    #     'action_max': ds.action_max.tolist(),
    # }
    # print(json_data)
