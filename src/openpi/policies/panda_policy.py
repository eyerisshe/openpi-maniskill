import dataclasses
from typing import ClassVar
import numpy as np
from openpi import transforms
from openpi.models import model as _model

@dataclasses.dataclass(frozen=True)
class PandaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:

        ### EDIT PROMPT BASED ON TASK
        prompt = "pick up red cube"

        ### IMAGE
        # "panda_wristcam" robot in Maniskill has an additional hand_camera, if you wish to use 
            # obs["sensor_data"]["3rd_view_camera"]["rgb"]
            # However, fine-tuning data from the RDT team does not have wrist images so I stuck with that format
        # Maniskill images already in np.ndarray of shape (H, W, 3), uint8
            # i.e. (128, 128, 3) for PickCube-v1
        base_image = np.asarray(data["sensor_data"]["base_camera"]["rgb"].cpu()) # Gymnasium during inference returns data["sensor_data"]["base_camera"]["rgb"]
        base_image = np.squeeze(base_image, axis=0)
  
        ### STATE
        # π0 accepts 8D array, gripper is 8th dimension and normalized to [0,1]
        # Maniskill obs["agent"] is outputting 9D state (7 joints + 2 grippers)
        # Can get rid of second gripper value because it is a mimic of the first 
        qpos = np.asarray(data["agent"]["qpos"]) # Gymnasium during inference returns data["agent"]["qpos"]
        qpos = np.squeeze(qpos, axis=0)
      
        # Gripper range from panda URDF
        gripper_min = 0.00   # Lower
        gripper_max = 0.04   # Upper

        arm_joints = qpos[:7]
        gripper_joints = qpos[7]

        gripper_pos_normalized = (gripper_joints - gripper_min) / (gripper_max - gripper_min)
        gripper_pos_normalized = np.clip(gripper_pos_normalized, 0.0, 1.0)
        gripper = gripper_pos_normalized[np.newaxis]

        state_8d = np.concatenate([arm_joints, gripper])

        inputs = {
            "state": state_8d,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
            "prompt": prompt,
        }

        # Maniskill obs does not provide action and prompt during inference
        # Use during training
        if "actions" in data:
            inputs["actions"] = data["actions"]

        return inputs

@dataclasses.dataclass(frozen=True)
class PandaOutputs(transforms.DataTransformFn):
    
    def __call__(self, data: dict) -> dict:
        pi0_actions = np.asarray(data["actions"])
  
        gripper_min = 0.00   
        gripper_max = 0.04   

        arm_actions = pi0_actions[:, :7]
        gripper_normalized = pi0_actions[:, 7]
        
        gripper_meters = gripper_min + gripper_normalized * (gripper_max - gripper_min)
        gripper_actions = (gripper_meters[np.newaxis]).T
        
        panda_actions = np.concatenate([arm_actions, gripper_actions], axis=1)

        return {"actions": panda_actions} 

# Maniskill env.step only expects one action vector
# In π0:
    #  For the 20Hz UR5e and Franka robots, 
    # We run inference every 0.8 seconds (after executing 16 actions)
    # Execute 16 actions from the 50-step action chunk before new inference call is made
