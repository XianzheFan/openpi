from pathlib import Path
import select
import sys
import time

import cv2
import numpy as np
import yaml


TASK_CONFIG_PATH = Path(__file__).with_name("task_configs.yaml")


def _load_task_configs():
    with TASK_CONFIG_PATH.open("r", encoding="utf-8") as file:
        task_configs = yaml.safe_load(file)
    if not isinstance(task_configs, dict):
        raise ValueError(f"Invalid task config format in {TASK_CONFIG_PATH}")
    return task_configs


TASK_CONFIGS = _load_task_configs()


def get_config(args):
    task_config = TASK_CONFIGS.get(args.task)
    if task_config is None:
        raise ValueError(f"Invalid task name: {args.task}")

    language_instruction = task_config.get("language_instruction")
    left0 = task_config.get("left0")
    right0 = task_config.get("right0")
    if language_instruction is None or left0 is None or right0 is None:
        raise ValueError(f"Task config for {args.task} is missing required fields")

    state_dim = 14

    config = {
        "episode_len": args.max_publish_step,
        "state_dim": state_dim,
        "left0": left0,
        "right0": right0,
        "action_postprocess": task_config.get("action_postprocess", {}),
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "task": args.task,
        "language_instruction": language_instruction,
        "ctrl_type": args.ctrl_type,
        "chunk_size": args.chunk_size,  # action chunk size from model
        "delay": getattr(args, "delay", None),
        "exec_horizon": getattr(args, "exec_horizon", None),  # execution horizon/inference interval
        "mode": getattr(args, "mode", None),
        "model": getattr(args, "model", None),
    }
    return config


def _apply_gripper_rules(gripper_value, rules):
    for rule in rules:
        condition = rule.get("when")
        threshold = rule.get("threshold")
        if condition == "below" and not gripper_value < threshold:
            continue
        if condition == "above" and not gripper_value > threshold:
            continue

        if "set" in rule:
            gripper_value = rule["set"]
        if "add" in rule:
            gripper_value += rule["add"]
    return gripper_value


def process_action(task, action):
    action = action.copy()
    left_action = action[:7]
    right_action = action[7:14]

    task_config = TASK_CONFIGS.get(task)
    if task_config is None:
        raise ValueError(f"Invalid task name: {task}")

    action_postprocess = task_config.get("action_postprocess", {})
    left_action[6] = _apply_gripper_rules(left_action[6], action_postprocess.get("left_gripper", []))
    right_action[6] = _apply_gripper_rules(right_action[6], action_postprocess.get("right_gripper", []))
    return left_action, right_action


### Utility Functions


def check_keyboard_input():
    """Check if a key was pressed without blocking."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def handle_interactive_mode(task_time):
    """
    Handle interactive mode when space is pressed.
    Returns: 'continue' to resume, 'reset' to restart from beginning
    """
    print("\n" + "=" * 50)
    print(f"Task time: {time.time() - task_time:.1f} s")
    print("INTERACTIVE MODE")
    print("  'c' - Continue running")
    print("  'r' - Reset to starting point and restart")
    print("  'q' - Quit/Stop")
    print("=" * 50)

    while True:
        key = sys.stdin.read(1).lower()
        if key == "c":
            print("Continuing...")
            return "continue"
        elif key == "r":
            print("Restarting...")
            return "reset"
        elif key == "q":
            print("Stopping...")
            return "quit"


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(im: np.ndarray, height: int, width: int, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize one image (H, W, C) to target height/width without distortion by padding with zeros."""
    cur_height, cur_width = im.shape[0], im.shape[1]
    if cur_width == width and cur_height == height:
        return np.ascontiguousarray(im)

    ratio = max(cur_width / width, cur_height / height)
    resized_width = int(cur_width / ratio)
    resized_height = int(cur_height / ratio)

    resized = cv2.resize(im, (resized_width, resized_height), interpolation=interpolation)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    out = np.zeros((height, width, resized.shape[2]), dtype=im.dtype)
    pad_top = (height - resized_height) // 2
    pad_left = (width - resized_width) // 2
    out[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return out
