#!/usr/bin/env python3
"""
Trial Orchestration Utility.

Executed by Parent Orchestrator to run each trial step-by-step.
"""

import json
import os
import subprocess
import sys
import time


def run_trial(trial_id: str, x_gt: float, y_gt: float, z_gt: float,
              lx: float, ly: float, lz: float,
              x_0: float = 0.0, y_0: float = 0.0, yaw_0: float = 0.0,
              has_distractor: bool = False,
              distractor_x: float = 0.0, distractor_y: float = 0.0):

    clean_id = str(trial_id).removeprefix("trial_")
    summary_path = f"/tmp/mic_eval/trial_{clean_id}_summary.json"
    if os.path.exists(summary_path):
        os.remove(summary_path)

    # 1. Clean previous target / distractor objects in Gazebo
    subprocess.run(["python3", "/tmp/mic_eval/spawner.py", "remove", "--name", "target_box"], capture_output=True)
    subprocess.run(["python3", "/tmp/mic_eval/spawner.py", "remove", "--name", "distractor_box"], capture_output=True)
    subprocess.run(["python3", "/tmp/mic_eval/spawner.py", "remove", "--name", "test_target"], capture_output=True)

    # 2. Reset robot pose
    subprocess.run(["python3", "/tmp/mic_eval/spawner.py", "set_pose", "--name", "khepera4",
                    "--x", str(x_0), "--y", str(y_0), "--yaw", str(yaw_0)], capture_output=True)

    # 3. Spawn target object
    subprocess.run(["python3", "/tmp/mic_eval/spawner.py", "spawn", "--name", "target_box",
                    "--x", str(x_gt), "--y", str(y_gt), "--z", str(z_gt),
                    "--lx", str(lx), "--ly", str(ly), "--lz", str(lz),
                    "--color", "red"], capture_output=True)

    # If distractor requested:
    if has_distractor:
        subprocess.run(["python3", "/tmp/mic_eval/spawner.py", "spawn", "--name", "distractor_box",
                        "--x", str(distractor_x), "--y", str(distractor_y), "--z", str(z_gt),
                        "--lx", str(lx), "--ly", str(ly), "--lz", str(lz),
                        "--color", "blue"], capture_output=True)

    # 4. Write current_trial.json for mic_probe
    trial_data = {
        "trial_id": clean_id,
        "x_gt": x_gt,
        "y_gt": y_gt,
        "z_gt": z_gt,
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "x_0": x_0,
        "y_0": y_0,
        "yaw_0": yaw_0
    }
    with open("/tmp/mic_eval/current_trial.json", "w") as f:
        json.dump(trial_data, f, indent=2)

    print(f"[ORCHESTRATOR] Trial {clean_id} initialized. Triggering pursuit and exploration...")

    # 5. Send initial mode, prompt, and enable triggers
    # Clear any terminal stop latch
    subprocess.run(["ros2", "topic", "pub", "-t", "5", "-r", "10", "/target_found_and_stopped", "std_msgs/msg/Bool", '{data: false}'], capture_output=True)
    # Publish mode: semantic
    subprocess.run(["ros2", "topic", "pub", "-t", "5", "-r", "10", "/pursuit/mode", "std_msgs/msg/String", '{data: "semantic"}'], capture_output=True)
    # Publish prompt: red box
    subprocess.run(["ros2", "topic", "pub", "-t", "5", "-r", "10", "/pursuit/target_prompt", "std_msgs/msg/String", '{data: "red box"}'], capture_output=True)
    # Publish pursuit enable: True
    subprocess.run(["ros2", "topic", "pub", "-t", "5", "-r", "10", "/pursuit/enable", "std_msgs/msg/Bool", '{data: true}'], capture_output=True)

    # 6. Monitor trial execution (up to 125s max)
    start_time = time.time()
    while time.time() - start_time < 130.0:
        if os.path.exists(summary_path):
            print(f"[ORCHESTRATOR] Summary file detected for trial {clean_id}!")
            time.sleep(1.0)  # allow final file sync
            break
        time.sleep(2.0)

    # 7. Post-trial cleanup: disable pursuit & zero cmd_vel
    subprocess.run(
        ["ros2", "topic", "pub", "-t", "3", "-r", "10",
         "/pursuit/enable", "std_msgs/msg/Bool", '{data: false}'],
        capture_output=True,
    )
    subprocess.run(
        ["ros2", "topic", "pub", "-t", "3", "-r", "10",
         "/cmd_vel", "geometry_msgs/msg/Twist", '{}'],
        capture_output=True,
    )

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        print(f"[ORCHESTRATOR] Trial {clean_id} complete. Outcome: {summary.get('status')}")
        return True
    else:
        print(f"[ORCHESTRATOR] Trial {clean_id} failed to produce summary within timeout.")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", default="1")
    parser.add_argument("--x", type=float, default=1.5)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--z", type=float, default=0.15)
    parser.add_argument("--lx", type=float, default=0.20)
    parser.add_argument("--ly", type=float, default=0.20)
    parser.add_argument("--lz", type=float, default=0.30)
    parser.add_argument("--distractor", action="store_true")
    parser.add_argument("--dx", type=float, default=2.0)
    parser.add_argument("--dy", type=float, default=-0.5)

    args = parser.parse_args()
    success = run_trial(
        trial_id=args.trial,
        x_gt=args.x,
        y_gt=args.y,
        z_gt=args.z,
        lx=args.lx,
        ly=args.ly,
        lz=args.lz,
        has_distractor=args.distractor,
        distractor_x=args.dx,
        distractor_y=args.dy
    )
    sys.exit(0 if success else 1)
