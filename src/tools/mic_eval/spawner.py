"""
Gazebo Harmonic Object & Robot Spawner / Controller.

Used exclusively by Parent Orchestrator to control simulation state.
"""

import argparse
import subprocess

COLORS = {
    "red": (1.0, 0.1, 0.1),
    "green": (0.1, 0.9, 0.1),
    "blue": (0.1, 0.2, 0.9),
    "yellow": (0.9, 0.9, 0.1),
    "cyan": (0.1, 0.9, 0.9),
}


def generate_box_sdf(name: str, x: float, y: float, z: float,
                     lx: float, ly: float, lz: float,
                     color_name: str = "red", yaw: float = 0.0) -> str:
    r, g, b = COLORS.get(color_name.lower(), (1.0, 0.1, 0.1))
    return f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <pose>{x} {y} {z} 0 0 {yaw}</pose>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box><size>{lx} {ly} {lz}</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>{lx} {ly} {lz}</size></box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def spawn_entity(world: str, name: str, sdf_str: str,
                 x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 yaw: float = 0.0) -> bool:
    # Use ros_gz_sim create with explicit coordinates
    cmd = [
        "ros2", "run", "ros_gz_sim", "create",
        "-world", world,
        "-name", name,
        "-string", sdf_str,
        "-x", str(x),
        "-y", str(y),
        "-z", str(z),
        "-Y", str(yaw),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[SPAWNER] Successfully spawned {name} at ({x}, {y}, {z}) in {world}")
        return True
    else:
        print(f"[SPAWNER] Error spawning {name}: {res.stderr}")
        return False


def remove_entity(world: str, name: str) -> bool:
    req = f'name: "{name}", type: MODEL'
    cmd = [
        "gz", "service", "-s", f"/world/{world}/remove",
        "--reqtype", "gz.msgs.Entity",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "2000",
        "--req", req
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[SPAWNER] Remove {name}: {res.stdout.strip()}")
    return "data: true" in res.stdout


def set_robot_pose(
    world: str, name: str, x: float, y: float, z: float = 0.02, yaw: float = 0.0
) -> bool:
    import math
    half_yaw = yaw / 2.0
    qw = math.cos(half_yaw)
    qz = math.sin(half_yaw)
    req = (
        f'name: "{name}", position: {{x: {x}, y: {y}, z: {z}}}, '
        f'orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}'
    )
    cmd = [
        "gz", "service", "-s", f"/world/{world}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "2000",
        "--req", req
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"[SPAWNER] Set pose {name} to ({x}, {y}, yaw={yaw}): {res.stdout.strip()}")
    return "data: true" in res.stdout


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    spawn_p = subparsers.add_parser("spawn")
    spawn_p.add_argument("--world", default="khepera4_world")
    spawn_p.add_argument("--name", default="target_box")
    spawn_p.add_argument("--x", type=float, default=1.5)
    spawn_p.add_argument("--y", type=float, default=0.0)
    spawn_p.add_argument("--z", type=float, default=0.15)
    spawn_p.add_argument("--lx", type=float, default=0.20)
    spawn_p.add_argument("--ly", type=float, default=0.20)
    spawn_p.add_argument("--lz", type=float, default=0.30)
    spawn_p.add_argument("--yaw", type=float, default=0.0)
    spawn_p.add_argument("--color", default="red")

    rem_p = subparsers.add_parser("remove")
    rem_p.add_argument("--world", default="khepera4_world")
    rem_p.add_argument("--name", default="target_box")

    pose_p = subparsers.add_parser("set_pose")
    pose_p.add_argument("--world", default="khepera4_world")
    pose_p.add_argument("--name", default="khepera4")
    pose_p.add_argument("--x", type=float, default=0.0)
    pose_p.add_argument("--y", type=float, default=0.0)
    pose_p.add_argument("--z", type=float, default=0.02)
    pose_p.add_argument("--yaw", type=float, default=0.0)

    args = parser.parse_args()
    if args.command == "spawn":
        sdf = generate_box_sdf(args.name, args.x, args.y, args.z, args.lx, args.ly, args.lz, args.color, args.yaw)
        spawn_entity(args.world, args.name, sdf, args.x, args.y, args.z, args.yaw)
    elif args.command == "remove":
        remove_entity(args.world, args.name)
    elif args.command == "set_pose":
        set_robot_pose(args.world, args.name, args.x, args.y, args.z, args.yaw)
