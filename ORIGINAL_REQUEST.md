# Original User Request

## Initial Request — 2026-10-24T14:50:10Z

Resolve Khepera IV simulation physical ramp climbing mechanics (chassis clearance, caster collisions, ramp lip bevel) and implement robust bump/stall detection using only the onboard sensor suite (IMU accelerometer and IR/US proximity sensors) to eliminate odometry runaway when the robot contacts obstacles, fully accounting for surface slipping and glancing wall contact.

Working directory: `/home/andy/Desktop/khepera4_ws`
Integrity mode: development

## Requirements

### R1. Onboard Sensor Bump & Surface Slip Detection (Eliminate Phantom Forward Travel)
Detect when the robot is bumping into walls or obstacles, taking into account both total stalls and glancing surface slip:
- Leverage the onboard 6-DOF IMU accelerometer (`/imu/data` linear acceleration and jerk) combined with the 8 IR ring sensors (`/khepera4/ir_{0..7}/scan`) and 5 Ultrasonic sensors (`/khepera4/us_{0..4}/scan`):
  - When proximity distance drops to near contact (< 0.015m) or acceleration indicates a stationary impact while `/cmd_vel` commands forward motion, detect the collision.
  - **Surface Slip Handling:** Account for partial or glancing slip against surfaces (where the robot scrapes or slides along a wall at an angle). Rather than an all-or-nothing freeze, compute the surface normal from proximity sensors and project actual allowed displacement tangentially, while strictly nullifying phantom displacement perpendicular to the contacting surface. If wheels spin completely in place with zero physical translation (total slip), clamp advancement to zero.
- In `src/khepera4_control/khepera4_control/khepera4_mapper.py`: integrate this slip-aware projection so the dashboard pose never penetrates walls or stretches phantom corridors.
- In `src/khepera4_control/khepera4_control/reactive_explorer.py`: immediately trigger evasive wall-following or backup-and-turn maneuvers when glancing or head-on contact is detected.

### R2. Ramp Climbing Physics & Caster/Chassis Geometry
Enable the Khepera IV to smoothly angle itself and climb 10° test ramps:
- In `src/khepera4_description/urdf/khepera4.urdf` and `khepera4.urdf.xacro`:
  - Refine the `base_link` collision geometry so the front edge does not form a blunt vertical lip extending beyond the front caster at ground clearance height. Chamfer/bevel or elevate the front lower edge so the robot can pitch/angle upward from the ground.
  - Verify and calibrate the passive front and rear caster rollers (spheres / friction) so they make primary ground contact and roll smoothly up inclines without catching.
- In `src/khepera4_description/worlds/khepera4_exploration_world.sdf`:
  - Ensure the approach lip of the inclined ramp meets the ground with a seamless knife-edge transition ($z=0$ at the contact boundary) so wheels and casters transition smoothly onto the incline without hitting a vertical step.

### R3. Dashboard & SLAM Consistency
Ensure the dedicated mapping dashboard (`mapping_dashboard.py`) and RViz reflect accurate robot positioning without phantom corridor elongation when contacting obstacles, scraping walls, or climbing ramps.

## Acceptance Criteria

### Ramp Climbing Verification
- [ ] Khepera IV successfully drives forward from flat floor onto the inclined ramp ($10^\circ$) at nominal cruising speed ($0.12\text{ m/s}$) without halting or wedging at the bottom lip.
- [ ] IMU pitch telemetry smoothly transitions from $\sim 0^\circ$ on flat ground to $\sim 10^\circ$ as the robot climbs the ramp.

### Bump, Stall & Surface Slip Verification
- [ ] When the robot is driven head-on into a rigid arena wall or obstacle box, linear dead-reckoning advance in `khepera4_mapper` stops within $0.03\text{ m}$ of contact, eliminating phantom forward travel.
- [ ] When driving into a wall at an angle (glancing contact / surface slipping), the robot does not penetrate through the wall; any valid motion is projected tangentially along the wall face without accumulating phantom normal displacement.
- [ ] The robot trajectory never penetrates outer arena perimeter walls ($5\text{ m} \times 5\text{ m}$) on the map.
- [ ] In autonomous mode, the robot detects bump/stall/slip within $0.4\text{ s}$ and executes an evasive backup turn away from the obstacle.

### System Verification & Code Quality
- [ ] No non-physical or synthetic sensors added (strictly uses Khepera IV physical hardware suite: 8 IR, 5 US, 4 Ground IR, 6-DOF IMU).
- [ ] Both URDF and exploration world SDF validate with `gz sdf -k`.
- [ ] All unit tests in `src/khepera4_control/test/` pass without regressions (`pytest-3`).
