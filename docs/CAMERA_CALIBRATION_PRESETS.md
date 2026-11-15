# Khepera IV Camera Extrinsics & Projection Presets

This document specifies the camera geometry presets for the Khepera IV object navigation and spatial projection stack, linking directly to the mathematical formulations in [`visual_orthomapper.py`](file:///home/andy/Desktop/khepera4_ws/src/khepera4_control/khepera4_control/visual_orthomapper.py#L440-L470) and configuration files.

---

## 1. Quick Preset Comparison

| Parameter | Preset 1: Built-in CMOS Camera (ACTIVE) | Preset 2: Elevated Turret Mount |
| :--- | :--- | :--- |
| **Physical Location** | Inside front robot chassis | Top deck / elevated mast |
| **`camera_height` ($h$)** | **`0.0283` m ($2.83$ cm)** | `0.125` m ($12.5$ cm) |
| **`camera_pitch_deg` ($\theta$)** | **`0.0` deg (horizontal forward)** | `18.5` deg (tilted downward) |
| **`camera_forward_offset` ($d_x$)** | **`0.0682` m ($6.82$ cm forward)** | `0.0682` m ($6.82$ cm forward) |
| **Horizon Row ($v_{\text{horizon}}$)** | **`240.0` px ($c_y$)** | `54.5` px |
| **Horizon Margin** | **`10.0` px (requires $v \ge 250$)** | `10.0` px (requires $v \ge 65$) |
| **`min_valid_distance`** | **`0.1338` m (at bottom row $v = 479$)** | `0.12` m |
| **`max_valid_distance`** | **`1.50` m (at cutoff row $v = 251$)** | `2.50` m |
| **Reference Implementation** | `visual_orthomapper.py` (L440–L470, L1230–L1375) | Top turret visual servoing |

---

## 2. Mathematical Derivation for Built-in Camera ($\theta = 0^\circ, h = 0.0283\text{ m}$)

### Optical-to-Base Rotation Matrix
When the camera is mounted horizontal ($\theta = 0^\circ$):
$$R_{\text{base\_opt}} = \begin{bmatrix} 0 & 0 & 1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix}$$

Given normalized optical ray $\mathbf{p}_{\text{opt}} = K^{-1} [u, v, 1]^T = \left[ \frac{u - c_x}{f_x}, \frac{v - c_y}{f_y}, 1 \right]^T$:
$$\mathbf{r}_B = R_{\text{base\_opt}} \mathbf{p}_{\text{opt}} = \begin{bmatrix} 1 \\ -\frac{u - c_x}{f_x} \\ -\frac{v - c_y}{f_y} \end{bmatrix}$$

### Ground Plane Intersection ($Z_B = 0$)
From camera origin $\mathbf{o}_B = [d_x, 0, h]^T$:
$$Z_B = h - \lambda \left( \frac{v - c_y}{f_y} \right) = 0 \implies \lambda = \frac{h \cdot f_y}{v - c_y}$$
$$X_B = d_x + \lambda = d_x + \frac{h \cdot f_y}{v - c_y}$$
$$Y_B = -\lambda \left( \frac{u - c_x}{f_x} \right) = -\frac{h \cdot f_y}{v - c_y} \left( \frac{u - c_x}{f_x} \right)$$

### Dependent Variables Breakdown:
1. **Horizon Line ($v_{\text{horizon}}$):**
   $$v_{\text{horizon}} = c_y - f_y \tan(0^\circ) = c_y = 240.0\text{ px}$$
   Any pixel with $v \le 240$ points horizontal or upward and can never intersect the ground.
2. **Closest Floor Point (`min_valid_distance`):**
   Occurs at the bottom image row ($v = 479$):
   $$\text{dist}_{c} = \frac{0.0283 \cdot 554.256}{479 - 240} = \frac{15.685}{239} = 0.0656\text{ m}$$
   $$X_B = d_x + \text{dist}_c = 0.0682 + 0.0656 = \mathbf{0.1338\text{ m}}$$
   *(Directly matches `visual_orthomapper.py` L455 `min_ground_dist: 0.1338`)*.
3. **Far Clipping Point (`max_valid_distance`):**
   At row $v = 251$ (11 pixels below the horizon line):
   $$\text{dist}_c = \frac{15.685}{251 - 240} = \frac{15.685}{11} \approx 1.426\text{ m} \implies X_B \approx \mathbf{1.50\text{ m}}$$
   *(Directly matches `visual_orthomapper.py` L456 `max_ground_dist: 1.50` and L457 `min_row_v: 251`)*.

---

## 3. How to Adjust or Toggle Presets

All nodes read these parameters directly from configuration YAMLs:

* **Simulation:** [`khepera_object_nav_bringup/config/fetch_sim_params.yaml`](file:///home/andy/Desktop/khepera4_ws/src/khepera_object_nav_bringup/config/fetch_sim_params.yaml#L20-L40)
* **Physical Robot:** [`khepera_object_nav_bringup/config/fetch_params.yaml`](file:///home/andy/Desktop/khepera4_ws/src/khepera_object_nav_bringup/config/fetch_params.yaml#L20-L40)
* **Spatial Projector Default:** [`khepera_spatial_projector/config/projector_params.yaml`](file:///home/andy/Desktop/khepera4_ws/src/khepera_spatial_projector/config/projector_params.yaml#L9-L30)

### Switching at Runtime via ROS 2 CLI:
```bash
ros2 param set /khepera_spatial_projector camera_height 0.0283
ros2 param set /khepera_spatial_projector camera_pitch_deg 0.0
ros2 param set /khepera_spatial_projector min_valid_distance 0.1338
ros2 param set /khepera_spatial_projector max_valid_distance 1.50
```
