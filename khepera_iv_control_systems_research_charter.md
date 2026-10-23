# Project Charter & Research Roadmap
## Closed-Loop Control Architecture for the Khepera IV Differential-Drive Robot
**Course:** Bazele Sistemelor Automate (BSA)  
**Frameworks:** ROS 2 (Humble / Iron), Gazebo Physics Simulator, Physical Khepera IV Hardware  

---

## 1. Executive Summary & Core Objective

The central objective of this research paper is to bridge the gap between **abstract continuous-time control theory** and **applied cyber-physical robotics**. 

Using the Khepera IV mobile robot as an experimental platform, this project investigates how classical automation control loops—specifically negative feedback, cascade regulation, and frequency-domain stability criteria—are modeled analytically, validated numerically in a high-fidelity physics simulator (Gazebo), orchestrated via distributed robotics middleware (ROS 2), and experimentally verified on physical hardware in the laboratory.

```
+---------------------------------------------------------------------------------+
|                                CONTROL SYSTEM HIERARCHY                         |
|                                                                                 |
|  [Reference]      Outer Loop: Pose / Trajectory Tracking (Kinematic)            |
|  (x_ref, y_ref)   e(t) = r(t) - y(t) ------> Controller C_p(s) ------> [v*, w*] |
|       |                                                                   |     |
|       +-------------------------------------------------------------------+     |
|                                                                           |     |
|                   Inner Loop: Actuator Velocity Control (Dynamic)         v     |
|                   [w_R*, w_L*] ----> Controller C_v(s) ----> [u_R, u_L] (PWM/V) |
|                                                                   |             |
|                                                                   v             |
|                                                     +------------------------+  |
|                                                     | Physical / Sim Plant   |  |
|                                                     | Khepera IV Actuators   |  |
|                                                     +------------------------+  |
|                                                                   |             |
|  [Feedback Loop: Encoders, Odometry, IMU] <-----------------------+             |
+---------------------------------------------------------------------------------+
```

---

## 2. Key Academic & Engineering Objectives

### Objective 1: First-Principles Mathematical Modeling
* **Kinematic Plant Formulation:** Derive the direct and inverse kinematics of the non-holonomic unicycle/differential-drive platform, mapping chassis velocities $(v, \omega)$ to individual wheel angular velocities $(\omega_R, \omega_L)$.
* **Electromechanical Actuator Dynamics:** Derive the differential equations governing the Khepera IV coreless DC motors, accounting for armature resistance ($R_a$), armature inductance ($L_a$), motor torque constant ($K_t$), back-EMF constant ($K_e$), equivalent rotor inertia ($J_{eq}$), and viscous friction ($b_{eq}$).
* **Continuous-Time Transfer Functions ($H(s)$):** Construct the open-loop and closed-loop transfer functions:
  $$H_m(s) = \frac{\Omega_m(s)}{U(s)} = \frac{K_m}{\tau_m s + 1}$$
  evaluating the motor plant as a first-order or second-order lag system.

### Objective 2: Automation Schemas & Cascade Feedback Design
* **Structural Block Diagrams (*Scheme Bloc*):** Design a hierarchical cascade control schema:
  1. **Inner Loop (Fast Dynamics, $\sim 100\text{ Hz}$):** Independent wheel angular velocity regulation ($\omega_R, \omega_L$) via PI controllers to actively reject load torque disturbances, gearbox backlash, and wheel-surface friction.
  2. **Outer Loop (Slow Dynamics, $\sim 10\text{--}20\text{ Hz}$):** Non-linear posture and path-following regulation $(x, y, \theta)$ generating velocity setpoints $(v^*, \omega^*)$.
* **Error Formulation:** Model error dynamics in both the Cartesian inertial frame and the robot-centric moving frame ($e_x, e_y, e_\theta$).

### Objective 3: Controller Synthesis & Formal Stability Analysis
* **Analytical Tuning:** Synthesize controllers (P, PI, PID, or State-Feedback via pole placement) using classical criteria taught in BSA:
  * Pole-zero cancellation.
  * Optimal damping ratio selection ($\zeta = \frac{\sqrt{2}}{2} \approx 0.707$) for a maximum overshoot $\sigma\% \le 4.3\%$.
* **Algebraic Stability Verification:** Apply the **Routh-Hurwitz criterion** to the characteristic polynomial $P(s) = \text{denom}(H_{cl}(s))$ to establish bounded-input bounded-output (BIBO) stability boundaries for controller gains ($K_p, T_i, T_d$).
* **Frequency-Domain Analysis:** Construct asymptotic and real **Bode plots** and **Nyquist diagrams** for the open-loop system $H_o(s) = C(s)G(s)$. Derive:
  * Gain Margin ($K_m$ / *Rezerva de amplificare*).
  * Phase Margin ($\gamma_k$ / *Rezerva de fază*).
  * Cutoff frequency ($\omega_c$) and closed-loop bandwidth.

### Objective 4: Virtual Prototyping in Gazebo & ROS 2
* **URDF/SDF Kinematic & Dynamic Description:** Accurately configure the virtual Khepera IV model:
  * Mass distribution ($m \approx 0.540\text{ kg}$).
  * Principal moments of inertia ($I_{xx}, I_{yy}, I_{zz}$).
  * Wheel radius ($r = 0.021\text{ m}$) and track wheelbase ($L = 0.1054\text{ m}$).
  * Ground contact coefficients (Coulomb and viscous friction parameters).
* **ROS 2 Computation Graph:** Build modular nodes communicating over DDS:
  * Trajectory/Setpoint Generator node.
  * Feedback Controller node subscribing to `/odom` (`nav_msgs/msg/Odometry`) and computing control efforts.
  * Command publisher outputting `/cmd_vel` (`geometry_msgs/msg/Twist`).
  * Telemetry monitoring nodes recording transient time responses.

### Objective 5: Sim-to-Real Gap Analysis & Lab Validation
* Deploy the identical ROS 2 controller node to the physical Khepera IV embedded computer in the lab.
* Empirically measure the discrepancies between the simulated plant and real physical hardware:
  * Odometry drift caused by non-ideal wheel slippage on the lab floor.
  * Actuator deadband and static friction ($u_{\text{deadband}}$).
  * Sensor quantization noise (wheel optical encoders) and communication latency over Wi-Fi/Bluetooth.

---

## 3. Methodological Structure of the Research Paper

The research paper must strictly adhere to an academic engineering format (e.g., IEEE Transactions style):

| Section | Title | Primary Theoretical & Practical Contents |
| :--- | :--- | :--- |
| **I** | **Introduction & Problem Formulation** | Industrial/academic relevance of differential-drive automation; open-loop vulnerability vs. closed-loop robustness; problem statement. |
| **II** | **Physical System & Kinematic Foundations** | Geometric parameters of Khepera IV; transformation matrices; non-holonomic constraints ($\dot{x}\sin\theta - \dot{y}\cos\theta = 0$); forward/inverse kinematics. |
| **III** | **Actuator Dynamics & Mathematical Modeling** | Electrical and mechanical balance equations of the DC motors; reduction gear ratio ($N$); plant transfer functions in the Laplace domain $H(s)$. |
| **IV** | **Automation Architecture & Block Diagrams** | Complete block diagrams (*scheme bloc*); inner-loop velocity control vs. outer-loop pose control; transfer matrix formulation. |
| **V** | **Controller Synthesis & Stability Analysis** | Controller derivation (PI/PID); Routh-Hurwitz algebraic stability proofs; frequency response (Bode & Nyquist curves); phase and gain margins. |
| **VI** | **Simulation Environment (Gazebo & ROS 2)** | URDF link/joint specifications; physics solver configuration (ODE/DART); ROS 2 node architecture, topics, and QoS policies. |
| **VII** | **Experimental Results & Performance Metrics** | Comparative analysis of step responses: rise time ($t_r$), settling time ($t_s$), overshoot ($\sigma\%$), steady-state error ($e_{st}$); tracking error for circle and lemniscate trajectories. |
| **VIII**| **Sim-to-Real Transfer & Lab Hardware Validation** | Experimental validation on physical Khepera IV; disturbance rejection tests; analysis of friction and slip. |
| **IX** | **Conclusions & Future Research** | Summary of control efficacy; evaluation against BSA course benchmarks; extensions to MPC or nonlinear Lyapunov control. |

---

## 4. Key Automation Formulas & Concepts to Cover

### 1. Actuator Transfer Function (DC Motor)
From Kirchoff's voltage law and D'Alembert's torque balance:
$$u(t) = R_a i_a(t) + L_a \frac{di_a(t)}{dt} + K_e \omega_m(t)$$
$$J_{eq} \frac{d\omega_m(t)}{dt} + b_{eq} \omega_m(t) = K_t i_a(t) - \tau_L(t)$$

Neglecting small armature inductance ($L_a \approx 0$):
$$H_m(s) = \frac{\Omega_m(s)}{U(s)} = \frac{\frac{K_t}{R_a b_{eq} + K_t K_e}}{\frac{R_a J_{eq}}{R_a b_{eq} + K_t K_e}s + 1} = \frac{K_m}{\tau_m s + 1}$$

### 2. Standard Second-Order Performance Indicators
When tuning the closed-loop transfer function $H_{cl}(s) = \frac{\omega_n^2}{s^2 + 2\zeta \omega_n s + \omega_n^2}$:
* **Maximum Overshoot (*Suprareglarea*):**
  $$\sigma = e^{-\frac{\pi \zeta}{\sqrt{1 - \zeta^2}}} \times 100\%$$
* **Settling Time (*Timpul de reglare*, $2\%$ or $5\%$ criterion):**
  $$t_s \approx \frac{4}{\zeta \omega_n} \quad (2\% \text{ criterion})$$
* **Rise Time (*Timpul de creștere*):**
  $$t_r \approx \frac{\pi - \arccos(\zeta)}{\omega_n \sqrt{1 - \zeta^2}}$$

### 3. Stability Margins (Frequency Domain)
* **Gain Crossover Frequency ($\omega_c$):** $|H_o(j\omega_c)| = 1 \implies 20\log_{10}|H_o(j\omega_c)| = 0\text{ dB}$
* **Phase Margin ($\gamma_k$):** $\gamma_k = 180^\circ + \arg[H_o(j\omega_c)] \ge 45^\circ\text{--}60^\circ$ (for robust stability).

---

## 5. Next Steps & Project Execution Plan

1. **Step 1: Kinematic & Dynamic Parametrization**
   Extract physical constants from K-Team's Khepera IV manual (mass, wheelbase, gear ratios, encoder ticks per revolution).
2. **Step 2: Continuous-Time Controller Synthesis in MATLAB/Python**
   Derive PI/PID gains analytically to satisfy Prof. Dumitrache’s frequency-domain specifications ($\gamma_k \ge 60^\circ$, zero steady-state error to step inputs).
3. **Step 3: Gazebo & ROS 2 Simulation Setup**
   Construct the URDF model and launch the `diff_drive_controller` plugin in a simulated maze or obstacle-free world.
4. **Step 4: Real-Time Closed-Loop Trajectory Scripting**
   Implement path-following (e.g., straight-line setpoint, circular orbit, square path) and log error metrics ($e(t)$).
5. **Step 5: Physical Lab Testing**
   Run the identical control node on the physical robot, record encoder odometry, and draft the comparative results section of your paper.