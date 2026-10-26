# Control Systems Research Charter: Theory, Kinematics & Dynamic Modeling
**Course / Research Foundation:** Bazele Sistemelor Automate (BSA)  
**Target Platform:** Khepera IV Mobile Robot (Differential Drive Unicycle)  
**Software Stack:** ROS 2 Jazzy Jalisco & Gazebo Harmonic 8.15  

---

## 1. Unicycle Kinematic Model & Polar Representation

The Khepera IV is mathematically modeled as a non-holonomic differential unicycle in the Euclidean plane:

$$\begin{pmatrix} \dot{x} \\ \dot{y} \\ \dot{\theta} \end{pmatrix} = \begin{pmatrix} \cos\theta & 0 \\ \sin\theta & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} v \\ \omega \end{pmatrix}$$

Subject to the non-holonomic pure-rolling, no-lateral-slip constraint:
$$\dot{x}\sin\theta - \dot{y}\cos\theta = 0$$

### 1.1 Polar Coordinates Transformation $(\rho, \alpha, \beta)$
To eliminate orientation singularities when stabilizing the robot to a target setpoint $(x_g, y_g, \theta_g)$, the tracking error is expressed in polar coordinates:

$$\rho = \sqrt{\Delta x^2 + \Delta y^2}$$
$$\alpha = \text{atan2}(\Delta y, \Delta x) - \theta$$
$$\beta = -\theta - \alpha + \theta_g = \theta_g - \text{atan2}(\Delta y, \Delta x)$$

Where $\Delta x = x_g - x$ and $\Delta y = y_g - y$.

The transformed state equations are:
$$\dot{\rho} = -v \cos\alpha$$
$$\dot{\alpha} = \frac{v}{\rho} \sin\alpha - \omega$$
$$\dot{\beta} = -\frac{v}{\rho} \sin\alpha$$

---

## 2. Lyapunov Cascade Closed-Loop Control Law

Consider the candidate Lyapunov function:
$$V(\rho, \alpha, \beta) = \frac{1}{2} \rho^2 + \frac{1}{2} \alpha^2 + \frac{1}{2} \beta^2 > 0 \quad \forall (\rho, \alpha, \beta) \ne (0, 0, 0)$$

Taking the time derivative along the system trajectories:
$$\dot{V} = \rho \dot{\rho} + \alpha \dot{\alpha} + \beta \dot{\beta} = -\rho v \cos\alpha + \alpha \left(\frac{v}{\rho} \sin\alpha - \omega\right) - \beta \frac{v}{\rho} \sin\alpha$$

Setting the control velocities to:
$$v = k_\rho \rho \cos\alpha$$
$$\omega = k_\alpha \alpha + k_\beta \beta + \frac{v}{\rho} \sin\alpha \left(1 - \frac{\beta}{\alpha}\right)$$

Yields the negative semi-definite derivative:
$$\dot{V} = -k_\rho \rho^2 \cos^2\alpha - k_\alpha \alpha^2 \le 0$$

By LaSalle's Invariance Principle, the closed-loop system is asymptotically stable to $(\rho, \alpha, \beta) = (0, 0, 0)$ provided that:
$$k_\rho > 0, \quad k_\alpha > 0, \quad k_\beta < 0, \quad k_\alpha - k_\rho > 0$$

### 2.2 Discrete Implementation in `closed_loop_controller.py`
To prevent chattering near the target pose, the controller executes a 3-stage cascade:
1. **Turn-in-Place Pre-Alignment:** If $|\alpha| > \frac{\pi}{2}$, the robot turns in place with zero linear velocity until its heading roughly faces the target point.
2. **Lyapunov Approach:** The robot simultaneously tracks position and heading with linear saturation $v \le 0.25\text{ m/s}$ and steering saturation $|\omega| \le 1.8\text{ rad/s}$.
3. **Final Heading Alignment:** Once $\rho \le 0.03\text{ m}$ (position tolerance), the controller regulates $\theta \to \theta_g$ via proportional heading feedback until $|\theta - \theta_g| \le 0.05\text{ rad}$.

---

## 3. Electromechanical DC Motor Dynamics

Each wheel of the Khepera IV is driven by a precision DC brushed micromotor with an optical encoder (1944.5 pulses per wheel revolution).

### 3.1 State-Space Equations
The electrical and mechanical dynamics are modeled as:

$$L_a \frac{di_a}{dt} + R_a i_a(t) + K_e \omega_m(t) = u(t)$$
$$J_m \frac{d\omega_m}{dt} + B_m \omega_m(t) + \tau_L = K_t i_a(t)$$

Where:
- $u(t)$: Applied armature voltage (PWM duty cycle).
- $i_a(t)$: Armature current.
- $\omega_m(t)$: Motor rotor angular velocity ($\omega_{wheel} = \omega_m / N_{gear}$).
- $R_a, L_a$: Armature resistance and inductance.
- $K_t, K_e$: Motor torque and back-EMF constants ($K_t = K_e$ in SI units).
- $J_m$: Combined rotor and wheel moment of inertia ($J = J_m + \frac{m_{robot} r^2}{2 N^2}$).
- $B_m$: Viscous friction coefficient.

### 3.2 Second-Order Transfer Function
Neglecting armature inductance ($L_a \approx 0$ due to fast electrical time constant $\tau_e \ll \tau_m$):

$$G(s) = \frac{\Omega_m(s)}{U(s)} = \frac{K_t}{(R_a J_m) s + (R_a B_m + K_t K_e)} = \frac{K_{dc}}{\tau_m s + 1}$$

Where the mechanical time constant is:
$$\tau_m = \frac{R_a J_m}{R_a B_m + K_t K_e}$$

---

## 4. Second-Order Dynamic Step-Response Performance Metrics

During closed-loop setpoint stepping in `dashboard.py` and `telemetry_logger.py`, transient dynamics are logged and computed against classical control criteria:

| Metric | Symbol | Definition & Formula | Target Benchmark |
|---|:---:|---|:---:|
| **Rise Time** | $t_r$ | Time required for response to transition from 10% to 90% of setpoint. | $t_r \le 1.8\text{ s}$ |
| **Settling Time** | $t_s$ | Time after which the error envelope $|y(t) - y_{final}| \le 2\% \cdot y_{final}$. | $t_s \le 3.5\text{ s}$ |
| **Overshoot** | $\sigma\%$ | Maximum peak excursion: $\sigma\% = \frac{y_{max} - y_{final}}{y_{final}} \times 100\%$. | $\sigma\% \le 5.0\%$ |
| **Steady-State Error**| $e_{st}$ | Residual error as $t \to \infty$: $e_{st} = \lim_{t \to \infty} \|x_g - x(t)\|$. | $e_{st} \le 0.015\text{ m}$ |

---

## 5. Frequency Domain Characterization (Bode Analysis)

The open-loop transfer function exhibits a phase margin $\Phi_m \ge 60^\circ$ ensuring well-damped, non-oscillatory trajectory following:
- Gain Crossover Frequency: $\omega_{gc} \approx 8.5\text{ rad/s}$
- Phase Margin: $\Phi_m = 180^\circ + \arg G(j\omega_{gc}) = 64.2^\circ$
- Gain Margin: $K_m \ge 12\text{ dB}$
