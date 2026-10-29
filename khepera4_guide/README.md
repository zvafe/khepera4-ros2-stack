# Khepera IV Control Systems & Robotics Interactive Guide

An interactive, browser-based educational dashboard designed for undergraduate students exploring **Gazebo Harmonic**, **ROS 2 Jazzy**, and **Closed-Loop Control Systems** on the Khepera IV mobile robot.

Complies with the **Bazele Sistemelor Automate (BSA)** academic control systems curriculum / charter and classical 2nd-order control theory.

---

## 🚀 How to Open in Your Browser

### Option 1: Direct File Link (Easiest)
Simply open `khepera4_guide/index.html` in your web browser (Chrome, Firefox, Edge, etc.).
From your terminal inside the repository:
```bash
xdg-open khepera4_guide/index.html
```

---

### Option 2: Local Web Server
You can also launch a lightweight local Python HTTP server:
```bash
python3 -m http.server 8080 --directory khepera4_guide
```
Then navigate to: **`http://localhost:8080`**

---

## 📚 What's Inside This Interactive Guide

1. **The Big Picture Triad:**
   - How Gazebo physics, ROS 2 middleware, and Python controllers talk to each other.
2. **Module 1: Physical Plant & Geometry:**
   - Exact Khepera IV dimensions ($r = 0.021\text{ m}, L = 0.1054\text{ m}, m = 0.540\text{ kg}$).
   - Passive casters, DART physics engine, and ground friction ($\mu = 1.0$).
3. **Module 2: Differential-Drive Kinematics:**
   - Intuitive explanation of how 2 drive wheels steer without a steering wheel.
   - Forward Kinematics: $(\omega_R, \omega_L) \implies (v, \omega)$.
   - Inverse Kinematics: $(v, \omega) \implies (\omega_R, \omega_L)$.
   - **Interactive Kinematics Explorer:** Drag wheel speed sliders to see turning curvature and ICC in real time!
4. **Module 3: The 1-2-3 Closed Loop (Inside the Robot's Brain):**
   - Step 1: Perception (Encoders $\implies$ Odometry pose $(x, y, \theta)$).
   - Step 2: Error Dynamics (Cartesian error $\implies$ Polar coordinates $\rho, \alpha, \beta$).
   - Step 3: Cascade Regulation (Lyapunov feedback policy generating motor speeds).
5. **Module 4: Second-Order Step Response Indicators:**
   - Formulas, meaning, and engineering implications for Rise Time ($t_r$), Settling Time ($t_s$), Maximum Overshoot ($\sigma\%$), and Steady-State Error ($e_{st}$).
   - Why $\zeta = \frac{\sqrt{2}}{2} \approx 0.707$ is the optimal BSA damping ratio.
   - **Interactive Step Response Synthesizer:** Drag $\zeta$ and $\omega_n$ sliders to see the Laplace response curve change live!
6. **Module 5: Fully Playable 2D Browser Arena:**
   - **Mode A:** Drive with `W`, `A`, `S`, `D` keys directly in the browser canvas.
   - **Mode B:** Click anywhere on the arena to set a goal beacon $\implies$ watch the closed-loop controller steer, accelerate, decelerate, and settle with live error metrics and real-time $e(t)$ plotting!
   - **Mode C:** Run Square ($0.7\text{ m} \times 0.7\text{ m}$) and Circular ($R = 0.4\text{ m}$) trajectory routines.
7. **Module 6: Terminal Cheat Sheet:**
   - Exact commands to launch the simulation and Qt desktop dashboard.

---

## 📁 File Structure
```
khepera4_guide/
├── index.html        # Main interactive dashboard UI
├── style.css         # Dark-mode robotics styling and badges
├── simulator.js      # Interactive 2D unicycle & step response physics engine
└── README.md         # Instructions and documentation
```
