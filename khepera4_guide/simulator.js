/**
 * Khepera IV Interactive Simulation & Kinematics Engine
 * Course: Bazele Sistemelor Automate (BSA)
 * 
 * Provides:
 * 1. Differential-Drive Kinematics Simulator (Forward & Inverse)
 * 2. Real-Time Unicycle 2D Canvas Arena with WASD & Closed-Loop Control
 * 3. 2nd-Order Step Response Synthesizer
 */

// Robot Physical Constants (K-Team Khepera IV)
const ROBOT_R = 0.021;    // Wheel radius (m) = 21 mm
const ROBOT_L = 0.1054;   // Wheel separation (m) = 105.4 mm
const ROBOT_DIAM = 0.14;  // Chassis diameter (m) = 140 mm

// Helper: Angle normalization to [-PI, PI]
function normalizeAngle(a) {
    while (a > Math.PI) a -= 2.0 * Math.PI;
    while (a < -Math.PI) a += 2.0 * Math.PI;
    return a;
}

// =============================================================================
// MODULE 1: INTERACTIVE KINEMATICS EXPLORER
// =============================================================================
function initKinematicsExplorer() {
    const sliderWL = document.getElementById('slider-wl');
    const sliderWR = document.getElementById('slider-wr');
    const valWL = document.getElementById('val-wl');
    const valWR = document.getElementById('val-wr');
    const outV = document.getElementById('out-v');
    const outW = document.getElementById('out-w');
    const outWDeg = document.getElementById('out-w-deg');
    const outICC = document.getElementById('out-icc');
    const outBehavior = document.getElementById('out-behavior');

    const canvas = document.getElementById('canvas-kinematics');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let simAngle = 0;

    function updateKinematics() {
        const wL = parseFloat(sliderWL.value);
        const wR = parseFloat(sliderWR.value);

        valWL.textContent = `${wL.toFixed(1)} rad/s`;
        valWR.textContent = `${wR.toFixed(1)} rad/s`;

        // Forward Kinematics formulas
        // v = (r/2) * (wR + wL)
        // w = (r/L) * (wR - wL)
        const v = (ROBOT_R / 2.0) * (wR + wL);
        const w = (ROBOT_R / ROBOT_L) * (wR - wL);

        outV.textContent = `${v.toFixed(3)} m/s`;
        outW.textContent = `${w.toFixed(3)} rad/s`;
        outWDeg.textContent = `(${((w * 180.0) / Math.PI).toFixed(1)} °/s)`;

        // Instantaneous Center of Curvature (ICC) radius R = v / w
        if (Math.abs(w) < 0.001) {
            outICC.textContent = '∞ (Straight Line)';
            if (Math.abs(v) < 0.001) {
                outBehavior.textContent = 'Stationary (At Rest)';
                outBehavior.className = 'text-gray-400 font-semibold';
            } else if (v > 0) {
                outBehavior.textContent = 'Pure Forward Translation';
                outBehavior.className = 'text-emerald-400 font-semibold';
            } else {
                outBehavior.textContent = 'Pure Reverse Translation';
                outBehavior.className = 'text-amber-400 font-semibold';
            }
        } else {
            const R = v / w;
            outICC.textContent = `${(R * 100).toFixed(1)} cm from center`;
            if (Math.abs(v) < 0.001) {
                outBehavior.textContent = w > 0 ? 'Pure In-Place Turn Left (Spin)' : 'Pure In-Place Turn Right (Spin)';
                outBehavior.className = 'text-cyan-400 font-semibold';
            } else {
                outBehavior.textContent = w > 0 ? 'Curved Motion Arc (Turning Left)' : 'Curved Motion Arc (Turning Right)';
                outBehavior.className = 'text-purple-400 font-semibold';
            }
        }
    }

    function renderKinematicsCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const wL = parseFloat(sliderWL.value);
        const wR = parseFloat(sliderWR.value);
        const v = (ROBOT_R / 2.0) * (wR + wL);
        const w = (ROBOT_R / ROBOT_L) * (wR - wL);

        simAngle += w * 0.03;

        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const scale = 350; // pixels per meter

        // Draw grid
        ctx.strokeStyle = '#232b38';
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 30) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 30) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }

        // Draw ICC radius circle if turning
        if (Math.abs(w) > 0.05 && Math.abs(v) > 0.01) {
            const R = v / w;
            const rPx = Math.abs(R) * scale;
            if (rPx < 500) {
                const iccX = cx - Math.sin(simAngle) * (R * scale);
                const iccY = cy + Math.cos(simAngle) * (R * scale);

                ctx.strokeStyle = 'rgba(168, 85, 247, 0.35)';
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.arc(iccX, iccY, rPx, 0, 2 * Math.PI);
                ctx.stroke();
                ctx.setLineDash([]);

                // ICC dot
                ctx.fillStyle = '#c084fc';
                ctx.beginPath();
                ctx.arc(iccX, iccY, 4, 0, 2 * Math.PI);
                ctx.fill();
            }
        }

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(-simAngle);

        // Robot Chassis (140mm diameter = 0.14m)
        const robotRadPx = (ROBOT_DIAM / 2) * scale;
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = 'rgba(0, 230, 118, 0.4)';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(0, 0, robotRadPx, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Front blue bumper
        ctx.strokeStyle = '#1d4ed8';
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(0, 0, robotRadPx, -Math.PI / 3, Math.PI / 3);
        ctx.stroke();

        // Left Wheel (Y = +L/2)
        const wheelW = 14;
        const wheelL = (ROBOT_R * 2) * scale;
        const trackPx = (ROBOT_L / 2) * scale;

        ctx.fillStyle = '#1e293b';
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.5;
        // Left wheel (top in canvas coordinate when facing right)
        ctx.fillRect(-wheelL / 2, -trackPx - wheelW / 2, wheelL, wheelW);
        ctx.strokeRect(-wheelL / 2, -trackPx - wheelW / 2, wheelL, wheelW);

        // Right Wheel (Y = -L/2)
        ctx.fillRect(-wheelL / 2, trackPx - wheelW / 2, wheelL, wheelW);
        ctx.strokeRect(-wheelL / 2, trackPx - wheelW / 2, wheelL, wheelW);

        // Heading arrow (forward +X)
        ctx.strokeStyle = '#10b981';
        ctx.fillStyle = '#10b981';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(robotRadPx + 16, 0);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(robotRadPx + 22, 0);
        ctx.lineTo(robotRadPx + 12, -6);
        ctx.lineTo(robotRadPx + 12, 6);
        ctx.closePath();
        ctx.fill();

        ctx.restore();

        requestAnimationFrame(renderKinematicsCanvas);
    }

    sliderWL.addEventListener('input', updateKinematics);
    sliderWR.addEventListener('input', updateKinematics);

    updateKinematics();
    renderKinematicsCanvas();
}

// =============================================================================
// MODULE 2: INTERACTIVE 2D CLOSED-LOOP & WASD ARENA SIMULATOR
// =============================================================================
function initArenaSimulator() {
    const canvas = document.getElementById('canvas-arena');
    const errCanvas = document.getElementById('canvas-error-curve');
    if (!canvas || !errCanvas) return;

    const ctx = canvas.getContext('2d');
    const errCtx = errCanvas.getContext('2d');

    // Robot state (Meters & Radians)
    let robot = {
        x: 0.0,
        y: 0.0,
        yaw: 0.0,
        v: 0.0,
        w: 0.0
    };

    let target = {
        x: 0.6,
        y: 0.0,
        yaw: 0.0
    };

    let mode = 'B'; // 'A' (WASD), 'B' (Step/Goal), 'C' (Trajectory)
    let isNavigating = false;
    let waypoints = [];
    let waypointIdx = 0;

    // Control parameters (Lyapunov)
    const K_rho = 0.9;
    const K_alpha = 2.2;
    const K_beta = -0.4;
    const K_yaw = 1.6;
    const V_MAX = 0.22; // m/s
    const W_MAX = 1.8;  // rad/s
    const POS_TOL = 0.02; // 2cm
    const YAW_TOL = 0.05;

    // Key states for WASD
    const keys = {};

    // Telemetry & metrics history
    let historyTrail = [];
    let errorHistory = [];
    let startTime = null;
    let initialDist = 0.0;

    // Scale: arena is -0.5m to 1.8m in X, -1.0m to 1.0m in Y
    const ppm = 180; // pixels per meter
    const originX = 100;
    const originY = canvas.height / 2;

    function toCanvasX(mX) { return originX + mX * ppm; }
    function toCanvasY(mY) { return originY - mY * ppm; }
    function fromCanvasX(pX) { return (pX - originX) / ppm; }
    function fromCanvasY(pY) { return (originY - pY) / ppm; }

    // UI elements
    const liveRho = document.getElementById('live-rho');
    const liveAlpha = document.getElementById('live-alpha');
    const liveBeta = document.getElementById('live-beta');
    const liveX = document.getElementById('live-x');
    const liveY = document.getElementById('live-y');
    const liveYaw = document.getElementById('live-yaw');
    const liveV = document.getElementById('live-v');
    const liveW = document.getElementById('live-w');

    const liveTr = document.getElementById('live-tr');
    const liveTs = document.getElementById('live-ts');
    const liveSigma = document.getElementById('live-sigma');
    const liveEst = document.getElementById('live-est');

    // Click canvas to set target in Mode B
    canvas.addEventListener('click', (e) => {
        if (mode !== 'B') return;
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        target.x = fromCanvasX(mouseX);
        target.y = fromCanvasY(mouseY);
        target.yaw = 0.0;

        startNavigation();
    });

    // Keyboard events for WASD
    window.addEventListener('keydown', (e) => {
        const k = e.key.toLowerCase();
        if (['w', 'a', 's', 'd', ' '].includes(k)) {
            keys[k] = true;
            highlightKey(k, true);
        }
    });

    window.addEventListener('keyup', (e) => {
        const k = e.key.toLowerCase();
        if (['w', 'a', 's', 'd', ' '].includes(k)) {
            keys[k] = false;
            highlightKey(k, false);
        }
    });

    function highlightKey(k, active) {
        const el = document.getElementById(`sim-key-${k}`);
        if (el) {
            if (active) {
                el.className = 'w-12 h-12 flex items-center justify-center font-bold rounded-lg border-2 border-emerald-400 bg-emerald-950 text-emerald-300 shadow-md shadow-emerald-500/30';
            } else {
                el.className = 'w-12 h-12 flex items-center justify-center font-bold rounded-lg border border-slate-700 bg-slate-800 text-slate-300';
            }
        }
    }

    function setMode(newMode) {
        mode = newMode;
        stopNavigation();

        // Update button highlights
        document.getElementById('btn-mode-a').className = mode === 'A' ? 'mode-btn-active' : 'mode-btn-idle';
        document.getElementById('btn-mode-b').className = mode === 'B' ? 'mode-btn-active' : 'mode-btn-idle';
        document.getElementById('btn-mode-c').className = mode === 'C' ? 'mode-btn-active' : 'mode-btn-idle';

        document.getElementById('panel-sim-a').style.display = mode === 'A' ? 'block' : 'none';
        document.getElementById('panel-sim-b').style.display = mode === 'B' ? 'block' : 'none';
        document.getElementById('panel-sim-c').style.display = mode === 'C' ? 'block' : 'none';

        if (mode === 'B') {
            target = { x: 0.6, y: 0.0, yaw: 0.0 };
        } else if (mode === 'C') {
            setupSquareTrajectory();
        }
    }

    function startNavigation() {
        if (mode === 'B') {
            waypoints = [{ x: target.x, y: target.y, yaw: target.yaw }];
        }
        waypointIdx = 0;
        isNavigating = true;
        startTime = performance.now();
        initialDist = Math.hypot(target.x - robot.x, target.y - robot.y);
        errorHistory = [];
    }

    function stopNavigation() {
        isNavigating = false;
        robot.v = 0.0;
        robot.w = 0.0;
    }

    function resetRobot() {
        robot = { x: 0.0, y: 0.0, yaw: 0.0, v: 0.0, w: 0.0 };
        historyTrail = [];
        errorHistory = [];
        stopNavigation();
        if (mode === 'B') target = { x: 0.6, y: 0.0, yaw: 0.0 };
    }

    function setupSquareTrajectory() {
        const cx = robot.x;
        const cy = robot.y;
        waypoints = [
            { x: cx + 0.7, y: cy + 0.0, yaw: 0.0 },
            { x: cx + 0.7, y: cy + 0.7, yaw: Math.PI / 2 },
            { x: cx + 0.0, y: cy + 0.7, yaw: Math.PI },
            { x: cx + 0.0, y: cy + 0.0, yaw: -Math.PI / 2 }
        ];
        waypointIdx = 0;
        target = waypoints[0];
    }

    function setupCircleTrajectory() {
        const R = 0.4;
        const cx = robot.x;
        const cy = robot.y + R;
        waypoints = [];
        const N = 16;
        for (let i = 0; i < N; i++) {
            const theta = -Math.PI / 2 + (2 * Math.PI * (i + 1)) / N;
            waypoints.push({
                x: cx + R * Math.cos(theta),
                y: cy + R * Math.sin(theta),
                yaw: normalizeAngle(theta + Math.PI / 2)
            });
        }
        waypointIdx = 0;
        target = waypoints[0];
    }

    // Step physics & control loop (dt = 0.02s)
    function updatePhysics(dt) {
        // Mode A: WASD manual drive
        if (mode === 'A') {
            let vCmd = 0.0;
            let wCmd = 0.0;

            if (keys['w']) vCmd += 0.18;
            if (keys['s']) vCmd -= 0.18;
            if (keys['a']) wCmd += 1.4;
            if (keys['d']) wCmd -= 1.4;
            if (keys[' ']) { vCmd = 0; wCmd = 0; }

            robot.v = vCmd;
            robot.w = wCmd;
        }

        // Mode B & C: Closed-Loop Cascade Controller
        else if (isNavigating && waypoints.length > 0 && waypointIdx < waypoints.length) {
            const curTarget = waypoints[waypointIdx];
            target = curTarget;

            const dx = curTarget.x - robot.x;
            const dy = curTarget.y - robot.y;
            const rho = Math.hypot(dx, dy);
            const alpha = normalizeAngle(Math.atan2(dy, dx) - robot.yaw);
            const beta = normalizeAngle(-robot.yaw - alpha + curTarget.yaw);

            // Record dynamic error
            const elapsed = (performance.now() - startTime) / 1000.0;
            errorHistory.push({ t: elapsed, rho: rho });

            // Update live UI
            liveRho.textContent = `${rho.toFixed(3)} m`;
            liveAlpha.textContent = `${((alpha * 180) / Math.PI).toFixed(1)} °`;
            liveBeta.textContent = `${((beta * 180) / Math.PI).toFixed(1)} °`;

            // Control Law
            if (rho > POS_TOL) {
                if (Math.abs(alpha) > Math.PI / 2.0) {
                    robot.v = 0.0;
                    robot.w = Math.max(-W_MAX, Math.min(W_MAX, K_alpha * alpha));
                } else {
                    robot.v = Math.max(-V_MAX, Math.min(V_MAX, K_rho * rho));
                    robot.w = Math.max(-W_MAX, Math.min(W_MAX, K_alpha * alpha + K_beta * beta));
                }
            } else {
                const yawErr = normalizeAngle(curTarget.yaw - robot.yaw);
                if (Math.abs(yawErr) > YAW_TOL) {
                    robot.v = 0.0;
                    robot.w = Math.max(-W_MAX, Math.min(W_MAX, K_yaw * yawErr));
                } else {
                    // Waypoint reached!
                    waypointIdx++;
                    if (waypointIdx < waypoints.length) {
                        target = waypoints[waypointIdx];
                    } else {
                        // Trajectory done!
                        stopNavigation();
                        computeSecondOrderMetrics();
                    }
                }
            }
        } else {
            robot.v = 0.0;
            robot.w = 0.0;
        }

        // Integrate unicycle equations of motion
        // x_dot = v * cos(yaw)
        // y_dot = v * sin(yaw)
        // yaw_dot = w
        robot.x += robot.v * Math.cos(robot.yaw) * dt;
        robot.y += robot.v * Math.sin(robot.yaw) * dt;
        robot.yaw = normalizeAngle(robot.yaw + robot.w * dt);

        // Update history trail
        historyTrail.push({ x: robot.x, y: robot.y });
        if (historyTrail.length > 500) historyTrail.shift();

        // Update state readout
        liveX.textContent = `${robot.x.toFixed(3)} m`;
        liveY.textContent = `${robot.y.toFixed(3)} m`;
        liveYaw.textContent = `${((robot.yaw * 180) / Math.PI).toFixed(1)} °`;
        liveV.textContent = `${robot.v.toFixed(2)} m/s`;
        liveW.textContent = `${robot.w.toFixed(2)} rad/s`;
    }

    function computeSecondOrderMetrics() {
        if (errorHistory.length < 5 || initialDist <= 0.0) return;

        const targetDisplacement = initialDist;
        const disps = errorHistory.map(e => targetDisplacement - e.rho);
        const times = errorHistory.map(e => e.t);

        // Overshoot
        const maxDisp = Math.max(...disps);
        const overshoot = Math.max(0.0, ((maxDisp - targetDisplacement) / targetDisplacement) * 100.0);

        // Steady state error
        const est = Math.abs(disps[disps.length - 1] - targetDisplacement);

        // Rise time (10% to 90%)
        let t10 = null;
        let t90 = null;
        for (let i = 0; i < disps.length; i++) {
            if (t10 === null && disps[i] >= 0.10 * targetDisplacement) t10 = times[i];
            if (t90 === null && disps[i] >= 0.90 * targetDisplacement) t90 = times[i];
        }
        const tr = (t10 !== null && t90 !== null) ? (t90 - t10) : null;

        // Settling time (2% band)
        const band = 0.02 * targetDisplacement;
        let lastOutsideIdx = 0;
        for (let i = 0; i < disps.length; i++) {
            if (Math.abs(disps[i] - targetDisplacement) > band) {
                lastOutsideIdx = i;
            }
        }
        const ts = times[lastOutsideIdx] || times[0];

        liveSigma.textContent = `${overshoot.toFixed(2)} %`;
        liveEst.textContent = `${est.toFixed(4)} m`;
        if (tr !== null) liveTr.textContent = `${tr.toFixed(2)} s`;
        liveTs.textContent = `${ts.toFixed(2)} s`;
    }

    // Render Canvas Loop
    function renderArena() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Grid
        ctx.strokeStyle = '#1e2430';
        ctx.lineWidth = 1;
        for (let x = 0; x < canvas.width; x += 40) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
        }
        for (let y = 0; y < canvas.height; y += 40) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
        }

        // Draw Origin Axes
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(0, originY); ctx.lineTo(canvas.width, originY); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(originX, 0); ctx.lineTo(originX, canvas.height); ctx.stroke();

        // Planned path for Mode C
        if (mode === 'C' && waypoints.length > 1) {
            ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(toCanvasX(waypoints[0].x), toCanvasY(waypoints[0].y));
            for (let i = 1; i < waypoints.length; i++) {
                ctx.lineTo(toCanvasX(waypoints[i].x), toCanvasY(waypoints[i].y));
            }
            if (waypoints.length === 4) ctx.closePath();
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Draw Trajectory Trail
        if (historyTrail.length > 1) {
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(toCanvasX(historyTrail[0].x), toCanvasY(historyTrail[0].y));
            for (let i = 1; i < historyTrail.length; i++) {
                ctx.lineTo(toCanvasX(historyTrail[i].x), toCanvasY(historyTrail[i].y));
            }
            ctx.stroke();
        }

        // Draw Goal Beacon Target
        if (mode !== 'A') {
            const bx = toCanvasX(target.x);
            const by = toCanvasY(target.y);

            // Pulsing target circle
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 2;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.arc(bx, by, (ROBOT_DIAM / 2) * ppm, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.setLineDash([]);

            // Gold Star Beacon
            ctx.fillStyle = '#fbbf24';
            ctx.shadowColor = '#f59e0b';
            ctx.shadowBlur = 10;
            ctx.beginPath();
            ctx.arc(bx, by, 7, 0, 2 * Math.PI);
            ctx.fill();
            ctx.shadowBlur = 0;

            // Target coordinates label
            ctx.fillStyle = '#94a3b8';
            ctx.font = '10px monospace';
            ctx.fillText(`Target (${target.x.toFixed(2)}, ${target.y.toFixed(2)})`, bx + 12, by - 6);
        }

        // Draw Robot Body
        const rx = toCanvasX(robot.x);
        const ry = toCanvasY(robot.y);
        const rRad = (ROBOT_DIAM / 2) * ppm;

        ctx.save();
        ctx.translate(rx, ry);
        ctx.rotate(-robot.yaw);

        // Body
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = 'rgba(16, 185, 129, 0.5)';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(0, 0, rRad, 0, 2 * Math.PI);
        ctx.fill();
        ctx.shadowBlur = 0;

        // Bumper
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(0, 0, rRad, -Math.PI / 3, Math.PI / 3);
        ctx.stroke();

        // Heading arrow
        ctx.strokeStyle = '#10b981';
        ctx.fillStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(rRad + 10, 0);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(rRad + 14, 0);
        ctx.lineTo(rRad + 6, -4);
        ctx.lineTo(rRad + 6, 4);
        ctx.closePath();
        ctx.fill();

        ctx.restore();

        // Render Error Curve Canvas
        renderErrorCanvas();

        requestAnimationFrame(() => {
            updatePhysics(0.02);
            renderArena();
        });
    }

    function renderErrorCanvas() {
        errCtx.clearRect(0, 0, errCanvas.width, errCanvas.height);

        errCtx.strokeStyle = '#1e2430';
        errCtx.lineWidth = 1;
        for (let y = 0; y < errCanvas.height; y += 30) {
            errCtx.beginPath(); errCtx.moveTo(0, y); errCtx.lineTo(errCanvas.width, y); errCtx.stroke();
        }

        if (errorHistory.length < 2) {
            errCtx.fillStyle = '#64748b';
            errCtx.font = '11px sans-serif';
            errCtx.textAlign = 'center';
            errCtx.fillText('Click on Arena or Press "Step Setpoint" to plot e(t)', errCanvas.width / 2, errCanvas.height / 2);
            return;
        }

        // Plot error curve
        const maxT = Math.max(8.0, errorHistory[errorHistory.length - 1].t);
        const maxRho = Math.max(0.6, initialDist || 0.6);

        const toPlotX = (t) => 40 + (t / maxT) * (errCanvas.width - 60);
        const toPlotY = (r) => (errCanvas.height - 30) - (r / maxRho) * (errCanvas.height - 50);

        // 2cm Tolerance line
        const tolY = toPlotY(POS_TOL);
        errCtx.strokeStyle = '#10b981';
        errCtx.setLineDash([4, 4]);
        errCtx.beginPath();
        errCtx.moveTo(40, tolY);
        errCtx.lineTo(errCanvas.width - 20, tolY);
        errCtx.stroke();
        errCtx.setLineDash([]);
        errCtx.fillStyle = '#10b981';
        errCtx.font = '9px monospace';
        errCtx.textAlign = 'right';
        errCtx.fillText('2cm tol', errCanvas.width - 22, tolY - 4);

        // Error line
        errCtx.strokeStyle = '#ef4444';
        errCtx.lineWidth = 2;
        errCtx.beginPath();
        errCtx.moveTo(toPlotX(errorHistory[0].t), toPlotY(errorHistory[0].rho));
        for (let i = 1; i < errorHistory.length; i++) {
            errCtx.lineTo(toPlotX(errorHistory[i].t), toPlotY(errorHistory[i].rho));
        }
        errCtx.stroke();
    }

    // Connect mode buttons
    document.getElementById('btn-mode-a').addEventListener('click', () => setMode('A'));
    document.getElementById('btn-mode-b').addEventListener('click', () => setMode('B'));
    document.getElementById('btn-mode-c').addEventListener('click', () => setMode('C'));

    document.getElementById('btn-step-05').addEventListener('click', () => {
        target = { x: robot.x + 0.5, y: robot.y, yaw: 0.0 };
        startNavigation();
    });
    document.getElementById('btn-step-10').addEventListener('click', () => {
        target = { x: robot.x + 1.0, y: robot.y, yaw: 0.0 };
        startNavigation();
    });
    document.getElementById('btn-reset').addEventListener('click', resetRobot);

    document.getElementById('btn-traj-square').addEventListener('click', () => {
        setupSquareTrajectory();
        startNavigation();
    });
    document.getElementById('btn-traj-circle').addEventListener('click', () => {
        setupCircleTrajectory();
        startNavigation();
    });

    setMode('B');
    renderArena();
}

// =============================================================================
// MODULE 3: SECOND-ORDER STEP RESPONSE INTERACTIVE SYNTHESIZER
// =============================================================================
function initStepResponseSynthesizer() {
    const sliderZeta = document.getElementById('slider-zeta');
    const sliderWn = document.getElementById('slider-wn');
    const lblZeta = document.getElementById('lbl-zeta');
    const lblWn = document.getElementById('lbl-wn');
    const lblOvershootCalc = document.getElementById('lbl-overshoot-calc');
    const lblTsCalc = document.getElementById('lbl-ts-calc');
    const lblTrCalc = document.getElementById('lbl-tr-calc');
    const lblRegime = document.getElementById('lbl-regime');

    const canvas = document.getElementById('canvas-step-response');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function updateStep() {
        const zeta = parseFloat(sliderZeta.value);
        const wn = parseFloat(sliderWn.value);

        lblZeta.textContent = zeta.toFixed(2);
        lblWn.textContent = `${wn.toFixed(1)} rad/s`;

        // Theoretical indicators
        // Overshoot sigma = exp(-pi*zeta / sqrt(1 - zeta^2)) * 100% (for zeta < 1)
        if (zeta < 1.0) {
            const sigma = Math.exp((-Math.PI * zeta) / Math.sqrt(1 - zeta * zeta)) * 100.0;
            lblOvershootCalc.textContent = `${sigma.toFixed(2)} %`;
            lblRegime.textContent = 'Subamortizat (Underdamped, Oscillatory)';
            lblRegime.className = 'text-amber-400 font-semibold';
        } else if (Math.abs(zeta - 1.0) < 0.01) {
            lblOvershootCalc.textContent = '0.00 %';
            lblRegime.textContent = 'Amortizat Critic (Critically Damped, Fastest Smooth)';
            lblRegime.className = 'text-emerald-400 font-semibold';
        } else {
            lblOvershootCalc.textContent = '0.00 %';
            lblRegime.textContent = 'Supraamortizat (Overdamped, Sluggish)';
            lblRegime.className = 'text-blue-400 font-semibold';
        }

        // Settling time ts approx 4 / (zeta * wn)
        const ts = 4.0 / (zeta * wn);
        lblTsCalc.textContent = `${ts.toFixed(2)} s`;

        // Rise time tr approx (pi - arccos(zeta)) / (wn * sqrt(1 - zeta^2))
        if (zeta < 1.0) {
            const tr = (Math.PI - Math.acos(zeta)) / (wn * Math.sqrt(1 - zeta * zeta));
            lblTrCalc.textContent = `${tr.toFixed(2)} s`;
        } else {
            lblTrCalc.textContent = `~${(2.2 / (zeta * wn)).toFixed(2)} s`;
        }

        drawStepCurve(zeta, wn);
    }

    function drawStepCurve(zeta, wn) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const padLeft = 45;
        const padRight = 20;
        const padTop = 30;
        const padBottom = 30;
        const plotW = canvas.width - padLeft - padRight;
        const plotH = canvas.height - padTop - padBottom;

        // Grid
        ctx.strokeStyle = '#1e2430';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 6; i++) {
            const y = padTop + (i / 6) * plotH;
            ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(canvas.width - padRight, y); ctx.stroke();
        }

        // 1.0 Setpoint Line
        const ySetpoint = padTop + plotH * 0.45;
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(padLeft, ySetpoint); ctx.lineTo(canvas.width - padRight, ySetpoint); ctx.stroke();

        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        ctx.fillText('1.00 Ref', padLeft - 6, ySetpoint + 3);

        // +/- 2% Settling band
        const bandH = plotH * 0.45 * 0.02;
        ctx.fillStyle = 'rgba(16, 185, 129, 0.1)';
        ctx.fillRect(padLeft, ySetpoint - bandH, plotW, bandH * 2);

        // Plot y(t) = 1 - (e^(-zeta*wn*t)/sqrt(1-zeta^2)) * sin(wd*t + phi)
        const tMax = 6.0;
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2.5;
        ctx.beginPath();

        const N = 300;
        for (let i = 0; i <= N; i++) {
            const t = (i / N) * tMax;
            let yVal = 0;

            if (zeta < 1.0) {
                const wd = wn * Math.sqrt(1 - zeta * zeta);
                const phi = Math.atan2(Math.sqrt(1 - zeta * zeta), zeta);
                yVal = 1.0 - (Math.exp(-zeta * wn * t) / Math.sqrt(1 - zeta * zeta)) * Math.sin(wd * t + phi);
            } else if (Math.abs(zeta - 1.0) < 0.01) {
                yVal = 1.0 - (1.0 + wn * t) * Math.exp(-wn * t);
            } else {
                const s1 = -zeta * wn + wn * Math.sqrt(zeta * zeta - 1);
                const s2 = -zeta * wn - wn * Math.sqrt(zeta * zeta - 1);
                yVal = 1.0 - (s2 * Math.exp(s1 * t) - s1 * Math.exp(s2 * t)) / (s2 - s1);
            }

            const canvasX = padLeft + (t / tMax) * plotW;
            const canvasY = ySetpoint - (yVal - 1.0) * (plotH * 0.45);

            if (i === 0) ctx.moveTo(canvasX, canvasY);
            else ctx.lineTo(canvasX, canvasY);
        }
        ctx.stroke();

        // X Axis labels
        ctx.fillStyle = '#64748b';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        for (let s = 0; s <= 6; s++) {
            const x = padLeft + (s / 6) * plotW;
            ctx.fillText(`${s}s`, x, canvas.height - 12);
        }
    }

    sliderZeta.addEventListener('input', updateStep);
    sliderWn.addEventListener('input', updateStep);

    updateStep();
}

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', () => {
    initKinematicsExplorer();
    initArenaSimulator();
    initStepResponseSynthesizer();
});
