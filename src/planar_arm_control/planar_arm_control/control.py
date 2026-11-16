"""
control.py  —  joint-space PID tracking + a simple arm dynamics model (no ROS).

ArmDynamics
    The "plant" for the pid control mode. Each joint is a decoupled
    second-order system driven by a motor torque:

        J_i * qdd_i = tau_i + tau_gravity_i(q) - b_i * qd_i

    * J_i: effective inertia seen by joint i (constant approximation of the
      configuration-dependent mass matrix; coupling is ignored),
    * b_i: viscous friction,
    * tau_gravity_i(q): gravity torque from point masses at link midpoints
      (the arm moves in a vertical plane: y is up, the ground is y = 0),
    * motor torque saturates at tau_max, joints stop hard at their limits.
    Integrated with semi-implicit Euler at 1 ms sub-steps.

JointPid
    Per-joint PID with acceleration feed-forward:

        tau = J * qdd_ref + Kp * e + Ki * integral(e) + Kd * (qd_ref - qd)

    Gravity is deliberately NOT fed forward: it is treated as an unmodelled
    load that the integral term must reject, which is the realistic case when
    the payload or link masses are only roughly known. Gains are derived from
    the inertia by pole placement (natural frequency w, damping zeta), so one
    "bandwidth" parameter tunes all joints consistently. Anti-windup: the
    integrator is clamped and frozen while the output saturates.
"""

import numpy as np

G = 9.81


class ArmDynamics:

    def __init__(self, arm, q0, inertia=(60.0, 12.0, 1.2),
                 friction=(5.0, 2.0, 0.5), link_masses=(3.0, 2.0, 1.5),
                 tau_max=(400.0, 150.0, 30.0), substep=1e-3):
        self.arm = arm
        self.J = np.asarray(inertia, dtype=float)
        self.b = np.asarray(friction, dtype=float)
        self.m = np.asarray(link_masses, dtype=float)
        self.tau_max = np.asarray(tau_max, dtype=float)
        self.substep = float(substep)
        self.q = np.asarray(q0, dtype=float).copy()
        self.qd = np.zeros_like(self.q)
        self.tau = np.zeros_like(self.q)      # applied (saturated) torque

    def gravity_torque(self, q):
        """Torque that gravity applies about each joint [N·m]."""
        pts = np.asarray(self.arm.forward_kinematics(q))          # base..tip
        com_x = 0.5 * (pts[:-1, 0] + pts[1:, 0])                  # link midpoints
        tau = np.zeros(len(q))
        for i in range(len(q)):
            # Force (0, -m g) at r: torque_z = r_x * (-m g)
            tau[i] = -G * np.sum(self.m[i:] * (com_x[i:] - pts[i, 0]))
        return tau

    def step(self, tau_cmd, dt):
        self.tau = np.clip(np.asarray(tau_cmd, dtype=float), -self.tau_max, self.tau_max)
        lo, hi = self.arm.joint_limits[:, 0], self.arm.joint_limits[:, 1]
        n = max(1, int(round(dt / self.substep)))
        h = dt / n
        for _ in range(n):
            qdd = (self.tau + self.gravity_torque(self.q) - self.b * self.qd) / self.J
            self.qd = self.qd + qdd * h
            self.q = self.q + self.qd * h
            hit = (self.q < lo) | (self.q > hi)                  # hard stops
            if np.any(hit):
                self.q = np.clip(self.q, lo, hi)
                self.qd[hit] = 0.0


class JointPid:

    def __init__(self, inertia, tau_max, bandwidth=10.0, damping=1.0,
                 integral_ratio=0.3):
        J = np.asarray(inertia, dtype=float)
        w, z = float(bandwidth), float(damping)
        self.J = J
        self.kp = J * w ** 2
        self.kd = J * 2.0 * z * w
        self.ki = self.kp * w * float(integral_ratio)
        self.tau_max = np.asarray(tau_max, dtype=float)
        self.integral = np.zeros_like(J)

    def reset(self):
        self.integral[:] = 0.0

    def update(self, q_ref, qd_ref, qdd_ref, q, qd, dt):
        e = np.asarray(q_ref) - np.asarray(q)
        ed = np.asarray(qd_ref) - np.asarray(qd)
        ff = self.J * (np.zeros_like(e) if qdd_ref is None else np.asarray(qdd_ref))
        tau_no_i = ff + self.kp * e + self.kd * ed
        # Conditional integration: don't wind up a joint that is saturated
        # in the direction its error is pushing.
        tau_try = tau_no_i + self.ki * (self.integral + e * dt)
        saturated = (np.abs(tau_try) >= self.tau_max) & (np.sign(tau_try) == np.sign(e))
        self.integral = np.where(saturated, self.integral, self.integral + e * dt)
        i_max = self.tau_max / np.maximum(self.ki, 1e-9)
        self.integral = np.clip(self.integral, -i_max, i_max)
        tau = tau_no_i + self.ki * self.integral
        return np.clip(tau, -self.tau_max, self.tau_max), e
