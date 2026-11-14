"""
planner.py  —  Cartesian target -> validated joint trajectory (no ROS).

Wraps the provided PlanarArm (treated as a black box) and adds the safety
checks the controller needs before it will execute anything:

  1. Workspace projection is detected and REPORTED (PlanarArm only print()s it).
  2. The IK answer is validated. PlanarArm.inverse_kinematics falls back to an
     unconstrained Jacobian solver that does NOT check joint limits or the
     ground constraint, so its output cannot be trusted blindly.
  3. The whole trajectory is checked, not just its endpoints. Joint limits are
     a box, so a straight joint-space line between two valid poses stays
     inside them, but the ground constraint (no link below y = 0) is not
     convex in joint space and CAN be violated mid-motion. If that happens we
     re-plan through a safe "ready" pose; if that also fails we reject.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from planar_arm_control.trajectory import (
    QuinticTrajectory, SegmentedTrajectory, Trajectory, auto_duration,
)


@dataclass
class PlanResult:
    success: bool
    message: str
    projected: bool = False
    goal_xy: tuple = (float("nan"), float("nan"))
    q_goal: Optional[np.ndarray] = None
    trajectory: Optional[Trajectory] = None
    via_ready_pose: bool = False
    warnings: list = field(default_factory=list)

    @property
    def duration(self):
        return self.trajectory.duration if self.trajectory else 0.0


class MotionPlanner:

    def __init__(self, arm, v_max=1.0, a_max=2.0, t_min=0.5,
                 check_samples=100, ik_tolerance=1e-3,
                 ready_pose=(np.pi / 2, 0.0, 0.0)):
        self.arm = arm
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.t_min = float(t_min)
        self.check_samples = int(check_samples)
        self.ik_tolerance = float(ik_tolerance)
        self.ready_pose = np.asarray(ready_pose, dtype=float)

    # ------------------------------------------------------------------
    # Validity checks
    # ------------------------------------------------------------------
    def pose_valid(self, q):
        return self.arm.within_joint_limits(q) and self.arm.arm_above_base(q)

    def trajectory_valid(self, traj):
        """Sample the trajectory densely and check every pose."""
        _, qs, _, _ = traj.sample_many(self.check_samples)
        return all(self.pose_valid(q) for q in qs)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan(self, q_now, qd_now, target_xy, duration=0.0):
        """Plan from the current state to a Cartesian target."""
        q_now = np.asarray(q_now, dtype=float)
        target = np.asarray(target_xy, dtype=float)

        # 1. Workspace projection (same rule PlanarArm uses internally).
        max_reach = sum(self.arm.link_lengths)
        dist = float(np.linalg.norm(target))
        projected = dist > max_reach
        goal = target * max_reach / dist if projected else target
        goal_xy = (float(goal[0]), float(goal[1]))

        if goal[1] < self.arm.min_y:
            return PlanResult(False, f"Target ({target[0]:.2f}, {target[1]:.2f}) "
                              "is below the ground plane y = 0.",
                              projected=projected, goal_xy=goal_xy)

        # 2. IK + validation of its answer.
        q_goal = np.asarray(
            self.arm.inverse_kinematics(goal, initial_guess=q_now), dtype=float)
        reached = np.asarray(self.arm.end_effector(q_goal), dtype=float)
        err = float(np.linalg.norm(reached - goal))
        if not self.arm.within_joint_limits(q_goal):
            return PlanResult(False, "No IK solution within joint limits.",
                              projected=projected, goal_xy=goal_xy)
        if not self.arm.arm_above_base(q_goal):
            return PlanResult(False, "No IK solution that keeps the arm above "
                              "the ground.", projected=projected, goal_xy=goal_xy)
        if err > self.ik_tolerance:
            return PlanResult(False, f"IK did not converge (error {err:.4f}).",
                              projected=projected, goal_xy=goal_xy)

        result = self.plan_joint(q_now, qd_now, q_goal, duration)
        result.projected = projected
        result.goal_xy = goal_xy
        if result.success:
            if projected:
                result.message = (
                    f"Target ({target[0]:.2f}, {target[1]:.2f}) is outside the "
                    f"workspace (reach {max_reach:.2f}); moving to closest "
                    f"reachable point ({goal_xy[0]:.3f}, {goal_xy[1]:.3f}).")
            else:
                result.message = f"Moving to ({goal_xy[0]:.3f}, {goal_xy[1]:.3f})."
            if result.via_ready_pose:
                result.message += " Routed via ready pose to stay above ground."
            for w in result.warnings:
                result.message += " " + w
        return result

    def plan_joint(self, q_now, qd_now, q_goal, duration=0.0):
        """Plan from the current state to a joint-space goal."""
        q_now = np.asarray(q_now, dtype=float)
        qd_now = np.zeros_like(q_now) if qd_now is None else np.asarray(qd_now, float)
        q_goal = np.asarray(q_goal, dtype=float)

        if not self.pose_valid(q_goal):
            return PlanResult(False, "Goal pose violates limits or ground.")

        warnings = []
        direct = self._segment(q_now, qd_now, q_goal, duration, warnings)
        if self.trajectory_valid(direct):
            return PlanResult(True, "ok", q_goal=q_goal, trajectory=direct,
                              warnings=warnings)

        # Direct path dips below the ground: go via the ready pose.
        seg1 = self._segment(q_now, qd_now, self.ready_pose, 0.0, [])
        seg2 = self._segment(self.ready_pose, None, q_goal, 0.0, [])
        if self.trajectory_valid(seg1) and self.trajectory_valid(seg2):
            return PlanResult(True, "ok", q_goal=q_goal,
                              trajectory=SegmentedTrajectory([seg1, seg2]),
                              via_ready_pose=True)

        return PlanResult(False, "No collision-free path to the goal "
                          "(the arm would pass below the ground).", q_goal=q_goal)

    def _segment(self, q0, qd0, q1, duration, warnings):
        """
        Build one quintic segment within the velocity/acceleration limits.

        The closed-form auto_duration is exact for rest-to-rest moves. When we
        start moving (re-targeting mid-motion) we check the sampled peaks and
        stretch T until the limits hold.
        """
        t_auto = auto_duration(q0, q1, self.v_max, self.a_max, self.t_min)
        T = t_auto
        if duration and duration > 0.0:
            if duration < t_auto:
                warnings.append(
                    f"Requested duration {duration:.2f}s exceeds joint limits; "
                    f"using {t_auto:.2f}s.")
            else:
                T = float(duration)

        traj = QuinticTrajectory(q0, q1, T, qd0=qd0)
        if qd0 is not None and np.any(np.abs(qd0) > 1e-9):
            for _ in range(30):
                _, _, qd, qdd = traj.sample_many(50)
                if (np.max(np.abs(qd)) <= self.v_max * 1.01
                        and np.max(np.abs(qdd)) <= self.a_max * 1.01):
                    break
                T *= 1.1
                traj = QuinticTrajectory(q0, q1, T, qd0=qd0)
        return traj
