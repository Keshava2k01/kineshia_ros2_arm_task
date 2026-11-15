"""
sequence.py  —  the pick-and-place step list (no ROS).

    APPROACH_PICK -> DESCEND_PICK -> GRIP -> LIFT ->
    APPROACH_PLACE -> DESCEND_PLACE -> RELEASE -> RETREAT

This is the brief's move -> pick -> move -> place, plus approach/retreat
points `approach_offset` above each target so the tool comes down onto the
object vertically instead of sweeping into it sideways (what you would do
with a real gripper). With approach_offset = 0 the approach/lift/retreat
moves are dropped and the sequence is just the four required steps. An
approach point that lies outside the reach is skipped.
"""

from dataclasses import dataclass

import numpy as np

MOVE, GRIP, RELEASE = "move", "grip", "release"


@dataclass(frozen=True)
class Step:
    phase: str
    kind: str                # MOVE, GRIP or RELEASE
    xy: tuple = None         # Cartesian target for MOVE steps


def validate_targets(pick, place, reach, min_y=0.0):
    """
    Reject targets a pick-and-place cannot honour. Unlike a plain move we do
    NOT project onto the workspace: the object is at `pick`, so grasping at
    a projected point would grasp nothing.
    """
    for name, xy in (("Pick", pick), ("Place", place)):
        x, y = float(xy[0]), float(xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return False, f"{name} target is not a finite number."
        if y < min_y:
            return False, f"{name} target ({x:.2f}, {y:.2f}) is below the ground (y < {min_y})."
        if np.hypot(x, y) > reach:
            return False, (f"{name} target ({x:.2f}, {y:.2f}) is outside the "
                           f"workspace (reach {reach:.2f}).")
    return True, "ok"


def build_sequence(pick, place, approach_offset=0.5, reach=6.5):
    pick = (float(pick[0]), float(pick[1]))
    place = (float(place[0]), float(place[1]))
    h = max(0.0, float(approach_offset))

    def above(xy):
        if h <= 0.0:
            return None
        p = (xy[0], xy[1] + h)
        return p if np.hypot(*p) <= reach else None

    pick_above, place_above = above(pick), above(place)
    steps = []
    if pick_above:
        steps.append(Step("APPROACH_PICK", MOVE, pick_above))
    steps.append(Step("DESCEND_PICK" if pick_above else "MOVE_TO_PICK", MOVE, pick))
    steps.append(Step("GRIP", GRIP))
    if pick_above:
        steps.append(Step("LIFT", MOVE, pick_above))
    if place_above:
        steps.append(Step("APPROACH_PLACE", MOVE, place_above))
    steps.append(Step("DESCEND_PLACE" if place_above else "MOVE_TO_PLACE", MOVE, place))
    steps.append(Step("RELEASE", RELEASE))
    if place_above:
        steps.append(Step("RETREAT", MOVE, place_above))
    return steps
