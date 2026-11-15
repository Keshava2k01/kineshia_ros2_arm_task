import pytest

from planar_arm_control.sequence import (
    GRIP, MOVE, RELEASE, build_sequence, validate_targets,
)

PICK, PLACE = (4.0, 2.0), (-3.0, 3.0)


def test_full_sequence_with_approach():
    steps = build_sequence(PICK, PLACE, approach_offset=0.5)
    assert [s.phase for s in steps] == [
        "APPROACH_PICK", "DESCEND_PICK", "GRIP", "LIFT",
        "APPROACH_PLACE", "DESCEND_PLACE", "RELEASE", "RETREAT"]
    assert steps[0].xy == (4.0, 2.5)
    assert steps[1].xy == PICK
    assert steps[5].xy == PLACE
    assert steps[2].kind == GRIP and steps[6].kind == RELEASE


def test_minimal_sequence_without_approach():
    steps = build_sequence(PICK, PLACE, approach_offset=0.0)
    assert [(s.phase, s.kind) for s in steps] == [
        ("MOVE_TO_PICK", MOVE), ("GRIP", GRIP),
        ("MOVE_TO_PLACE", MOVE), ("RELEASE", RELEASE)]


def test_approach_skipped_near_reach_boundary():
    steps = build_sequence((0.0, 6.3), PLACE, approach_offset=0.5, reach=6.5)
    phases = [s.phase for s in steps]
    assert "APPROACH_PICK" not in phases and "LIFT" not in phases
    assert phases[0] == "MOVE_TO_PICK"
    assert "APPROACH_PLACE" in phases


def test_scenario_targets_valid():
    assert validate_targets(PICK, PLACE, reach=6.5)[0]


@pytest.mark.parametrize("pick, place, word", [
    ((7.0, 3.0), PLACE, "outside the workspace"),
    (PICK, (7.0, 3.0), "outside the workspace"),
    ((3.0, -1.0), PLACE, "below the ground"),
    ((float("nan"), 1.0), PLACE, "finite"),
])
def test_invalid_targets_rejected(pick, place, word):
    ok, msg = validate_targets(pick, place, reach=6.5)
    assert not ok
    assert word in msg
