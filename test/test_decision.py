"""Unit tests for the overtake decision FSM."""
from overtaking_spline.decision import (
    DecisionConfig,
    DecisionInput,
    OvertakeFSM,
    OvertakeState,
)


def _fsm(**overrides):
    defaults = {"plan_to_execute_count": 2, "plan_failure_count": 2}
    defaults.update(overrides)
    return OvertakeFSM(config=DecisionConfig(**defaults))


def test_starts_in_follow_and_idles_without_opponent():
    fsm = _fsm()
    r = fsm.step(DecisionInput(has_opponent=False))
    assert r.state is OvertakeState.FOLLOW
    assert not r.publish_overtake_path


def test_triggers_to_plan_when_opponent_in_range_and_closing():
    fsm = _fsm(trigger_distance=6.0, min_closing_speed=0.3)
    r = fsm.step(DecisionInput(
        has_opponent=True, ds_to_opponent=4.0, closing_speed=0.5
    ))
    assert r.state is OvertakeState.PLAN_OVERTAKE
    assert r.reason == "trigger"


def test_commits_to_execute_after_streak_of_feasible():
    fsm = _fsm(trigger_distance=6.0, min_closing_speed=0.3,
               plan_to_execute_count=2)
    base = DecisionInput(has_opponent=True, ds_to_opponent=4.0,
                         closing_speed=0.5, plan_feasible=True)
    fsm.step(base)  # FOLLOW -> PLAN_OVERTAKE
    r1 = fsm.step(base)
    assert r1.state is OvertakeState.PLAN_OVERTAKE
    r2 = fsm.step(base)
    assert r2.state is OvertakeState.EXECUTE_OVERTAKE
    assert r2.publish_overtake_path is True


def test_returns_to_follow_after_repeated_plan_failures():
    fsm = _fsm(plan_failure_count=2)
    fsm.step(DecisionInput(has_opponent=True, ds_to_opponent=4.0,
                           closing_speed=0.5))
    bad = DecisionInput(has_opponent=True, ds_to_opponent=4.0,
                        closing_speed=0.5, plan_feasible=False)
    fsm.step(bad)
    r = fsm.step(bad)
    assert r.state is OvertakeState.FOLLOW
    assert r.reason == "plan_failed"


def test_execute_transitions_to_remerge_after_clearing():
    fsm = _fsm(plan_to_execute_count=1, clear_distance=1.0)
    good = DecisionInput(has_opponent=True, ds_to_opponent=4.0,
                         closing_speed=0.5, plan_feasible=True)
    fsm.step(good)
    fsm.step(good)
    assert fsm.state is OvertakeState.EXECUTE_OVERTAKE
    # Opponent is now behind by 2 m.
    r = fsm.step(DecisionInput(
        has_opponent=True, ds_to_opponent=-2.0, closing_speed=0.5,
        plan_feasible=True,
    ))
    assert r.state is OvertakeState.REMERGE


def test_remerge_completes_when_back_on_centerline():
    fsm = _fsm(plan_to_execute_count=1, clear_distance=1.0,
               remerge_lateral=0.1)
    good = DecisionInput(has_opponent=True, ds_to_opponent=4.0,
                         closing_speed=0.5, plan_feasible=True)
    fsm.step(good); fsm.step(good)  # noqa: E702
    fsm.step(DecisionInput(has_opponent=True, ds_to_opponent=-2.0,
                           closing_speed=0.5, plan_feasible=True))
    assert fsm.state is OvertakeState.REMERGE
    r = fsm.step(DecisionInput(has_opponent=False, ego_d=0.05))
    assert r.state is OvertakeState.FOLLOW
    assert r.reason == "remerge_done"
