"""Overtake decision FSM with hysteresis. No ROS dependency."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OvertakeState(Enum):
    FOLLOW = "follow"
    PLAN_OVERTAKE = "plan_overtake"
    EXECUTE_OVERTAKE = "execute_overtake"
    REMERGE = "remerge"


@dataclass
class DecisionConfig:
    trigger_distance: float = 6.0      # opponent must be within this arc-length [m]
    min_closing_speed: float = 0.3     # ego - opp longitudinal speed [m/s]
    clear_distance: float = 1.5        # arc-length past opponent considered cleared [m]
    remerge_lateral: float = 0.10      # |d| under which we consider remerge done [m]
    plan_to_execute_count: int = 2     # consecutive feasible plans needed to commit
    plan_failure_count: int = 3        # consecutive failures before bailing back to FOLLOW
    abort_distance: float = 1.0        # if opponent gets this close laterally we abort


@dataclass
class DecisionInput:
    has_opponent: bool
    ds_to_opponent: float = 0.0   # signed forward arc-length, ego -> opp
    opp_d: float = 0.0
    ego_d: float = 0.0
    closing_speed: float = 0.0    # ego_v_s - opp_v_s
    plan_feasible: bool = False


@dataclass
class DecisionResult:
    state: OvertakeState
    publish_overtake_path: bool
    reason: str


@dataclass
class _Counters:
    feasible_streak: int = 0
    infeasible_streak: int = 0


@dataclass
class OvertakeFSM:
    config: DecisionConfig = field(default_factory=DecisionConfig)
    state: OvertakeState = OvertakeState.FOLLOW
    _counters: _Counters = field(default_factory=_Counters)

    def step(self, inp: DecisionInput) -> DecisionResult:
        cfg = self.config

        # Universal early-out: opponent disappeared and we're not mid-overtake.
        if not inp.has_opponent and self.state in (
            OvertakeState.FOLLOW, OvertakeState.PLAN_OVERTAKE
        ):
            self._reset()
            self.state = OvertakeState.FOLLOW
            return DecisionResult(self.state, False, "no_opponent")

        if self.state is OvertakeState.FOLLOW:
            if (inp.has_opponent
                and 0.0 < inp.ds_to_opponent <= cfg.trigger_distance
                and inp.closing_speed >= cfg.min_closing_speed):
                self.state = OvertakeState.PLAN_OVERTAKE
                self._reset()
                return DecisionResult(self.state, False, "trigger")
            return DecisionResult(self.state, False, "follow")

        if self.state is OvertakeState.PLAN_OVERTAKE:
            if inp.plan_feasible:
                self._counters.feasible_streak += 1
                self._counters.infeasible_streak = 0
                if self._counters.feasible_streak >= cfg.plan_to_execute_count:
                    self.state = OvertakeState.EXECUTE_OVERTAKE
                    return DecisionResult(self.state, True, "commit")
                return DecisionResult(self.state, True, "planning")
            self._counters.infeasible_streak += 1
            self._counters.feasible_streak = 0
            if self._counters.infeasible_streak >= cfg.plan_failure_count:
                self.state = OvertakeState.FOLLOW
                self._reset()
                return DecisionResult(self.state, False, "plan_failed")
            return DecisionResult(self.state, False, "planning_retry")

        if self.state is OvertakeState.EXECUTE_OVERTAKE:
            # Cleared the opponent (passed by clear_distance), move to remerge.
            if not inp.has_opponent or inp.ds_to_opponent < -cfg.clear_distance:
                self.state = OvertakeState.REMERGE
                return DecisionResult(self.state, True, "cleared")
            # Lost feasibility while committed -- keep publishing the last spline
            # for a few cycles but if it stays infeasible, abort to remerge.
            if not inp.plan_feasible:
                self._counters.infeasible_streak += 1
                if self._counters.infeasible_streak >= cfg.plan_failure_count:
                    self.state = OvertakeState.REMERGE
                    return DecisionResult(self.state, True, "abort_to_remerge")
            else:
                self._counters.infeasible_streak = 0
            return DecisionResult(self.state, True, "executing")

        # REMERGE
        if abs(inp.ego_d) <= cfg.remerge_lateral:
            self.state = OvertakeState.FOLLOW
            self._reset()
            return DecisionResult(self.state, False, "remerge_done")
        return DecisionResult(self.state, True, "remerging")

    def _reset(self) -> None:
        self._counters = _Counters()
