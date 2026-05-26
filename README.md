# overtaking_spline

Frenet-frame quintic-spline overtaking planner for the GIU Berlin F1TENTH stack.
Closed-form, single-digit-millisecond per cycle. Designed to plug into the
existing `control_gateway` route the same way `overtaking` (DWA) does.

## What it does

1. Builds a KD-tree-backed Frenet frame from the reference raceline (`/pp_path`).
2. Whenever an opponent is in front and we are closing, evaluates 5–9
   quintic lateral candidates `d(s)` over a `lookahead`-meter window.
3. Scores candidates on clearance to the opponent, max curvature feasibility
   at the local reference speed (`κ ≤ a_lat_max / v²`), and a configurable
   racing-line-side bias.
4. Decision FSM (`FOLLOW → PLAN_OVERTAKE → EXECUTE_OVERTAKE → REMERGE → FOLLOW`)
   with hysteresis on commit and on failure.
5. Publishes the chosen path as `nav_msgs/Path` and a follower drive command
   as `ackermann_msgs/AckermannDriveStamped`.

## Topics

| Direction | Topic                              | Type                              | Notes |
|---|---|---|---|
| sub | `/pp_path`                         | `nav_msgs/Path`                    | velocity in `pose.orientation.w` (convention from `csv_path_pub`) |
| sub | `/car_state/odom`                  | `nav_msgs/Odometry`                | sensor QoS |
| sub | `/object_centers`                  | `visualization_msgs/MarkerArray`   | publisher prepends a `DELETEALL`; we skip `[0]` |
| sub | `/control_selector`                | `std_msgs/String`                  | from FSM; node only emits drive when `data == "overtaking_spline"` (configurable) |
| pub | `/overtaking_spline/path`          | `nav_msgs/Path`                    | overtake path (with per-pose ref velocity in `orientation.w`) |
| pub | `/overtaking_spline/drive`         | `ackermann_msgs/AckermannDriveStamped` | follower output, routed by `control_gateway` |
| pub | `/overtaking_spline/diagnostics`   | `std_msgs/Float64MultiArray`       | `[plan_ms, total_ms, ego_speed, state_id, side]` |
| pub | `/overtaking_spline/candidates`    | `visualization_msgs/MarkerArray`   | debug viz (toggle with `publish_candidates`) |

`state_id` in diagnostics: `0=FOLLOW, 1=PLAN_OVERTAKE, 2=EXECUTE_OVERTAKE, 3=REMERGE`.

## Parameters

See [`config/overtaking_spline.yaml`](config/overtaking_spline.yaml). Highlights:

- **`planner.car_width`, `planner.safety_margin`, `planner.track_width`** — the
  usable half-width for the vehicle centerline is
  `track_width/2 − car_width/2 − safety_margin`. If non-positive, the
  planner short-circuits with reason `no_drivable_width`.
- **`planner.lookahead`, `planner.rejoin_distance`** — longitudinal envelope
  of the maneuver. The total spline length is
  `max(lookahead, ds_to_opponent + rejoin_distance)`.
- **`planner.max_lateral_acc`** — converts to a curvature cap
  `κ_max = a_lat / v²` evaluated at the current ego speed.
- **`planner.racing_line_side`** (`-1` right, `+1` left, `0` none) and
  **`racing_line_bias`** — bias toward the racing-line side when clearances
  are within `racing_line_bias` of each other.
- **`decision.plan_to_execute_count`** — feasible plans in a row before
  committing. Raise it for less twitchy commitment, lower for faster reaction.
- **`decision.plan_failure_count`** — consecutive infeasible plans before
  bailing back to FOLLOW (or to REMERGE while executing).

## State machine

```
        opponent ahead AND
        closing AND in range
FOLLOW ───────────────────────►  PLAN_OVERTAKE
   ▲                                  │
   │ remerge_done                     │ plan_to_execute_count
   │                                  ▼
REMERGE ◄──────────────── EXECUTE_OVERTAKE
   ▲      opp cleared            (publishes path + drive)
   │      OR abort               │
   └──────────────────────────────
```

Hysteresis lives in two counters: `feasible_streak` (for commit) and
`infeasible_streak` (for bail-out). Both reset on state transitions.

## How to tune

| If you see… | Try… |
|---|---|
| Planner flickers between sides each tick | Increase `plan_to_execute_count`, increase `w_offset` |
| Commits too late and rear-ends opponents | Increase `trigger_distance`, raise `w_obstacle`, drop `min_closing_speed` |
| Bails to FOLLOW mid-overtake | Increase `plan_failure_count`, widen `safety_margin` only if you know the track allows it |
| Path scrapes the wall | Reduce `track_width` parameter to match real track or widen `safety_margin` |
| Steering twitches at high speed | Lower `max_lateral_acc` (the curvature cap will reject sharper candidates), raise `follower.lookahead` |

## Running standalone

```bash
colcon build --packages-select overtaking_spline
source install/setup.bash
ros2 launch overtaking_spline overtaking_spline.launch.py
```

The node will idle (FOLLOW state, publishing only diagnostics) until it sees
a reference path and an odometry message.

## Tests

No ROS deps needed:

```bash
cd src/planning/overtaking_spline
python3 -m pytest test/
```

There is a hot-loop benchmark in `test_spline_planner.py` that asserts the
planner stays under 4 ms average over 100 iterations.

## Integration into f1tenth_stack (next pass, not done here)

When ready to wire into the main launch:

1. Add `StateTraits.OVERTAKING_SPLINE = auto()` in
   `src/decision/decision/fsm/state.py`.
2. Create `decision/fsm/states/overtaking_spline.py` and
   `overtaking_spline_only.py` mirroring `dwa.py` / `dwa_only.py`.
3. Add `"overtaking_spline"` to `__get_control_topic_from_current_state` in
   `fsm_node.py`.
4. In `bringup_launch.py`: a `use_overtaking_spline` arg, a `Node(...)` for
   this package gated by `IfCondition`, and copy `config/overtaking_spline.yaml`
   into `src/giu_f1t_system/f1tenth_stack/config/`.
5. Add a `/overtaking_spline/drive` route to `control_gateway_params.yaml`.

## Design notes / caveats

- **Track-width is static.** This matches the rest of the stack
  (`detection_config.yaml` also treats it as a parameter). A future pass
  could replace it with a live lidar corridor; for a known track on a known
  map the marginal gain is small and it would eat budget.
- **No iterative solver in the hot loop.** Each quintic is a 3×3 solve via
  `numpy.linalg.solve`; total planning is closed-form.
- **Opponent longitudinal speed is estimated** from Frenet differencing
  between consecutive `/object_centers` messages with a 0.8 EMA. If you need
  better, subscribe to `/object_velocities` and read the marker
  start→end vector — left as a TODO since the EMA is enough to gate
  closing-speed triggers.
- **Single quintic, not piecewise.** The maneuver uses a single quintic with
  endpoints `(d, 0, 0)` → `(0, 0, 0)` and an initial-slope offset chosen so
  the polynomial peaks near the opponent. This avoids the standard
  Werling two-piece (lateral × longitudinal) lattice and saves ~2 ms.
  Trade-off: less control over the peak location for sharply asymmetric
  trigger geometries. For the common F1TENTH overtake this is fine.
- **No tf lookups in the hot path.** Everything is in the `map` frame
  because both `/object_centers` and `/car_state/odom` are already there.
