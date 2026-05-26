# overtaking_spline

Frenet-frame quintic-spline overtaking **path generator** for the GIU Berlin
F1TENTH stack. Closed-form, single-digit-millisecond per cycle.

This package's sole responsibility is to **generate and publish candidate
overtake splines** whenever an opponent is detected ahead. It does **not**
decide whether to commit to an overtake and it does **not** execute one --
both of those concerns are handled downstream (e.g. by the FSM /
`control_gateway`).

## What it does

1. Builds a KD-tree-backed Frenet frame from the reference raceline (`/pp_path`).
2. Whenever an opponent is in front and within `planner.lookahead`,
   evaluates 5–9 quintic lateral candidates `d(s)` over a `lookahead`-meter
   window.
3. Scores candidates on clearance to the opponent, max curvature feasibility
   at the local reference speed (`κ ≤ a_lat_max / v²`), and a configurable
   racing-line-side bias.
4. Publishes the best-scoring feasible candidate as `nav_msgs/Path` and the
   full candidate set as a `visualization_msgs/MarkerArray` for debugging.

## Topics

| Direction | Topic                              | Type                              | Notes |
|---|---|---|---|
| sub | `/pp_path`                         | `nav_msgs/Path`                    | velocity in `pose.orientation.w` (convention from `csv_path_pub`) |
| sub | `/car_state/odom`                  | `nav_msgs/Odometry`                | sensor QoS |
| sub | `/object_centers`                  | `visualization_msgs/MarkerArray`   | publisher prepends a `DELETEALL`; we skip `[0]` |
| pub | `/overtaking_spline/path`          | `nav_msgs/Path`                    | best-scoring overtake path (with per-pose ref velocity in `orientation.w`) |
| pub | `/overtaking_spline/diagnostics`   | `std_msgs/Float64MultiArray`       | `[plan_ms, total_ms, ego_speed, side]` |
| pub | `/overtaking_spline/candidates`    | `visualization_msgs/MarkerArray`   | debug viz (toggle with `publish_candidates`) |

`side` in diagnostics: `-1` right, `+1` left, `0` no chosen candidate this tick.

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

## How to tune

| If you see… | Try… |
|---|---|
| Chosen side flips between ticks | Increase `w_offset` |
| Path scrapes the wall | Reduce `track_width` parameter to match real track or widen `safety_margin` |
| No candidate ever feasible at high speed | Lower `max_lateral_acc` (the curvature cap will reject sharper candidates) or widen `lookahead` |

## Running standalone

```bash
colcon build --packages-select overtaking_spline
source install/setup.bash
ros2 launch overtaking_spline overtaking_spline.launch.py
```

The node idles (publishing only diagnostics) until it sees a reference path
and an odometry message.

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

1. Add a `Node(...)` for this package to `bringup_launch.py` gated by a
   `use_overtaking_spline` arg.
2. Copy `config/overtaking_spline.yaml` into
   `src/giu_f1t_system/f1tenth_stack/config/`.
3. Subscribe to `/overtaking_spline/path` from whichever node owns the
   overtake commit decision and routes execution downstream.

## Design notes / caveats

- **Track-width is static.** This matches the rest of the stack
  (`detection_config.yaml` also treats it as a parameter). A future pass
  could replace it with a live lidar corridor; for a known track on a known
  map the marginal gain is small and it would eat budget.
- **No iterative solver in the hot loop.** Each quintic is a 3×3 solve via
  `numpy.linalg.solve`; total planning is closed-form.
- **Single quintic, not piecewise.** The maneuver uses a single quintic with
  endpoints `(d, 0, 0)` → `(0, 0, 0)` and an initial-slope offset chosen so
  the polynomial peaks near the opponent. This avoids the standard
  Werling two-piece (lateral × longitudinal) lattice and saves ~2 ms.
  Trade-off: less control over the peak location for sharply asymmetric
  trigger geometries. For the common F1TENTH overtake this is fine.
- **No tf lookups in the hot path.** Everything is in the `map` frame
  because both `/object_centers` and `/car_state/odom` are already there.
