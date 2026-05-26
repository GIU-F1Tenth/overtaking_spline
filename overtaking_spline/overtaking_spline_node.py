from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from overtaking_spline.frenet import FrenetConverter, build_reference_path
from overtaking_spline.spline_planner import PlannerConfig, SplinePlanner


class OvertakingSplineNode(Node):
    def __init__(self) -> None:
        super().__init__("overtaking_spline_node")

        # Topic parameters
        self.declare_parameter("reference_path_topic", "/pp_path")
        self.declare_parameter("odom_topic", "/car_state/odom")
        self.declare_parameter("object_centers_topic", "/object_centers")
        self.declare_parameter("overtake_path_topic", "/overtaking_spline/path")
        self.declare_parameter("diagnostics_topic", "/overtaking_spline/diagnostics")
        self.declare_parameter("candidates_topic", "/overtaking_spline/candidates")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_candidates", True)
        self.declare_parameter("publish_diagnostics", True)

        # Planner parameters
        self.declare_parameter("planner.car_width", 0.30)
        self.declare_parameter("planner.safety_margin", 0.20)
        self.declare_parameter("planner.track_width", 1.5)
        self.declare_parameter("planner.lookahead", 6.0)
        self.declare_parameter("planner.rejoin_distance", 4.0)
        self.declare_parameter("planner.n_samples", 50)
        self.declare_parameter("planner.n_candidates", 7)
        self.declare_parameter("planner.max_lateral_acc", 6.0)
        self.declare_parameter("planner.racing_line_side", 0)
        self.declare_parameter("planner.racing_line_bias", 0.10)
        self.declare_parameter("planner.w_obstacle", 50.0)
        self.declare_parameter("planner.w_curvature", 1.0)
        self.declare_parameter("planner.w_offset", 0.5)
        self.declare_parameter("planner.w_bias", 2.0)

        # Runtime
        self.declare_parameter("control_loop_frequency", 50.0)

        planner_cfg = PlannerConfig(
            car_width=self._p("planner.car_width"),
            safety_margin=self._p("planner.safety_margin"),
            track_width=self._p("planner.track_width"),
            lookahead=self._p("planner.lookahead"),
            rejoin_distance=self._p("planner.rejoin_distance"),
            n_samples=int(self._p("planner.n_samples")),
            n_candidates=int(self._p("planner.n_candidates")),
            max_lateral_acc=self._p("planner.max_lateral_acc"),
            racing_line_side=int(self._p("planner.racing_line_side")),
            racing_line_bias=self._p("planner.racing_line_bias"),
            w_obstacle=self._p("planner.w_obstacle"),
            w_curvature=self._p("planner.w_curvature"),
            w_offset=self._p("planner.w_offset"),
            w_bias=self._p("planner.w_bias"),
        )

        self.converter = FrenetConverter()
        self.planner = SplinePlanner(self.converter, planner_cfg)

        self.frame_id = str(self._p("frame_id"))
        self.publish_candidates = bool(self._p("publish_candidates"))
        self.publish_diagnostics = bool(self._p("publish_diagnostics"))
        
        # Ego state cache
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0
        self.ego_vx = 0.0
        self.ego_vy = 0.0
        self.have_ego = False

        # Opponent: nearest object center to ego (in arc-length forward sense).
        self.opp_xy: Optional[tuple] = None

        # Subscriptions
        self.create_subscription(
            Path, str(self._p("reference_path_topic")),
            self._on_reference_path, 10,
        )
        self.create_subscription(
            Odometry, str(self._p("odom_topic")),
            self._on_odom, qos_profile_sensor_data,
        )
        self.create_subscription(
            MarkerArray, str(self._p("object_centers_topic")),
            self._on_object_centers, 10,
        )

        # Publishers
        self.path_pub = self.create_publisher(
            Path, str(self._p("overtake_path_topic")), 10,
        )
        self.diag_pub = self.create_publisher(
            Float64MultiArray, str(self._p("diagnostics_topic")), 10,
        )
        self.candidates_pub = self.create_publisher(
            MarkerArray, str(self._p("candidates_topic")), 10,
        )

        freq = float(self._p("control_loop_frequency"))
        self.timer = self.create_timer(1.0 / freq, self._tick)

        self.get_logger().info(
            f"overtaking_spline_node up; loop {freq:.1f} Hz, "
            f"track_width={planner_cfg.track_width:.2f}m, "
            f"lookahead={planner_cfg.lookahead:.1f}m"
        )

    def _p(self, name: str):
        return self.get_parameter(name).value

    # -------- subscriptions --------

    def _on_reference_path(self, msg: Path) -> None:
        if len(msg.poses) < 2:
            return
        xs = np.fromiter(
            (p.pose.position.x for p in msg.poses), dtype=float, count=len(msg.poses)
        )
        ys = np.fromiter(
            (p.pose.position.y for p in msg.poses), dtype=float, count=len(msg.poses)
        )
        # csv_path_publisher stores velocity in pose.orientation.w
        vs = np.fromiter(
            (p.pose.orientation.w for p in msg.poses),
            dtype=float, count=len(msg.poses),
        )
        ref = build_reference_path(xs, ys, vs)
        self.converter.update_reference(ref)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        self.ego_x = p.x
        self.ego_y = p.y
        self.ego_vx = v.x
        self.ego_vy = v.y
        self.ego_yaw = float(np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        ))
        self.have_ego = True

    def _on_object_centers(self, msg: MarkerArray) -> None:
        # detection_node.publish_markers() prepends a DELETEALL marker, skip it.
        markers = [m for m in msg.markers[1:]
                   if m.action == Marker.ADD and m.ns == "object_centers"]
        if not markers or not self.have_ego or not self.converter.ready:
            self.opp_xy = None
            return

        # Pick closest by Euclidean distance to ego.
        best = None
        best_d2 = float("inf")
        for m in markers:
            dx = m.pose.position.x - self.ego_x
            dy = m.pose.position.y - self.ego_y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = (m.pose.position.x, m.pose.position.y)

        self.opp_xy = best

    # -------- main tick --------

    def _tick(self) -> None:
        t0 = time.perf_counter()
        if not self.have_ego or not self.converter.ready:
            self._publish_diag(0.0, 0.0, 0.0, 0)
            return

        ego_s, ego_d, _ = self.converter.cartesian_to_frenet(self.ego_x, self.ego_y)
        ego_speed_long = float(np.hypot(self.ego_vx, self.ego_vy))

        has_opp = self.opp_xy is not None
        ds_to_opp = 0.0
        opp_d = 0.0
        if has_opp:
            try:
                opp_s, opp_d, _ = self.converter.cartesian_to_frenet(*self.opp_xy)
                ds_to_opp = self.converter.wrap_delta_s(opp_s, ego_s)
                # Map forward-wrap to signed offset so "behind" is negative.
                if ds_to_opp > self.converter.reference.total_length * 0.5:
                    ds_to_opp -= self.converter.reference.total_length
            except RuntimeError:
                has_opp = False

        t_plan_start = time.perf_counter()
        plan = None
        if has_opp and 0.0 < ds_to_opp <= self.planner.config.lookahead:
            plan = self.planner.plan(
                ego_s=ego_s, ego_d=ego_d,
                opp_s=ego_s + ds_to_opp, opp_d=opp_d,
                ego_speed=ego_speed_long,
            )
        t_plan = (time.perf_counter() - t_plan_start) * 1e3

        chosen = plan.chosen if plan is not None else None

        if chosen is not None:
            self._publish_path(chosen)

        if self.publish_candidates and plan is not None:
            self._publish_candidates(plan.candidates,
                                     getattr(chosen, "d_target", None))

        total_ms = (time.perf_counter() - t0) * 1e3
        side = 0 if chosen is None else int(np.sign(chosen.d_target))
        
        if self.publish_diagnostics:
            self._publish_diag(t_plan, total_ms, ego_speed_long, side)

    # -------- publishers --------

    def _publish_path(self, chosen) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        # Decorate each pose with the reference velocity at that arc-length
        # so downstream pure-pursuit reads orientation.w just like /pp_path.
        vs = self.converter.velocity_at_batch(chosen.s_samples)
        for x, y, v in zip(chosen.xs, chosen.ys, vs):
            p = PoseStamped()
            p.header.frame_id = self.frame_id
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.orientation.w = float(v)
            msg.poses.append(p)
        self.path_pub.publish(msg)

    def _publish_candidates(self, candidates, chosen_d_target) -> None:
        arr = MarkerArray()
        arr.markers.append(Marker(action=Marker.DELETEALL))
        for i, c in enumerate(candidates):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "candidates"
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.03
            is_chosen = (chosen_d_target is not None
                         and abs(c.d_target - chosen_d_target) < 1e-6)
            if is_chosen:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
            elif c.feasible:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.5, 1.0, 0.6
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.2, 0.2, 0.4
            for x, y in zip(c.xs, c.ys):
                from geometry_msgs.msg import Point
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                m.points.append(p)
            arr.markers.append(m)
        self.candidates_pub.publish(arr)

    def _publish_diag(self, plan_ms: float, total_ms: float, ego_speed: float,
                      side: int) -> None:
        msg = Float64MultiArray()
        msg.data = [
            float(plan_ms),
            float(total_ms),
            float(ego_speed),
            float(side),
        ]
        self.diag_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OvertakingSplineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
