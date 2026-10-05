"""Person B — Streamlit dashboard (PLAN.md §7 Phase 2: `app.py`).

The judge-facing demo: scrub/play a slow-motion video with the IMU-derived
racket path overlaid (path-projection), plus a free-rotation 3D view and the
metrics panel.

Run::

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import streamlit as st  # noqa: E402

from src import (  # noqa: E402
    config,
)


@st.cache_data(show_spinner=False)
def load_artifacts():
    """Load the swing contract + video metadata + sync tables (cached)."""
    from src.swing import Swing
    from src.video import probe
    from src.synced_data import SyncedData

    swing_path = config.SWING_NPZ
    if not Path(swing_path).exists():
        st.error(f"Missing contract file: {swing_path}. "
                 f"Run `python scripts/verify_fusion.py` or `mock_swing.py`.")
        st.stop()

    swing = Swing.load(swing_path)

    videos = []
    for p in sorted(config.VIDEO_DIR.glob("*.mp4")):
        videos.append((p.name, probe(str(p))))

    # Per-video IMU<->frame sync tables (from scripts/run_sync.py).
    synced = SyncedData.all_angles()

    from src.metrics import compute_metrics
    metrics = compute_metrics(swing)
    return swing, videos, metrics, synced


@st.cache_data(show_spinner=False)
def get_frame(meta_path: str, idx: int) -> "np.ndarray":
    from src.video import frame
    return frame(meta_path, idx)


@st.cache_data(show_spinner=False)
def get_impact_pixel(video_path: str) -> tuple[float, float] | None:
    """Find the ball-contact pixel using the sync pipeline's ball detector."""
    import numpy as np
    from src.sync.video_motion import ball_contact, motion_energy
    from src.sync.video_probe import probe_video

    path = Path(video_path)
    probe = probe_video(path)
    energy = motion_energy(path, probe.frame_times)
    lo = max(0, energy.peak_frame - 25)
    hi = min(probe.decoded_frame_count - 1, energy.peak_frame + 25)
    contact = ball_contact(path, lo, hi)
    if not contact.found or not contact.is_strike or len(contact.frames) == 0:
        return None
    sample = int(np.argmin(np.abs(contact.frames - contact.contact_frame_exact)))
    return float(contact.xs[sample]), float(contact.ys[sample])


def render_overlay_controls(swing):
    """Sidebar controls for the overlay (camera, align, layers)."""
    from src import overlay_calib  # noqa: PLC0415

    with st.sidebar:
        st.subheader("🔧 Align & Overlay")

        fov = st.slider("Camera FOV (deg)", 40.0, 100.0,
                        overlay_calib.CALIB_FOV_DEG, 1.0,
                        help="Intrinsics assumption. The default is the value "
                             "the pose was solved at; the reprojection error "
                             "barely moves across this whole range, so the "
                             "footage does not pin it down (see "
                             "scripts/check_overlay_match.py).")

        st.markdown("**Slider nudge** (adjustment on top of the solved pose)")
        yaw = st.slider("yaw", -20.0, 20.0, 0.0, 0.5,
                        help="Rotates the camera about its own centre, which "
                             "now sits ~3 m from the swing rather than the old "
                             "fabricated 10 m — so a few degrees moves the "
                             "path much further than it used to.")
        elev = st.slider("elevation", -20.0, 20.0, 0.0, 0.5)
        zoom = st.slider("zoom", 0.5, 2.0, 1.0, 0.01)
        pan_x = st.slider("pan x", -80.0, 80.0, 0.0, 2.0)
        pan_y = st.slider("pan y", -80.0, 80.0, 0.0, 2.0)

        anchor_impact = st.checkbox(
            "anchor impact to ball pixel (legacy)", value=False,
            help="Pre-calibration behaviour: slide the whole path until the "
                 "projected impact point lands on the detected ball-contact "
                 "pixel. With a pose fitted to 14 correspondences this moves "
                 "every other frame by that one point's residual, so it is "
                 "off by default. Turning it on also costs a full video "
                 "decode on first render.")

        st.markdown("**Rough time sync**")
        scale = st.number_input("slow-mo scale", 1.0, 12.0,
                                config.SLOWMO_SCALE, 0.1,
                                help="config.SLOWMO_SCALE is now 8.0 "
                                     "(240 fps capture / 30 fps playback, "
                                     "measured in scripts/measure_capture_fps.py). "
                                     "Used only as a fallback for clips with "
                                     "no per-video sync table — swing_angle_1.mp4 "
                                     "uses the exact rescaled frame_exact "
                                     "mapping from SyncedData instead.")
        offset = st.number_input("sync offset (s)", -3.0, 3.0, 0.0, 0.01)

        show_path = st.checkbox("path", True)
        show_dot = st.checkbox("animated dot", True)
        show_impact = st.checkbox("impact marker", True)

        return dict(
            fov=fov, yaw=yaw, elev=elev, zoom=zoom, pan_x=pan_x, pan_y=pan_y,
            scale=scale, offset=offset, anchor_impact=anchor_impact,
            show_path=show_path, show_dot=show_dot, show_impact=show_impact,
        )


# Legacy pose, used only for clips with no committed calibration.  It is a
# basis conversion plus a guess at the distance, not a calibration — see
# OVERLAY_FRAME.md.
LEGACY_RVEC = (1.5707963267948966, 0.0, 0.0)   # [+pi/2, 0, 0]
LEGACY_TVEC = (0.0, 0.0, 10.0)


def build_overlay_camera(meta, opts, swing=None, impact_pixel=None,
                         video_name=None):
    """Build the projection camera for one clip.

    For ``swing_angle_1.mp4`` the rotation and translation come from
    ``src.overlay_calib.solved_pose`` — a ``solvePnP`` fit to the committed
    racket-head annotations, recomputed here (memoised, microseconds) rather
    than baked in as constants, so the app and
    ``scripts/check_overlay_match.py`` can never disagree.  The Z-up world ->
    Y-down camera conversion that ``OVERLAY_FRAME.md`` describes still applies;
    it is now absorbed into the solved rotation instead of being written out.

    Any other clip falls back to the old guessed pose and says so.  The
    intrinsics always come from the FOV slider, so that slider still visibly
    changes the projection; its default is the FOV the pose was solved at.
    """
    from src import camera, overlay_calib
    import numpy as _np

    K = camera.estimate_intrinsics(meta.width, meta.height, opts["fov"])
    rvec, tvec = _np.array(LEGACY_RVEC), _np.array(LEGACY_TVEC)

    if video_name is not None and overlay_calib.has_annotations(video_name):
        try:
            sol = overlay_calib.solved_pose(meta.width, meta.height,
                                            float(opts["fov"]))
        except Exception as exc:                       # noqa: BLE001
            st.warning(f"⚠️ Pose solve failed for `{video_name}` ({exc}); "
                       "falling back to the uncalibrated guess.")
        else:
            if sol.ok:
                rvec, tvec = sol.rvec, sol.tvec
            else:
                st.warning(f"⚠️ No valid pose for `{video_name}`: "
                           f"{'; '.join(sol.messages) or 'solver failed'}. "
                           "Falling back to the uncalibrated guess.")
    elif video_name is not None:
        st.info(f"ℹ️ No camera calibration exists for `{video_name}` — the "
                "overlay pose is a guess, not a fit. Use the sliders.")

    cam = camera.Camera.from_pnps(K, _np.zeros(5), rvec, tvec,
                                  meta.width, meta.height)
    cam = camera.nudge(cam, yaw=opts["yaw"], elev=opts["elev"],
                       zoom=opts["zoom"])
    pan_x, pan_y = opts["pan_x"], opts["pan_y"]
    if opts.get("anchor_impact") and swing is not None and impact_pixel is not None:
        projected = camera.project_points(cam, swing.head[swing.impact_idx])[0]
        pan_x += float(impact_pixel[0] - projected[0])
        pan_y += float(impact_pixel[1] - projected[1])
    return camera.nudge(cam, pan_x=pan_x, pan_y=pan_y)


def main():
    st.set_page_config(page_title="IMU Swing → Video Overlay",
                       layout="wide")
    st.title("🎾 Hack4Humanity — IMU Racket Path Projection")

    swing, videos, metrics, synced = load_artifacts()

    tab_video, tab_3d, tab_signals, tab_metrics = st.tabs(
        ["Swing + Video", "3D Trajectory", "Signals", "Metrics"])

    # ---------------- Overview metrics chips ---------------- #
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak head speed", f"{metrics['peak_head_speed_kmh']} km/h")
    m2.metric("Peak g-force", f"{metrics['peak_gforce']} g")
    m3.metric("Peak RPM", f"{metrics['peak_rpm']} rpm")
    m4.metric("Swing duration", f"{metrics['swing_duration_s']} s")

    # ---------------- Sidebar overlay controls ---------------- #
    opts = render_overlay_controls(swing)

    with tab_video:
        if not videos:
            st.warning("No MP4 found in data/video — 3D view still available.")
        else:
            names = [name for name, _ in videos]
            sel = st.selectbox("Camera", names)
            meta = dict(videos)[sel]
            sync_table = synced.get(sel)
            syncable = bool(sync_table is not None and sync_table.syncable)
            # Only decode the whole video for the ball detector when the
            # legacy one-point anchor is actually switched on — that decode is
            # the ~60 s first render.
            impact_pixel = (get_impact_pixel(meta.path)
                            if syncable and opts["anchor_impact"] else None)
            cam = build_overlay_camera(meta, opts, swing, impact_pixel,
                                       video_name=sel)

            # ---- Mode C fallback: side-by-side (video | 3D) ---- #
            side_by_side = st.checkbox(
                "Side-by-side fallback (video next to 3D)", value=False,
                help="Safety net: show the slowed video beside the IMU 3D "
                     "view with no projection calibration required.")

            if side_by_side:
                col_l, col_r = st.columns(2)
                img = get_frame(meta.path, frame_idx_default(meta, metrics,
                                                             sync_table))
                with col_l:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                             caption="slow-motion video")
                with col_r:
                    from src.plot import figure_3d
                    st.plotly_chart(figure_3d(swing, metrics), width="stretch",
                                    key="fig3d_side")
                st.caption("Mode C — same shared time axis, no pose needed.")
                col_l2, _col_sp = st.columns([1, 1])
                with col_l2:
                    f2 = st.slider("frame", 0, meta.n_frames - 1,
                                   frame_idx_default(meta, metrics, sync_table))
                img2 = get_frame(meta.path, f2)
                st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),
                         caption=f"frame {f2}")
            else:
                if not syncable:
                    st.warning(
                        f"⚠️ `{sel}` No sync data is available"
                    )
                _player(meta, cam, swing, opts, metrics, sync_table)

    with tab_3d:
        from src.plot import figure_3d
        st.plotly_chart(figure_3d(swing, metrics), width="stretch", key="fig3d_main")
        st.caption("Red point = impact. White fragments = racket orientation "
                   "sticks. Blue line = pivot-model head path.")

    with tab_signals:
        from src.plot import figure_signals
        st.plotly_chart(figure_signals(swing), width="stretch", key="figsignals")

    with tab_metrics:
        st.subheader("Metrics panel")
        for k, v in metrics.items():
            st.write(f"**{k}:** {v}")
        st.caption("Head speed uses the pivot model (v = |ω| × r).")


def frame_idx_default(meta, metrics, sync_table=None) -> int:
    """Default scrub position.

    With a sync table, this is the video frame matching the IMU impact
    (e.g. angle_1: sample 205 -> frame ~132).  Without one, it falls back to
    the old rough mapping (scale * fps).
    """
    if sync_table is not None and sync_table.syncable:
        return min(meta.n_frames - 1,
                   max(0, sync_table.impact_frame(metrics["impact_idx"])))
    imp_frame = int(metrics["impact_time_s"] * config.SLOWMO_SCALE * meta.fps)
    return min(meta.n_frames - 1, max(0, imp_frame))


# --------------------------------------------------------------------------- #
# Video player (fragment) — scrub + play/pause
# --------------------------------------------------------------------------- #
def _toggle_playback() -> None:
    st.session_state.playback = not st.session_state.playback


def _sync_scrub() -> None:
    """Slider on_change: keep frame_pos in sync with the widget value."""
    st.session_state.frame_pos = st.session_state.scrub


@st.fragment(run_every=0.1)
def _player(meta, cam, swing, opts, metrics, sync_table=None):
    """Overlay player: scrub slider + play/pause with an animated dot.

    ``frame_pos`` (a plain session-state value, NOT a widget key) is the
    source of truth.  The slider widget state is only ever read here and in
    its own ``on_change`` callback — never assigned after instantiation.

    When a sync table exists, the animated dot is placed with the exact
    IMU sample that maps to the current frame (``dot_idx``); otherwise the
    rough ``scale/offset`` mapping is used.
    """
    from src import camera as _cam

    st.session_state.setdefault("frame_pos", frame_idx_default(meta, metrics,
                                                               sync_table))
    st.session_state.setdefault("playback", False)

    # Update the widget key before the slider is instantiated. This keeps its
    # thumb aligned with the frame advanced by the playback fragment.
    st.session_state.scrub = st.session_state.frame_pos

    col_play, col_scrub, col_fps = st.columns([1, 6, 2])
    with col_play:
        st.button("▶ Play" if not st.session_state.playback else "⏸ Pause",
                  on_click=_toggle_playback)
    with col_scrub:
        st.slider("frame", 0, meta.n_frames - 1,
                  key="scrub", on_change=_sync_scrub)
    with col_fps:
        play_speed = st.selectbox("play speed", ["1×", "2×", "4×", "8×"],
                                  index=0)

    idx = st.session_state.frame_pos
    if st.session_state.playback:
        # Advance at the video's true rate: with run_every=0.1 s, stepping
        # fps*0.1*factor frames per tick plays the clip at the selected speed.
        factor = [1, 2, 4, 8][["1×", "2×", "4×", "8×"].index(play_speed)]
        step = max(1, int(round(meta.fps * 0.1 * factor)))
        st.session_state.frame_pos = (idx + step) % meta.n_frames

    # ---- Overlay the swing path on the current frame ---- #
    t_video = idx / meta.fps
    img = get_frame(meta.path, int(idx))

    covered = True
    if sync_table is not None and sync_table.syncable:
        # Centred, not `imu_idx_for_frame`: that one argmins over the rounded
        # frame index and so returns the FIRST sample of the frame's bin.
        dot_idx = sync_table.imu_idx_for_frame_centered(int(idx))
        covered = sync_table.covers_frame(int(idx))
        scale, offset = 1.0, 0.0   # unused when dot_idx is passed
    else:
        dot_idx = None
        scale, offset = opts["scale"], opts["offset"]

    overlaid = _cam.draw_overlay(
        img, cam, swing, t_video, meta.fps,
        scale=scale, offset=offset, dot_idx=dot_idx,
        show_path=opts["show_path"] and covered,
        show_dot=opts["show_dot"] and covered,
        show_impact=opts["show_impact"],
    )
    st.image(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB),
             caption=f"frame {idx} / {meta.n_frames - 1} · "
                     f"t_video={t_video:.2f}s")
    st.caption("🔵 path (pivot model) · 🔴 animated dot · "
               "impact ring · grey = sliders aligned to racket")
    if not covered:
        st.caption("⚠️ The 400-sample IMU record does not reach this frame "
                   f"({idx}); at {config.CAPTURE_FPS:g} fps capture it covers "
                   "roughly frames 15-245. Overlay hidden rather than frozen "
                   "at the record's first or last sample.")


if __name__ == "__main__":
    main()
