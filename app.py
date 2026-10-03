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
    """Load the swing contract + video metadata (cached for smooth demo)."""
    from src.swing import Swing
    from src.video import probe

    swing_path = config.SWING_NPZ
    if not Path(swing_path).exists():
        st.error(f"Missing contract file: {swing_path}. "
                 f"Run `python scripts/verify_fusion.py` or `mock_swing.py`.")
        st.stop()

    swing = Swing.load(swing_path)

    videos = []
    for p in sorted(config.VIDEO_DIR.glob("*.mp4")):
        videos.append((p.name, probe(str(p))))

    from src.metrics import compute_metrics
    metrics = compute_metrics(swing)
    return swing, videos, metrics


@st.cache_data(show_spinner=False)
def get_frame(meta_path: str, idx: int) -> "np.ndarray":
    from src.video import frame
    return frame(meta_path, idx)


def render_overlay_controls(swing):
    """Sidebar controls for the overlay (camera, align, layers)."""
    from src import camera  # noqa: PLC0415

    with st.sidebar:
        st.subheader("🔧 Align & Overlay")

        fov = st.slider("Camera FOV (deg)", 40.0, 100.0, 60.0, 1.0,
                        help="Intrinsics assumption; refine until the path "
                             "lands on the racket.")

        st.markdown("**Slider nudge** (fallback alignment)")
        yaw = st.slider("yaw", -20.0, 20.0, 0.0, 0.5)
        elev = st.slider("elevation", -20.0, 20.0, 0.0, 0.5)
        zoom = st.slider("zoom", 0.5, 2.0, 1.0, 0.01)
        pan_x = st.slider("pan x", -80.0, 80.0, 0.0, 2.0)
        pan_y = st.slider("pan y", -80.0, 80.0, 0.0, 2.0)

        st.markdown("**Rough time sync**")
        scale = st.number_input("slow-mo scale", 1.0, 12.0,
                                config.SLOWMO_SCALE, 0.1)
        offset = st.number_input("sync offset (s)", -3.0, 3.0, 0.0, 0.01)

        show_path = st.checkbox("path", True)
        show_dot = st.checkbox("animated dot", True)
        show_impact = st.checkbox("impact marker", True)

        return dict(
            fov=fov, yaw=yaw, elev=elev, zoom=zoom, pan_x=pan_x, pan_y=pan_y,
            scale=scale, offset=offset,
            show_path=show_path, show_dot=show_dot, show_impact=show_impact,
        )


def build_overlay_camera(meta, opts):
    """Build a Camera for the current view from intrinsics + nudge sliders."""
    from src import camera
    import numpy as _np
    K = camera.estimate_intrinsics(meta.width, meta.height, opts["fov"])
    cam = camera.Camera.from_pnps(
        K, _np.zeros(5),
        _np.array([0.0, 0.0, 0.0]), _np.array([0.0, 0.0, 10.0]),
        meta.width, meta.height,
    )
    return camera.nudge(cam, yaw=opts["yaw"], elev=opts["elev"], zoom=opts["zoom"],
                        pan_x=opts["pan_x"], pan_y=opts["pan_y"])


def main():
    st.set_page_config(page_title="IMU Swing → Video Overlay",
                       layout="wide")
    st.title("🎾 Hack4Humanity — IMU Racket Path Projection")

    swing, videos, metrics = load_artifacts()

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

            cam = build_overlay_camera(meta, opts)

            # ---- Mode C fallback: side-by-side (video | 3D) ---- #
            side_by_side = st.checkbox(
                "Side-by-side fallback (video next to 3D)", value=False,
                help="Safety net: show the slowed video beside the IMU 3D "
                     "view with no projection calibration required.")

            if side_by_side:
                col_l, col_r = st.columns(2)
                img = get_frame(meta.path, frame_idx_default(meta, metrics))
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
                                   frame_idx_default(meta, metrics))
                img2 = get_frame(meta.path, f2)
                st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),
                         caption=f"frame {f2}")
            else:
                if "playback" not in st.session_state:
                    st.session_state.playback = False

                col_play, col_scrub, col_fps = st.columns([1, 6, 2])
                with col_play:
                    if st.button("▶ Play" if not st.session_state.playback else "⏸ Pause"):
                        st.session_state.playback = not st.session_state.playback
                with col_scrub:
                    frame_idx = st.slider("frame", 0, meta.n_frames - 1,
                                          frame_idx_default(meta, metrics),
                                          key="scrub")
                with col_fps:
                    play_speed = st.selectbox("play speed", [1, 2, 4, 8], index=0)

                # ---- Overlay the swing path on the current frame ---- #
                t_video = frame_idx / meta.fps
                img = get_frame(meta.path, frame_idx)

                from src import camera as _cam
                overlaid = _cam.draw_overlay(
                    img, cam, swing, t_video, meta.fps,
                    scale=opts["scale"], offset=opts["offset"],
                    show_path=opts["show_path"], show_dot=opts["show_dot"],
                    show_impact=opts["show_impact"],
                )
                st.image(cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB),
                         caption=f"frame {frame_idx} · t_video={t_video:.2f}s")

                st.caption("🔵 path (pivot model) · 🔴 animated dot · "
                           "impact ring · grey = sliders aligned to racket")

                # ---- Simple playback loop via placeholder ---- #
                import time  # noqa: PLC0415
                placeholder = st.empty()
                if st.session_state.playback:
                    idx = frame_idx
                    step = [1, 2, 4, 8][play_speed_index(play_speed)]
                    n = meta.n_frames
                    while st.session_state.playback:
                        idx = (idx + step) % n
                        t_vlog = idx / meta.fps
                        fr = get_frame(meta.path, int(idx))
                        ov = _cam.draw_overlay(
                            fr, cam, swing, t_vlog, meta.fps,
                            scale=opts["scale"], offset=opts["offset"],
                            show_path=opts["show_path"], show_dot=opts["show_dot"],
                            show_impact=opts["show_impact"])
                        placeholder.image(
                            cv2.cvtColor(ov, cv2.COLOR_BGR2RGB),
                            caption=f"frame {idx}")
                        time.sleep(0.02)
                        st.session_state.scrub = idx
                        st.session_state.playback = True
                    placeholder.empty()

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


def play_speed_index(value) -> int:
    return [1, 2, 4, 8].index(value)


def frame_idx_default(meta, metrics) -> int:
    """Default scrub position = the impact frame (roughly synced)."""
    imp_frame = int(metrics["impact_time_s"] * config.SLOWMO_SCALE * meta.fps)
    return min(meta.n_frames - 1, max(0, imp_frame))


if __name__ == "__main__":
    main()
