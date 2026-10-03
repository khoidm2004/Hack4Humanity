"""Person B — Plotly figures (PLAN.md §7 Phase 2: `plot.py`).

Provides the two interactive visualisations used by the demo:
  * a free-rotation 3D view of the racket orientation + head trajectory,
  * a 6-channel signal plot with the impact moment highlighted.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .swing import Swing


# --------------------------------------------------------------------------- #
# 3D trajectory view
# --------------------------------------------------------------------------- #
def figure_3d(swing: Swing, metrics: dict | None = None) -> go.Figure:
    """Build the interactive 3D racket + racket-head-trajectory figure.

    The racket is drawn as a line from the pivot (wrist) to the head tip for a
    sample of orientations; the tip is the animated point of interest.
    """
    fig = go.Figure()

    # Racket-head trajectory (pivot model) — primary curve.
    fig.add_trace(go.Scatter3d(
        x=swing.tip[:, 0], y=swing.tip[:, 1], z=swing.tip[:, 2],
        mode="lines",
        line=dict(color="#00E5FF", width=4),
        name="Racket path (pivot)",
        hovertext=[f"t={t:.3f}s" for t in swing.t],
        hovertemplate="%{hovertext}<extra></extra>",
    ))

    # A few racket sticks to show orientation.
    n_sticks = 26
    skip = max(1, len(swing.tip) // n_sticks)
    for i in range(0, len(swing.tip), skip):
        tip = swing.tip[i]
        fig.add_trace(go.Scatter3d(
            x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.25)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Sensor (rest-to-rest) path — data-driven, thinner.
    fig.add_trace(go.Scatter3d(
        x=swing.cog[:, 0], y=swing.cog[:, 1], z=swing.cog[:, 2],
        mode="lines", line=dict(color="#FFB300", width=2, dash="dot"),
        name="Sensor path (rest-to-rest)", hoverinfo="skip",
    ))

    # Impact point.
    imp = swing.tip[swing.impact_idx]
    fig.add_trace(go.Scatter3d(
        x=[imp[0]], y=[imp[1]], z=[imp[2]], mode="markers",
        marker=dict(size=8, color="#FF0000"),
        name=f"Impact (t={swing.t[swing.impact_idx]:.2f}s)",
    ))

    # Axis-locked, equal aspect for a truthful shape.
    fig.update_scenes(aspectmode="data", xaxis_title="x (m)",
                      yaxis_title="y (m)",
                      zaxis_title="z (m)")
    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Racket-head trajectory (IMU-derived)",
        legend=dict(orientation="h", y=1.02),
    )
    return fig


# --------------------------------------------------------------------------- #
# 6-channel signals
# --------------------------------------------------------------------------- #
def figure_signals(swing: Swing) -> go.Figure:
    """Plot the 6 IMU channels vs time, highlighting the impact instant."""
    t = swing.t
    labels = ["ax (m/s²)", "ay (m/s²)", "az (m/s²)",
              "gx (deg/s)", "gy (deg/s)", "gz (deg/s)"]
    data = np.hstack([swing.accel, swing.gyro_deg])

    fig = go.Figure()
    for i in range(6):
        fig.add_trace(go.Scatter(
            x=t, y=data[:, i], mode="lines", name=labels[i],
            line=dict(width=1.4),
        ))

    t_imp = t[swing.impact_idx]
    fig.add_vline(x=t_imp, line_dash="dash", line_color="red",
                  annotation_text=f"impact {t_imp:.3f}s",
                  annotation_position="top left")

    fig.update_layout(
        template="plotly_dark",
        height=560,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Raw IMU signals (impact highlighted)",
        xaxis_title="time (s)",
        legend=dict(orientation="h", y=1.02),
    )
    return fig
