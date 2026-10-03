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

    The racket-head path is drawn in two segments — before/at impact and after
    impact — so the follow-through is clearly visible instead of appearing to
    end at the impact peak.  The racket itself is drawn as a line from the
    pivot (wrist) to the head tip for a sample of orientations.
    """
    fig = go.Figure()
    imp = swing.impact_idx

    # Racket-head trajectory (pivot model) — primary curve, split at impact.
    pre = slice(0, imp + 1)
    post = slice(imp, None)
    fig.add_trace(go.Scatter3d(
        x=swing.tip[pre, 0], y=swing.tip[pre, 1], z=swing.tip[pre, 2],
        mode="lines",
        line=dict(color="#00E5FF", width=4),
        name="Path → impact",
        hovertext=[f"t={t:.3f}s (pre)" for t in swing.t[pre]],
        hovertemplate="%{hovertext}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=swing.tip[post, 0], y=swing.tip[post, 1], z=swing.tip[post, 2],
        mode="lines",
        line=dict(color="#69F0AE", width=4),
        name="Path after impact (follow-through)",
        hovertext=[f"t={t:.3f}s (post)" for t in swing.t[post]],
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
    fig.add_trace(go.Scatter3d(
        x=[swing.tip[imp, 0]], y=[swing.tip[imp, 1]], z=[swing.tip[imp, 2]],
        mode="markers",
        marker=dict(size=8, color="#FF0000"),
        name=f"Impact (t={swing.t[imp]:.2f}s)",
    ))

    # Axis-locked, equal aspect for a truthful shape.
    fig.update_scenes(aspectmode="data", xaxis_title="x (m)",
                      yaxis_title="y (m)",
                      zaxis_title="z (m)")
    fig.update_layout(
        template="plotly_dark",
        height=620,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Racket-head trajectory (IMU-derived, string-ringing filtered)",
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
