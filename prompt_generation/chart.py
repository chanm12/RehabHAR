"""
RehabHAR Chart Utilities
Based on: LanHAR prompt_generation/chart.py

Generates Matplotlib charts of IMU data for visual analysis.
Returns base64-encoded images for VLM input.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
import io
from pathlib import Path

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
})


def generate_imu_chart(
    acc: np.ndarray,
    gyro: np.ndarray,
    fs: float = 50.0,
    title: str = "IMU Sensor Window",
    save_path: str | None = None,
    mode: str = "full",
    rep_boundaries: list[float] | None = None,
) -> str:
    """
    Generate a multi-panel time-series chart for a single IMU window.

    Parameters
    ----------
    acc  : (T, 3) gravity-aligned accelerometer data (m/s²)
    gyro : (T, 3) gyroscope data (rad/s)
    fs   : sampling rate in Hz
    title: chart super-title
    save_path : if provided, save PNG to this path
    mode : "full" for 4 panels (XYZ + magnitudes), "compact" for 2 panels (magnitudes only)
    rep_boundaries : optional list of times (in seconds) to draw as vertical rep markers

    Returns
    -------
    base64-encoded PNG string (for direct use in a vision API)
    """
    T = acc.shape[0]
    t = np.arange(T) / fs

    acc_svm = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    dpi = 200

    if mode == "compact":
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    else:
        fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

    fig.suptitle(title, fontsize=14, fontweight="bold")

    if mode == "compact":
        # Panel 1: Accelerometer SVM
        ax = axes[0]
        ax.plot(t, acc_svm, color="tab:red", linewidth=1.2)
        ax.set_ylabel("SVM (m/s²)")
        ax.set_title("Accel SVM")
        ax.grid(True, alpha=0.3)

        # Panel 2: Gyroscope magnitude
        ax = axes[1]
        ax.plot(t, gyro_mag, color="tab:purple", linewidth=1.2)
        ax.set_ylabel("|ω| (rad/s)")
        ax.set_xlabel("Time (s)")
        ax.set_title("Gyro |ω|")
        ax.grid(True, alpha=0.3)
    else:
        # Panel 1: Accelerometer X/Y/Z
        ax = axes[0]
        ax.plot(t, acc[:, 0], label="Acc X", color="tab:blue", linewidth=1.2)
        ax.plot(t, acc[:, 1], label="Acc Y", color="tab:green", linewidth=1.2)
        ax.plot(t, acc[:, 2], label="Acc Z (vertical)", color="tab:orange", linewidth=1.2)
        ax.set_ylabel("Acceleration (m/s²)")
        ax.set_title("Accel XYZ")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Panel 2: Accelerometer SVM magnitude
        ax = axes[1]
        ax.plot(t, acc_svm, color="tab:red", linewidth=1.2)
        ax.set_ylabel("SVM (m/s²)")
        ax.set_title("Accel SVM")
        ax.grid(True, alpha=0.3)

        # Panel 3: Gyroscope X/Y/Z
        ax = axes[2]
        ax.plot(t, gyro[:, 0], label="Gyro X (roll)", color="tab:blue", linewidth=1.2)
        ax.plot(t, gyro[:, 1], label="Gyro Y (pitch)", color="tab:green", linewidth=1.2)
        ax.plot(t, gyro[:, 2], label="Gyro Z (yaw)", color="tab:orange", linewidth=1.2)
        ax.set_ylabel("Angular vel. (rad/s)")
        ax.set_title("Gyro XYZ")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Panel 4: Gyroscope magnitude
        ax = axes[3]
        ax.plot(t, gyro_mag, color="tab:purple", linewidth=1.2)
        ax.set_ylabel("|ω| (rad/s)")
        ax.set_xlabel("Time (s)")
        ax.set_title("Gyro |ω|")
        ax.grid(True, alpha=0.3)

    # Optional rep/segment boundary annotations
    if rep_boundaries is not None:
        if not isinstance(axes, (list, np.ndarray)):
            axes_iter = [axes]
        else:
            axes_iter = axes
        for rb in rep_boundaries:
            for ax in axes_iter:
                ax.axvline(rb, color="k", linestyle="--", linewidth=0.8, alpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save to file if requested
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # Encode to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")

    plt.close(fig)
    return b64
