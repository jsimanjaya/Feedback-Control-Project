import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

DEFAULTS = {
    "lp_cut_hz": 5.0, # cutoff frequency
    "lp_order": 4, # 4-th order Butterworth filter
    "kp": 1.0, # propotional gain Kp
    "kd": 0.5, # derivative gain Kd
    "baseline_s": 2.0, 
    "target_duty": 0.25, # target duty cycle
    "hyst_frac": 0.30, # hysteresis fraction must be [0,1) 
}
TACTORS = ("AP_Positive", "AP_Negative", "ML_Positive", "ML_Negative")

def load_angles(path):
    """Load raw one-segment workbook angles and validate their time base."""
    if path.suffix.lower() == ".xlsx":
        data = pd.read_excel(path)
        ap_column, ml_column = "Trunk_Flex_Ext_deg", "Trunk_LatBend_deg"
    else:
        raise ValueError("Input must be a one-segment .xlsx workbook.")

    time_s = data["Time_s"].to_numpy(dtype=float)
    ap_raw = data[ap_column].to_numpy(dtype=float)
    ml_raw = data[ml_column].to_numpy(dtype=float)

    dt = time_s[1] - time_s[0] # change in time
    fs = 1/ dt # sampling frequency

    return time_s, ap_raw, ml_raw, fs, {"AP": ap_column, "ML": ml_column}

def hysteresis(u, pos_on, neg_on, n0, hyst_frac):
    """Convert a signed signal into −1, 0, +1 with hysteresis"""
    pos_off = pos_on * (1 - hyst_frac)
    neg_off = neg_on * (1 - hyst_frac)

    state = np.zeros(len(u), dtype=int)

    for i in range(n0, len(u)):
        previous = state[i - 1]
        value = u[i]

        if previous == 1:
            if value >= pos_off:
                state[i] = 1
            elif value <= neg_on:
                state[i] = -1

        elif previous == -1:
            if value <= neg_off:
                state[i] = -1
            elif value >= pos_on:
                state[i] = 1

        else:
            if value >= pos_on:
                state[i] = 1
            elif value <= neg_on:
                state[i] = -1

    return state

def simulate(time_s, ap_raw, ml_raw, fs, cfg):
    n0 = int(round(cfg["baseline_s"] * fs)) # how many beginning samples make up the neutral baseline period
    sos = butter( 
        cfg["lp_order"],
        cfg["lp_cut_hz"], 
        fs=fs, 
        btype="low",
        output="sos"
    )
    
    """ Subtracts  the average neutral angle from all samples then smooths the signal """
    ap_angle = sosfilt(sos, ap_raw - np.mean(ap_raw[:n0])) 
    ml_angle = sosfilt(sos, ml_raw - np.mean(ml_raw[:n0]))

    ap_vel = np.diff(ap_angle, prepend=ap_angle[0]) * fs
    ml_vel = np.diff(ml_angle, prepend=ml_angle[0]) * fs

    """ PD controller """
    ap_u = cfg["kp"] * ap_angle + cfg["kd"] * ap_vel
    ml_u = cfg["kp"] * ml_angle + cfg["kd"] * ml_vel

    """ Derive one positive and one negative threshold per axis """
    q = 100 * (1 - cfg["target_duty"]) # use 75th percentile

    ap_posterior = ap_u[n0:]
    ml_posterior = ml_u[n0:]

    """
    AP positive: flexion -> +1
    AP negative: extension -> -1
    ML positive: right bend -> +1
    ML negative: left bend -> -1
    """
    ap_pos_threshold = np.percentile(ap_posterior[ap_posterior > 0], q)
    ap_neg_threshold = -np.percentile(-ap_posterior[ap_posterior < 0], q) # selects negative values -> calculates the percentile -> convert to negative

    ml_pos_threshold = np.percentile(ml_posterior[ml_posterior > 0], q)
    ml_neg_threshold = -np.percentile(-ml_posterior[ml_posterior < 0], q)

    # Output arrays
    ap_state = hysteresis(
        ap_u,
        ap_pos_threshold,
        ap_neg_threshold,
        n0,
        cfg["hyst_frac"],
    )

    ml_state = hysteresis(
        ml_u,
        ml_pos_threshold,
        ml_neg_threshold,
        n0,
        cfg["hyst_frac"],
    )


    # Keep baseline output off
    ap_state[:n0] = 0
    ml_state[:n0] = 0

    """
    Four independent directional output channels:
        positive AP -> front tactor pairs assigned as tactor[0]
        negative AP -> back tactor pairs assigned as tactor[1]
        positive ML -> right tactor pairs assigned as tactor[2]
        negative ML -> left tactor pairs assigned as tactor[3]
    """
    tactors = np.zeros((len(time_s), 4), dtype=int)
    tactors[ap_state == 1, 0] = 1   # positive AP / front
    tactors[ap_state == -1, 1] = 1  # negative AP / back 
    tactors[ml_state == 1, 2] = 1   # positive ML / right
    tactors[ml_state == -1, 3] = 1  # negative ML / left 

    return {
        "baseline_count": n0, 
        "ap_angle": ap_angle, 
        "ml_angle": ml_angle,
        "ap_velocity": ap_vel, 
        "ml_velocity": ml_vel, 
        "ap_control": ap_u,
        "ml_control": ml_u, 
        "ap_state": ap_state,
        "ml_state": ml_state,
        "ap_pos_threshold": ap_pos_threshold,
        "ap_neg_threshold": ap_neg_threshold,
        "ml_pos_threshold": ml_pos_threshold,
        "ml_neg_threshold": ml_neg_threshold,
        "tactors": tactors,
    }


def save_results(time_s, ap_raw, ml_raw, result, cfg, output_dir, label, input_mapping):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / label
    table = np.column_stack((time_s, ap_raw, ml_raw, result["ap_angle"], result["ml_angle"],
                             result["ap_velocity"], result["ml_velocity"], result["ap_control"],
                             result["ml_control"], result["tactors"]))
    np.savetxt(f"{stem}_channels.csv", table, delimiter=",", fmt="%.6f",
               header=("Time_s,AP_Raw_deg,ML_Raw_deg,AP_Filtered_deg,ML_Filtered_deg,"
                       "AP_Velocity_deg_s,ML_Velocity_deg_s,AP_PD_u,ML_PD_u,PD_Magnitude,"
                       "Front_Left,Front_Right,Back_Left,Back_Right"), comments="")

    # Four signal panels plus one separate, time-aligned panel per corner
    # tactor.  Separate panels make activation overlap and chatter visible.
    fig, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    baseline_end = time_s[result["baseline_count"] - 1]
    axes[0].plot(time_s, result["ap_angle"], label="AP: flexion/extension", color="#005A9C")
    axes[0].plot(time_s, result["ml_angle"], label="ML: lateral bending", color="#D55E00")
    axes[0].axvspan(0, baseline_end, color="0.93", label="Neutral baseline")
    axes[0].set(ylabel="Filtered angle (deg)")
    axes[0].legend(loc="upper right", ncol=3, frameon=False)

    axes[1].plot(time_s, result["ap_control"], label="AP PD", color="#0072B2")
    axes[1].plot(time_s, result["ml_control"], label="ML PD", color="#E69F00")
    axes[1].axhline(0, color="0.5", linewidth=0.7)
    axes[1].set(ylabel="Signed PD control")
    axes[1].legend(loc="upper right", frameon=False)

    axes[1].axhline(result["ap_pos_threshold"], color="#0072B2",
                linestyle="--", label="AP positive threshold")
    axes[1].axhline(result["ap_neg_threshold"], color="#0072B2",
                    linestyle=":", label="AP negative threshold")

    axes[1].axhline(result["ml_pos_threshold"], color="#E69F00",
                    linestyle="--", label="ML positive threshold")
    axes[1].axhline(result["ml_neg_threshold"], color="#E69F00",
                    linestyle=":", label="ML negative threshold")

    colors = ("#009E73", "#CC79A7", "#56B4E9", "#E69F00")
    for axis, channel, tactor, color in zip(
        axes[2:], result["tactors"].T, TACTORS, colors
    ):
        axis.step(time_s, channel, where="post", color=color, linewidth=1.3)
        axis.set(yticks=[0, 1], ylim=(-0.12, 1.12),
                ylabel=tactor.replace("_", "\n"))
        axis.grid(axis="x", alpha=0.25)
    axes[-1].set(xlabel="Time (s)")
    fig.suptitle(f"Four-direction vibrotactile feedback | +AP = front, −AP = back | +ML = right, −ML = left")
    fig.savefig(f"{stem}.png", dpi=220, facecolor="white")
    fig.savefig(f"{stem}.svg", facecolor="white")
    plt.close(fig)

    active = result["tactors"][result["baseline_count"]:]
    summary = {
        "input_mapping": input_mapping,
        "mapping": {
            "positive_AP": "flexion -> Front_Left + Front_Right",
            "negative_AP": "extension -> Back_Left + Back_Right",
            "positive_ML": "right bend -> Front_Right + Back_Right",
            "negative_ML": "left bend -> Front_Left + Back_Left",
        },
        "controller": cfg,
        "ap_pos_threshold": float(result["ap_pos_threshold"]),
        "ap_neg_threshold": float(result["ap_neg_threshold"]),
        "ml_pos_threshold": float(result["ml_pos_threshold"]),
        "ml_neg_threshold": float(result["ml_neg_threshold"]),
        "post_baseline_duty_pct": {name: float(100 * active[:, i].mean())
                                    for i, name in enumerate(TACTORS)},
        "total_active_cue_pct": float(100 * np.any(active, axis=1).mean()),
    }
    with open(f"{stem}_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary

def main():
    """Command line argument parser for fine tuning"""
    parser = argparse.ArgumentParser(description="Simulate a four-direction trunk-feedback belt.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/four_direction_feedback"))
    parser.add_argument("--kp", type=float, default=DEFAULTS["kp"])
    parser.add_argument("--kd", type=float, default=DEFAULTS["kd"])
    parser.add_argument("--lp-cut-hz", type=float, default=DEFAULTS["lp_cut_hz"])
    parser.add_argument("--target-duty", type=float, default=DEFAULTS["target_duty"])
    parser.add_argument("--hyst-frac", type=float, default=DEFAULTS["hyst_frac"])

    args = parser.parse_args()
    if not 0 <= args.hyst_frac < 1:
        raise ValueError("hyst-frac must be in [0, 1).")
    if not 0 < args.target_duty <= 1:
        raise ValueError("target-duty must be in (0, 1].")
    time_s, ap_raw, ml_raw, fs, input_mapping = load_angles(args.input)
    cfg = {**DEFAULTS, "kp": args.kp, "kd": args.kd, "lp_cut_hz": args.lp_cut_hz,
           "target_duty": args.target_duty,
           "hyst_frac": args.hyst_frac}

    result = simulate(time_s, ap_raw, ml_raw, fs, cfg)
    summary = save_results(time_s, ap_raw, ml_raw, result, cfg, args.output_dir,
                           args.input.stem, input_mapping)
    print(
        f"Sample rate: {fs:.1f} Hz | "
        f"AP thresholds: {summary['ap_neg_threshold']:.3f}, "
        f"{summary['ap_pos_threshold']:.3f} | "
        f"ML thresholds: {summary['ml_neg_threshold']:.3f}, "
        f"{summary['ml_pos_threshold']:.3f}"
    )
    print("Post-baseline duty (%): " + ", ".join(f"{key}={value:.1f}" for key, value in summary["post_baseline_duty_pct"].items()))
    print(f"Wrote four-direction figure, channel CSV, and summary to {args.output_dir}")

if __name__ == "__main__":
    main()
