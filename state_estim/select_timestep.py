"""Velocity-entropy collision timestep selection.

For each valid timestep, StateEstim predicts a categorical speed distribution
for local lane regions around the target vehicle.  We average the distributions
inside the target-centered ellipse and select the timestep with the largest
entropy.
"""

import argparse
import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ego_lane_overlap import EGO_SEMI_LAT, EGO_SEMI_LONG, build_inference_lookup, draw_scenario
from utils.scenario_loader import _ensure_pickle_compat, load_scenario


SKIP_FIRST_LAST = 5
EPS = 1e-12


def _load_full_scenario(pkl_path):
    _ensure_pickle_compat()
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _get_target_state_all_timesteps(data):
    """Return target states over time. The current release uses ego as target."""
    if "all_agent" in data:
        all_agent = np.asarray(data["all_agent"])
        return all_agent[:, 0, :], all_agent.shape[0]
    if "map_features" in data:
        meta = data.get("metadata", {})
        sdc_id = str(meta.get("sdc_id", "0"))
        st = data["tracks"][sdc_id]["state"]
        T = int(st["position"].shape[0])
        target = np.zeros([T, 9], dtype=np.float64)
        target[:, 0] = st["position"][:, 0]
        target[:, 1] = st["position"][:, 1]
        target[:, 2] = st["velocity"][:, 0]
        target[:, 3] = st["velocity"][:, 1]
        target[:, 4] = st["heading"][:T]
        target[:, 5] = st["length"][:T]
        target[:, 6] = st["width"][:T]
        target[:, 7] = 1
        target[:, 8] = st["valid"][:T]
        return target, T
    raise ValueError("Unknown PKL format")


def _target_ellipse(target_state):
    unit = Point(0, 0).buffer(1.0, resolution=64)
    scaled = shapely_scale(unit, xfact=EGO_SEMI_LONG, yfact=EGO_SEMI_LAT)
    rotated = shapely_rotate(scaled, np.degrees(float(target_state[4])))
    return shapely_translate(rotated, float(target_state[0]), float(target_state[1]))


def _entropy(prob):
    prob = np.asarray(prob, dtype=np.float64)
    prob = prob / max(float(prob.sum()), EPS)
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log(prob + EPS)))


def _velocity_entropy_for_timestep(pkl_path, timestep, target_state, engine):
    results, _, _ = engine.run(pkl_path, timestep=timestep)
    infer_lookup = build_inference_lookup(results)
    if infer_lookup is None or infer_lookup[0] is None:
        return 0.0, {
            "n_regions": 0,
            "mean_placement_prob": 0.0,
            "velocity_distribution": None,
        }

    region_xy, region_states = infer_lookup
    speed_probs = region_states.get("speed_probs")
    if speed_probs is None or len(speed_probs) == 0:
        return 0.0, {
            "n_regions": 0,
            "mean_placement_prob": 0.0,
            "velocity_distribution": None,
        }

    ellipse = _target_ellipse(target_state)
    inside = np.array([ellipse.contains(Point(float(x), float(y))) for x, y in region_xy])
    if not inside.any():
        return 0.0, {
            "n_regions": 0,
            "mean_placement_prob": 0.0,
            "velocity_distribution": None,
        }

    velocity_distribution = speed_probs[inside].mean(axis=0)
    velocity_distribution = velocity_distribution / max(float(velocity_distribution.sum()), EPS)
    placement_prob = region_states.get("prob", np.zeros(len(region_xy), dtype=np.float32))
    return _entropy(velocity_distribution), {
        "n_regions": int(inside.sum()),
        "mean_placement_prob": float(np.mean(placement_prob[inside])),
        "velocity_distribution": velocity_distribution,
    }


def velocity_entropy_all_timesteps(pkl_path, engine, min_t=0):
    """Compute velocity entropy for every valid timestep."""
    data = _load_full_scenario(pkl_path)
    target_states, T = _get_target_state_all_timesteps(data)
    scores = np.full(T, -1.0, dtype=np.float64)
    valid = np.zeros(T, dtype=bool)
    n_regions = np.zeros(T, dtype=np.int64)
    mean_placement_prob = np.zeros(T, dtype=np.float64)
    distributions = [None] * T

    for t in range(T):
        target = target_states[t]
        if target[8] < 0.5:
            continue
        if t < max(min_t, SKIP_FIRST_LAST) or t > T - 1 - SKIP_FIRST_LAST:
            continue
        try:
            H_t, info = _velocity_entropy_for_timestep(pkl_path, t, target, engine)
        except Exception as exc:
            print(f"  [entropy] timestep {t}: inference failed - {exc}")
            continue
        scores[t] = H_t
        valid[t] = info["n_regions"] > 0
        n_regions[t] = info["n_regions"]
        mean_placement_prob[t] = info["mean_placement_prob"]
        distributions[t] = info["velocity_distribution"]

    details = {
        "valid": valid,
        "velocity_entropy": np.where(scores >= 0, scores, 0.0),
        "n_regions": n_regions,
        "mean_placement_prob": mean_placement_prob,
        "velocity_distribution": distributions,
    }
    return scores, details, T


def select_best_timestep(pkl_path, engine, min_t=0):
    """Select collision time by maximum target-neighborhood velocity entropy."""
    if engine is None:
        raise ValueError("StateEstim inference engine is required for velocity-entropy selection.")

    print(f"\n{'=' * 60}")
    print(f"  select_best_timestep: {os.path.basename(pkl_path)}")
    print(f"{'=' * 60}")

    scores, details, _ = velocity_entropy_all_timesteps(pkl_path, engine, min_t=min_t)
    valid_idx = np.where(details["valid"])[0]
    if len(valid_idx) == 0:
        fallback = max(min_t, 0)
        print(f"  No valid entropy timestep found. Falling back to t={fallback}.")
        return fallback, scores, None, details

    best_t = int(valid_idx[np.argmax(scores[valid_idx])])
    print(f"  Selected t={best_t}: H={scores[best_t]:.4f}, "
          f"regions={details['n_regions'][best_t]}, "
          f"mean_prob={details['mean_placement_prob'][best_t]:.3f}")
    return best_t, scores, None, details


def plot_timestep_analysis(scores, details, best_t, output_path=None, scenario_name=""):
    valid = details["valid"]
    timesteps = np.arange(len(scores))
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(timesteps[valid], details["velocity_entropy"][valid],
            "-", linewidth=2, color="black", label="velocity entropy")
    ax.axvline(best_t, color="red", linewidth=2, label=f"selected t={best_t}")
    ax.scatter([best_t], [details["velocity_entropy"][best_t]],
               color="red", s=100, marker="*", zorder=10)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Velocity entropy")
    ax.set_title(f"Velocity-Entropy Collision Time Selection - {scenario_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {output_path}")
    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Select collision timestep by StateEstim velocity entropy")
    parser.add_argument("--data_dir", required=True,
                        help="Directory containing StateEstim-style scenario PKLs")
    parser.add_argument("--n_scenarios", type=int, default=4)
    parser.add_argument("--ckpt_path", default="ckpt/v4bi_ep049.ckpt")
    parser.add_argument("--output", default="velocity_entropy_selection.png")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min_t", type=int, default=0)
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in '{args.data_dir}'")

    from global_inference import GlobalInitInference
    engine = GlobalInitInference(
        ckpt_path=args.ckpt_path,
        device=args.device,
        map_size=50,
        anchor_min_spacing=40,
    )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for pkl_path in pkl_files[:args.n_scenarios]:
        scenario_name = os.path.splitext(os.path.basename(pkl_path))[0]
        best_t, scores, _, details = select_best_timestep(
            pkl_path, engine=engine, min_t=args.min_t)
        output_path = os.path.join(out_dir, f"{scenario_name}_{args.output}")
        plot_timestep_analysis(scores, details, best_t, output_path, scenario_name)

        lane_raw, agents_t0, _ = load_scenario(pkl_path, timestep=best_t)
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        draw_scenario(ax, lane_raw, agents_t0, title=f"Selected t={best_t}")
        scene_path = os.path.join(out_dir, f"{scenario_name}_selected_scene.png")
        fig.savefig(scene_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
