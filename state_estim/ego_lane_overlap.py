"""Target-ellipse region visualization helpers.

This module is intentionally limited to geometry, StateEstim inference lookup
flattening, and lightweight visualization.  Collision snapshot placement is
implemented in ``pipeline.py`` so this helper does not contain standalone
vehicle-placement heuristics.
"""

import argparse
import glob
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import scale as shapely_scale
from shapely.affinity import translate as shapely_translate
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.scenario_loader import load_scenario


CENTER_LANE_TYPES = {1, 2, 3}
LANE_VEC_HALF_WIDTH = 2.5
EGO_SEMI_LONG = 7.0
EGO_SEMI_LAT = 3.5
VIEW_RANGE = 30


def _rotate_2d(x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return c * x - s * y, s * x + c * y


def _lane_vec_polygon(x1, y1, x2, y2, half_w=LANE_VEC_HALF_WIDTH):
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return None
    px, py = -dy / length, dx / length
    return Polygon([
        (x1 + px * half_w, y1 + py * half_w),
        (x2 + px * half_w, y2 + py * half_w),
        (x2 - px * half_w, y2 - py * half_w),
        (x1 - px * half_w, y1 - py * half_w),
    ])


def _agent_polygon(x, y, heading, length, width):
    half_l, half_w = length / 2.0, width / 2.0
    corners = []
    for lx, ly in [(half_l, half_w), (half_l, -half_w),
                   (-half_l, -half_w), (-half_l, half_w)]:
        rx, ry = _rotate_2d(np.array(lx), np.array(ly), heading)
        corners.append((float(x + rx), float(y + ry)))
    return Polygon(corners)


def _target_ellipse(x, y, heading):
    unit = Point(0, 0).buffer(1.0, resolution=64)
    scaled = shapely_scale(unit, xfact=EGO_SEMI_LONG, yfact=EGO_SEMI_LAT)
    rotated = shapely_rotate(scaled, np.degrees(float(heading)))
    return shapely_translate(rotated, float(x), float(y))


def extract_center_vectors(lane_raw):
    """Return consecutive center-lane vector segments from ``lane_raw``."""
    types = lane_raw[:, 2].astype(int)
    center_pts = lane_raw[np.isin(types, list(CENTER_LANE_TYPES))]
    if len(center_pts) == 0:
        return np.empty((0, 4))

    vectors = []
    for lane_id in np.unique(center_pts[:, 3]):
        pts = center_pts[center_pts[:, 3] == lane_id][:, :2]
        for i in range(len(pts) - 1):
            vectors.append([pts[i, 0], pts[i, 1], pts[i + 1, 0], pts[i + 1, 1]])
    return np.array(vectors) if vectors else np.empty((0, 4))


def find_overlapping_mask(vectors, ego_shape):
    """Return lane-vector regions intersecting the target ellipse."""
    mask = np.zeros(len(vectors), dtype=bool)
    for i, (x1, y1, x2, y2) in enumerate(vectors):
        rect = _lane_vec_polygon(x1, y1, x2, y2)
        if rect is not None and ego_shape.intersects(rect):
            mask[i] = True
    return mask


def build_inference_lookup(results):
    """Merge per-anchor StateEstim outputs into flat position/state arrays."""
    if not results:
        return None, None

    keys = [
        'prob', 'speed', 'speed_probs', 'speed_centers', 'vel_heading',
        'pos_mean', 'pos_long_probs', 'pos_lat_probs', 'pos_centers',
        'pos_logits', 'pos_loc', 'pos_cov', 'vec_global',
        'bbox_mean', 'length_probs', 'length_centers',
        'width_probs', 'width_centers',
        'heading_mean', 'heading_probs', 'heading_centers',
        'lane_directions',
    ]
    all_pos, all_states = [], {k: [] for k in keys}

    for result in results:
        if len(result['global_positions']) == 0:
            continue
        all_pos.append(result['global_positions'])
        for key in keys:
            if key in result:
                all_states[key].append(result[key])

    if not all_pos:
        return None, None

    positions = np.concatenate(all_pos, axis=0)
    for key in list(all_states):
        if all_states[key]:
            all_states[key] = np.concatenate(all_states[key], axis=0)
        else:
            all_states.pop(key)
    return positions, all_states


def _draw_agent(ax, agent, facecolor, edgecolor, alpha, zorder):
    if len(agent) < 9 or agent[8] < 0.5 or int(agent[7]) != 1:
        return
    poly = _agent_polygon(float(agent[0]), float(agent[1]), float(agent[4]),
                          float(agent[5]), float(agent[6]))
    ax.add_patch(MplPolygon(
        np.array(poly.exterior.coords[:-1]), closed=True,
        facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
        linewidth=1.0, zorder=zorder,
    ))
    arrow_len = max(float(agent[5]), 4.0)
    ax.annotate(
        "",
        xy=(float(agent[0]) + np.cos(float(agent[4])) * arrow_len,
            float(agent[1]) + np.sin(float(agent[4])) * arrow_len),
        xytext=(float(agent[0]), float(agent[1])),
        arrowprops=dict(arrowstyle="-|>", color=edgecolor, lw=1.1),
        zorder=zorder + 1,
    )


def _draw_inference_regions(ax, infer_lookup, target_ellipse, max_labels=12):
    if infer_lookup is None or infer_lookup[0] is None:
        return
    positions, states = infer_lookup
    if 'prob' not in states:
        return

    inside = np.array([
        target_ellipse.contains(Point(float(x), float(y)))
        for x, y in positions
    ])
    if not inside.any():
        return

    idx = np.where(inside)[0]
    probs = states['prob'][idx]
    order = idx[np.argsort(probs)[::-1]]

    ax.scatter(
        positions[idx, 0], positions[idx, 1],
        c=probs, cmap='magma', s=18, alpha=0.75,
        edgecolors='none', zorder=8,
    )

    best = int(order[0])
    ax.scatter(
        [positions[best, 0]], [positions[best, 1]],
        marker='*', s=180, c='cyan', edgecolors='black',
        linewidths=0.8, zorder=10,
    )
    ax.text(
        positions[best, 0], positions[best, 1],
        f" max Bern\np={states['prob'][best]:.2f}",
        fontsize=7, color='black', zorder=11,
        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='black', alpha=0.8),
    )

    for region_idx in order[1:max_labels]:
        ax.text(
            positions[region_idx, 0], positions[region_idx, 1],
            f"{states['prob'][region_idx]:.2f}",
            fontsize=5, color='black', ha='center', va='center', zorder=9,
        )


def draw_scenario(ax, lane_raw, agents_t0, title="", infer_lookup=None):
    """Draw target ellipse, intersecting local regions, and StateEstim probabilities."""
    target = agents_t0[0]
    tx, ty = float(target[0]), float(target[1])
    heading = float(target[4])
    target_ellipse = _target_ellipse(tx, ty, heading)

    vectors = extract_center_vectors(lane_raw)
    overlap_mask = find_overlapping_mask(vectors, target_ellipse)

    for i, (x1, y1, x2, y2) in enumerate(vectors):
        color = 'red' if overlap_mask[i] else '#aaaaaa'
        width = 1.5 if overlap_mask[i] else 0.5
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=1)
        if overlap_mask[i]:
            rect = _lane_vec_polygon(x1, y1, x2, y2)
            if rect is not None:
                ax.add_patch(MplPolygon(
                    np.array(rect.exterior.coords[:-1]), closed=True,
                    facecolor='red', edgecolor='red', alpha=0.18,
                    linewidth=0.8, zorder=2,
                ))

    for i in range(1, len(agents_t0)):
        ag = agents_t0[i]
        if abs(float(ag[0]) - tx) > VIEW_RANGE * 2:
            continue
        if abs(float(ag[1]) - ty) > VIEW_RANGE * 2:
            continue
        _draw_agent(ax, ag, 'limegreen', 'green', 0.35, 4)

    _draw_agent(ax, target, 'royalblue', 'navy', 0.80, 6)
    ellipse_patch = MplPolygon(
        np.array(target_ellipse.exterior.coords[:-1]), closed=True,
        facecolor='none', edgecolor='red', alpha=0.9,
        linestyle='--', linewidth=1.2, zorder=7,
    )
    ax.add_patch(ellipse_patch)
    _draw_inference_regions(ax, infer_lookup, target_ellipse)

    ax.set_xlim(tx - VIEW_RANGE, tx + VIEW_RANGE)
    ax.set_ylim(ty - VIEW_RANGE, ty + VIEW_RANGE)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.legend(handles=[
        mpatches.Patch(facecolor='red', alpha=0.25,
                       edgecolor='red', label='Regions inside target ellipse'),
        mpatches.Patch(facecolor='royalblue', alpha=0.8,
                       edgecolor='navy', label='Target vehicle'),
        mpatches.Patch(facecolor='limegreen', alpha=0.4,
                       edgecolor='green', label='Other vehicles'),
    ], fontsize=7, loc='upper right', framealpha=0.85)


def print_inference_summary(scenario_name, agents_t0, infer_lookup):
    """Print top Bernoulli regions inside the target ellipse."""
    if infer_lookup is None or infer_lookup[0] is None:
        return
    positions, states = infer_lookup
    target = agents_t0[0]
    target_ellipse = _target_ellipse(float(target[0]), float(target[1]), float(target[4]))
    inside = np.array([
        target_ellipse.contains(Point(float(x), float(y)))
        for x, y in positions
    ])
    if not inside.any():
        print(f"\n{scenario_name}: no StateEstim regions inside target ellipse")
        return

    idx = np.where(inside)[0]
    order = idx[np.argsort(states['prob'][idx])[::-1]]
    print(f"\n{'=' * 70}")
    print(f"  Scenario: {scenario_name}   regions in target ellipse: {len(idx)}")
    print(f"  {'rank':>4}  {'prob':>6}  {'speed(m/s)':>10}  "
          f"{'heading(°)':>10}  {'x':>9}  {'y':>9}")
    print(f"  {'-' * 62}")
    for rank, region_idx in enumerate(order[:10], start=1):
        heading = states['lane_directions'][region_idx] + states['heading_mean'][region_idx]
        print(f"  {rank:>4}  {states['prob'][region_idx]:>6.3f}  "
              f"{states['speed'][region_idx]:>10.2f}  "
              f"{np.degrees(heading):>10.1f}  "
              f"{positions[region_idx, 0]:>9.2f}  {positions[region_idx, 1]:>9.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize target-ellipse regions and StateEstim probabilities')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing StateEstim-style scenario PKLs')
    parser.add_argument('--n_scenarios', type=int, default=5)
    parser.add_argument('--ckpt_path', default='ckpt/v4bi_ep049.ckpt')
    parser.add_argument('--output', default='target_region_probs.png')
    parser.add_argument('--no_inference', action='store_true',
                        help='show region overlaps only, without inference')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--timestep', default='mid',
                        help="inference/visualization timestep: int, 'mid', or 'last'")
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.data_dir, '*.pkl')))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in '{args.data_dir}'")
    pkl_files = pkl_files[:args.n_scenarios]

    engine = None
    if not args.no_inference:
        from global_inference import GlobalInitInference
        engine = GlobalInitInference(
            ckpt_path=args.ckpt_path,
            device=args.device,
            map_size=50,
            anchor_min_spacing=40,
        )

    fig, axes = plt.subplots(1, len(pkl_files), figsize=(6 * len(pkl_files), 6))
    if len(pkl_files) == 1:
        axes = [axes]

    for ax, pkl_path in zip(axes, pkl_files):
        scenario_name = os.path.splitext(os.path.basename(pkl_path))[0]
        if engine is not None:
            results, lane_raw, agents_t0 = engine.run(pkl_path, timestep=args.timestep)
            infer_lookup = build_inference_lookup(results)
        else:
            lane_raw, agents_t0, _ = load_scenario(pkl_path, timestep=args.timestep)
            infer_lookup = None

        print_inference_summary(scenario_name, agents_t0, infer_lookup)
        draw_scenario(
            ax, lane_raw, agents_t0,
            title=f"{scenario_name} | t={args.timestep}",
            infer_lookup=infer_lookup,
        )

    fig.suptitle(
        f"Target Ellipse ({EGO_SEMI_LONG}m x {EGO_SEMI_LAT}m) and StateEstim Regions",
        fontsize=10, fontweight='bold',
    )
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved -> {output_path}")


if __name__ == '__main__':
    main()
