# COSTER

Collision Scenario Generation via Reverse-Time Trajectory Denoising.

COSTER inserts a collision vehicle into a driving scenario, generates the
vehicle's pre-collision trajectory with a conditional VAE, and refines the
result with template snapping plus physics-based rejection.

## Pipeline

```text
Waymo / StateEstim-style scenario PKL
    -> velocity-entropy collision time selection
    -> Bernoulli-region collision snapshot placement
    -> CVAE reverse-time trajectory generation
    -> K-means representative selection on lateral/arc-length features
    -> refine + physics reject
    -> exported scenario dumps and optional plots
```

The end-to-end inference pipeline currently expects a StateEstim-style PKL
schema:

```python
{
    "all_agent": np.ndarray,      # (T, N, 9): x, y, vx, vy, heading, length, width, type, valid
    "lane": np.ndarray,           # (P, 4): x, y, type_code, lane_id
    "traffic_light": list,        # optional per-timestep traffic-light data
}
```

ScenarioNet data is used directly for CVAE training and refine vocabulary
construction. For end-to-end inference, ScenarioNet/Waymo data should be
preprocessed into the StateEstim-style schema above.

## Repository Layout

```text
COSTER/
├── state_estim/         # end-to-end inference and collision vehicle placement
│   ├── pipeline.py      # main entry point
│   ├── select_timestep.py
│   ├── ego_lane_overlap.py
│   ├── global_inference.py
│   ├── model/
│   └── utils/
├── cvae/                # reverse-time trajectory CVAE
├── refine/              # template snapping and physics rejection
├── requirements.txt
└── LICENSE
```

Internal paper experiments, cluster scripts, logs, and local outputs are
quarantined under `_archive/` and ignored by git.

## Installation

```bash
cd COSTER
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CVAE training uses ADV-BMT preprocessing utilities. Set `ADV_BMT_ROOT` before
training:

```bash
export ADV_BMT_ROOT=/path/to/Adv-BMT
```

Optional plotting with the Reverse Waymo visualizer is disabled by default. To
enable it, set:

```bash
export REVERSE_WAYMO_ROOT=/path/to/Reverse_waymo
```

## Required Assets

The repository does not ship large model weights or data. Prepare these paths
before running inference:

```text
state_estim/ckpt/v4bi_ep049.ckpt     # StateEstim initialization model
cvae/ckpt/ckpt-best                  # trained CVAE checkpoint
refine/data/vocab_t0_K384.npz        # optional refine vocabulary
your_data_dir/*.pkl                   # StateEstim-style scenario PKLs
```

You can also pass explicit paths with CLI flags:

```bash
python state_estim/pipeline.py \
    --data_dir /path/to/scenario_pkls \
    --state_estim_ckpt /path/to/v4bi_ep049.ckpt \
    --cvae_ckpt /path/to/ckpt-best \
    --refine_vocab /path/to/vocab_t0_K384.npz \
    --out_dir output/demo \
    --device cuda \
    --skip_plots
```

If `--refine_vocab` is missing, the pipeline still runs but skips template
snapping. If model checkpoints are missing, the pipeline exits with a clear
error instead of running an untrained model.

The collision timestep is selected by averaging StateEstim velocity categorical
distributions over local regions inside the target-centered ellipse and taking
the timestep with maximum entropy. At that timestep, the collision snapshot uses
the highest Bernoulli-probability local region; the final contact placement is
chosen by maximizing the predicted local position probability among target-face
contact candidates.

CVAE samples are summarized with lateral-displacement and arc-length features.
K-means is run in this feature space, and the nearest-to-centroid
representatives are refined with the motion-token vocabulary and physics checks.

## Train CVAE

```bash
python -m cvae.train \
    --config cvae/config.py \
    --train /path/to/scenarionet/training \
    --test /path/to/scenarionet/validation \
    --ckpt cvae/ckpt/my_run \
    --device cuda:0
```

The training dataset reads ScenarioNet scenarios and uses ADV-BMT preprocessing
functions, so `ADV_BMT_ROOT` must point to a valid ADV-BMT checkout.

## Build Refine Vocabulary

```bash
python -m refine.collect_transitions \
    --data_dirs /path/to/scenarionet/training \
    --output refine/data/transitions.npz

python -m refine.build_vocab \
    --transitions refine/data/transitions.npz \
    --output refine/data/vocab_t0_K384.npz \
    --vocab_size 384
```

The vocabulary is used to snap CVAE rollouts to realistic transition templates
before physics-based rejection.

## Run End-to-End Inference

```bash
python state_estim/pipeline.py \
    --data_dir /path/to/state_estim_pkls \
    --n_scenarios 10 \
    --max_selected 2 \
    --state_estim_ckpt state_estim/ckpt/v4bi_ep049.ckpt \
    --cvae_ckpt cvae/ckpt/ckpt-best \
    --refine_vocab refine/data/vocab_t0_K384.npz \
    --out_dir output/pipeline_run \
    --device cuda \
    --skip_plots \
    --save_meta
```

Outputs are written under `--out_dir`:

```text
dumps/              # scenarios with inserted collision vehicles
vec_maps/           # vector-map exports for optional visualization
meta/               # per-scenario metadata when --save_meta is set
```

Remove `--skip_plots` only when `REVERSE_WAYMO_ROOT` is configured.

## Citation

```bibtex
@article{coster2026,
    title={COSTER: Collision Scenario Generation via Reverse-Time Trajectory Denoising},
    author={COSTER authors},
    year={2026}
}
```
