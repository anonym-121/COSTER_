# COSTER

**CO**llision **S**napshot guided **T**im**E**-**R**eversed safety-critical scenario generation (**COSTER**)

Weights and additional assets will be released after acceptance.

## Setup

```bash
cd COSTER
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Waymo Data

Waymo Open Motion Dataset v1.1.0 is used. Download the Waymo Open Motion Dataset (WOMD) in **Scenario proto TFRecord**
format.

```bash
python -m scenarionet.convert_waymo \
    -d data/scenarionet/training \
    --raw_data_path data/waymo_motion/training_20s \
    --num_workers 32

python -m scenarionet.convert_waymo \
    -d data/scenarionet/validation \
    --raw_data_path data/waymo_motion/validation \
    --num_workers 32
```

Build StateEstim training PKLs from the ScenarioNet training database:

```bash
python -m state_estim.preprocess \
    --data_dirs data/scenarionet/training \
    --out_dir data/state_estim_train
```

## Train StateEstim

Preprocess ScenarioNet data into StateEstim-style PKLs:

```bash
python -m state_estim.preprocess \
    --data_dirs /path/to/scenarionet/training \
    --out_dir data/state_estim_train
```


Train the categorical StateEstim initialization model:

```bash
python -m state_estim.train \
    --data_dir data/state_estim_train \
    --out_dir state_estim/ckpt \
    --epochs 50 \
    --batch_size 16 \
    --device cuda
```

This writes `state_estim/ckpt/state_estim.ckpt`. The model predicts local-region
Bernoulli placement probabilities and categorical distributions for position,
velocity, heading, and size.

## Train CVAE

Train the reverse-time trajectory CVAE:

```bash
python -m cvae.train \
    --config cvae/config.py \
    --train /path/to/scenarionet/training \
    --test /path/to/scenarionet/validation \
    --ckpt cvae/ckpt/my_run \
    --device cuda:0
```

Optional refine vocabulary:

```bash
python -m refine.collect_transitions \
    --data_dirs /path/to/scenarionet/training \
    --output refine/data/transitions.npz

python -m refine.build_vocab \
    --transitions refine/data/transitions.npz \
    --output refine/data/vocab_t0_K384.npz \
    --vocab_size 384
```

## Run Inference

Run the full COSTER pipeline:

```bash
python state_estim/pipeline.py \
    --data_dir /path/to/state_estim_pkls \
    --n_scenarios 10 \
    --state_estim_ckpt state_estim/ckpt/state_estim.ckpt \
    --cvae_ckpt cvae/ckpt/my_run/ckpt-best \
    --refine_vocab refine/data/vocab_t0_K384.npz \
    --out_dir output/pipeline_run \
    --device cuda \
    --skip_plots \
    --save_meta
```
