# COSTER

**CO**llision **S**napshot guided **T**im**E**-**R**eversed safety-critical scenario generation (**COSTER**)


## Setup

```bash
cd COSTER
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

CVAE preprocessing is performed online by `cvae/data.py`. It reads ScenarioNet
directly and builds the CVAE tensors for target state, neighboring agents, and
27D vector-map features inside this repository.

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
