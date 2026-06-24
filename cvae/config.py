"""
Default configuration for COSTER's CVAE trajectory generator.

Based on the best-performing VectorMapCVAE-SS (Single-State) model:
  - OB_HORIZON=0: single collision state, no extrapolation
  - Reverse-time autoregressive decoding with dynamic map cross-attention
  - Cosine LR schedule with warmup, KL warmup, and scheduled sampling
"""

# ── Horizons (OB_HORIZON=0: single collision state) ──
OB_HORIZON = 0
PRED_HORIZON = 10
NUM_SKIPPED_STEPS = 5       # 10Hz → 2Hz (dt = 0.5s)
DT = NUM_SKIPPED_STEPS * 0.1  # 0.5s

# ── Map settings ──
MAP_FEATURE_DIM = 27
MAX_VECTORS = 128
MAX_MAP_FEATURES = 256
MAX_TRAFFIC_LIGHTS = 64

# ── Agent settings ──
MAX_AGENTS = 64
MAX_NEIGHBORS = 32
OB_RADIUS = 50.0

# ── Model architecture ──
D_MODEL = 256
N_ATTENTION_HEADS = 8
NUM_FREQ_BANDS = 64

# ── Training ──
lr = 3e-4
weight_decay = 1e-4
epochs = 200
test_since = 10
batch_size = 64
val_batch_size = 64
num_workers = 12
pred_samples = 5
clustering = 0
kl_warmup_epochs = 30

# ── Stability & Convergence ──
max_grad_norm = 1.0
use_cosine_lr = True
lr_warmup_epochs = 5

# ── Scheduled Sampling ──
ss_start_epoch = 50
ss_max_ratio = 0.5
ss_ramp_epochs = 100

# ── Loss ──
time_weight_exp = 0.5

# ── Mixed Precision ──
use_amp = False

# ── Data ──
reverse_time = True
limit_map_range = True
augment = True
fraction = None

# ── Model kwargs (VectorMapCVAE constructor) ──
model = dict(
    horizon=PRED_HORIZON,
    d_model=D_MODEL,
    ob_radius=OB_RADIUS,
    map_feature_dim=MAP_FEATURE_DIM,
    max_vectors=MAX_VECTORS,
    n_attention_heads=N_ATTENTION_HEADS,
    num_freq_bands=NUM_FREQ_BANDS,
    dt=DT,
)

# ── Dataset kwargs ──
_common_dataset = dict(
    ob_horizon=OB_HORIZON,
    pred_horizon=PRED_HORIZON,
    num_skipped_steps=NUM_SKIPPED_STEPS,
    max_agents=MAX_AGENTS,
    max_map_features=MAX_MAP_FEATURES,
    max_vectors=MAX_VECTORS,
    max_traffic_lights=MAX_TRAFFIC_LIGHTS,
    ob_radius=OB_RADIUS,
    reverse_time=reverse_time,
    max_neighbors=MAX_NEIGHBORS,
    limit_map_range=limit_map_range,
)

train_dataset = dict(**_common_dataset, augment=True, split="train", split_ratio=0.8)
test_dataset = dict(**_common_dataset, augment=False, split="val", split_ratio=0.8)
