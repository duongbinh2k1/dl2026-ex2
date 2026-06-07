"""
focal_experiment.py
===================
Focal Loss for Class-Imbalanced Learning on CIFAR-10

Problem: Standard Cross-Entropy ignores minority classes on imbalanced data,
         achieving high overall accuracy while failing on rare classes.

Solution (Lin et al., ICCV 2017 — Focal Loss):
    FL(pt) = -(1 - pt)^γ · log(pt)

    pt = P(correct class)   — model confidence on the ground-truth class
    γ  = focusing parameter
         γ=0 → standard CE  (no focusing)
         γ=1 → linear down-weighting of easy examples
         γ=2 → quadratic (original paper, strongest focusing)

Dataset: Long-tail CIFAR-10 (imbalance ratio r=100)
    Class 0 (airplane): 5000 samples   ← most frequent
    Class 9 (truck):      50 samples   ← 100× fewer
    Exponential decay: n_c = N_max · (1/r)^(c/(C-1))

Four conditions (same model, same optimiser, only loss differs):
    1. Cross-Entropy           — standard baseline, no balancing
    2. Weighted CE             — inverse-frequency class weights
    3. Focal Loss γ=1          — gentle focusing
    4. Focal Loss γ=2          — original paper recommendation

Key insight: Overall accuracy is misleading on imbalanced data.
             Macro-F1 (treats all classes equally) reveals true performance.
"""

import copy, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE          = torch.device('mps'  if torch.backends.mps.is_available()
                  else         'cuda' if torch.cuda.is_available()
                  else         'cpu')
DATA_DIR        = './data'
NUM_CLASSES     = 10
BATCH_SIZE      = 128
EPOCHS          = 30
N_MAX           = 5000    # samples for most-frequent class
IMBALANCE_RATIO = 100     # N_max / N_min

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_MEAN    = [0.4914, 0.4822, 0.4465]
CIFAR10_STD     = [0.2470, 0.2435, 0.2616]

print(f"[Config] device={DEVICE}  epochs={EPOCHS}  batch={BATCH_SIZE}")
print(f"[Config] imbalance_ratio={IMBALANCE_RATIO}  N_max={N_MAX}  N_min={N_MAX//IMBALANCE_RATIO}")

# ── Data ─────────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

print("[Data] Loading CIFAR-10 ...")
full_train = torchvision.datasets.CIFAR10(
    DATA_DIR, train=True,  download=True, transform=train_transform)
test_set   = torchvision.datasets.CIFAR10(
    DATA_DIR, train=False, download=True, transform=test_transform)

# ── Long-tail subset ──────────────────────────────────────────────────────────
def make_longtail_indices(targets, n_max, imbalance_ratio, seed=42):
    """
    Exponential decay distribution over classes.
    Class c gets: n_c = n_max * (1/r)^(c / (C-1))
      c=0  → n_max          (most frequent)
      c=C-1 → n_max / r     (rarest)
    """
    rng     = np.random.RandomState(seed)
    targets = np.array(targets)
    C       = NUM_CLASSES
    n_per_class = []
    for c in range(C):
        n_c = int(n_max * (1.0 / imbalance_ratio) ** (c / (C - 1)))
        n_per_class.append(max(n_c, 1))

    indices = []
    for c in range(C):
        cls_idx = np.where(targets == c)[0]
        rng.shuffle(cls_idx)
        indices.extend(cls_idx[:n_per_class[c]].tolist())
    return indices, n_per_class

train_indices, N_PER_CLASS = make_longtail_indices(
    full_train.targets, N_MAX, IMBALANCE_RATIO)
imbalanced_set = Subset(full_train, train_indices)

train_loader = DataLoader(imbalanced_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_set,       batch_size=BATCH_SIZE,
                          shuffle=False,  num_workers=0, pin_memory=False)

TOTAL_TRAIN = sum(N_PER_CLASS)
print(f"[Data] Train: {TOTAL_TRAIN} imbalanced  |  Test: {len(test_set)} balanced")
print(f"[Data] Per-class counts: {N_PER_CLASS}")

# ── Class weights for Weighted CE ─────────────────────────────────────────────
# Inverse-frequency weights, normalised so mean weight = 1
freq          = torch.tensor(N_PER_CLASS, dtype=torch.float)
class_weights = (freq.sum() / (NUM_CLASSES * freq)).to(DEVICE)
print(f"[Weights] {class_weights.cpu().numpy().round(2)}")

# ── Model ─────────────────────────────────────────────────────────────────────
class ConvNet(nn.Module):
    """
    3-block ConvNet for CIFAR-10 (32x32).
    ~667K parameters. Used identically across all 4 conditions.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 32 → 16
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 16 → 8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                  # 8  → 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# All conditions start from IDENTICAL initial weights (fair comparison)
torch.manual_seed(SEED)
INIT_STATE = copy.deepcopy(ConvNet().state_dict())
N_PARAMS   = sum(p.numel() for p in ConvNet().parameters())
print(f"[Model] ConvNet: {N_PARAMS:,} parameters")

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017):

        FL(pt) = -(1 - pt)^γ · log(pt)

    where pt = exp(-CE(x, y)) is the probability the model assigns to the
    ground-truth class.

    The modulating factor (1-pt)^γ:
      - pt ≈ 1  (easy, well-classified)  →  weight ≈ 0   (down-weighted)
      - pt ≈ 0  (hard, misclassified)    →  weight ≈ 1   (full weight)
    """
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_per_sample = F.cross_entropy(logits, targets, reduction='none')
        pt            = torch.exp(-ce_per_sample)           # P(correct class)
        focal_weight  = (1.0 - pt).pow(self.gamma)
        return (focal_weight * ce_per_sample).mean()

# ── Conditions ────────────────────────────────────────────────────────────────
CONDITIONS = [
    {'key': 'ce',
     'label': 'Cross-Entropy (Baseline)',
     'loss':  nn.CrossEntropyLoss(),
     'color': '#e74c3c'},
    {'key': 'weighted_ce',
     'label': 'Weighted CE',
     'loss':  nn.CrossEntropyLoss(weight=class_weights),
     'color': '#f39c12'},
    {'key': 'focal_g1',
     'label': 'Focal Loss  γ=1',
     'loss':  FocalLoss(gamma=1.0),
     'color': '#3498db'},
    {'key': 'focal_g2',
     'label': 'Focal Loss  γ=2',
     'loss':  FocalLoss(gamma=2.0),
     'color': '#2ecc71'},
]

# ── Metrics ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_all_metrics(model, loader):
    """
    Returns overall accuracy, per-class accuracy, macro-F1, confusion matrix.
    Evaluated on the BALANCED test set (1 000 samples per class).
    """
    model.eval()
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(1)
        for t, p in zip(y, preds):
            cm[t.item(), p.item()] += 1

    overall_acc   = cm.diagonal().sum().item() / cm.sum().item()
    per_class_acc = (cm.diagonal().float() / cm.sum(dim=1).float().clamp(min=1)).tolist()

    # Macro-F1 (manual — no sklearn dependency)
    f1_per_class = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2*prec*recall / (prec+recall) if (prec+recall) > 0 else 0.0
        f1_per_class.append(f1)
    macro_f1 = sum(f1_per_class) / NUM_CLASSES

    return overall_acc, per_class_acc, macro_f1, cm.tolist(), f1_per_class

# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        n          += x.size(0)
    return total_loss / n, correct / n

# ── Experiment runner ─────────────────────────────────────────────────────────
def run_condition(cond: dict) -> dict:
    print(f"\n{'─'*60}")
    print(f"  {cond['label']}")
    print(f"{'─'*60}")

    model = ConvNet().to(DEVICE)
    model.load_state_dict(copy.deepcopy(INIT_STATE))   # identical starting point

    optimizer = optim.SGD(model.parameters(), lr=0.05,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-4)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_overall': [], 'val_macro_f1': [],
    }
    t0 = time.time()

    best_macro_f1 = 0.0
    best_state    = None
    best_metrics  = None   # (overall, per_class_acc, macro_f1, cm, f1_per_class)

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, cond['loss'])
        scheduler.step()
        overall, per_class_acc, macro_f1, cm, f1_per_class = \
            compute_all_metrics(model, test_loader)

        history['train_loss'].append(round(tr_loss,   6))
        history['train_acc'].append(round(tr_acc,    4))
        history['val_overall'].append(round(overall, 4))
        history['val_macro_f1'].append(round(macro_f1, 4))

        # Track best epoch — save full state in memory (no retraining needed)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_state    = copy.deepcopy(model.state_dict())
            best_metrics  = (overall, per_class_acc, macro_f1, cm, f1_per_class)

        elapsed = time.time() - t0
        eta     = elapsed / ep * (EPOCHS - ep)
        print(f"  ep {ep:2d}/{EPOCHS}  train={tr_acc:.4f}  "
              f"overall={overall:.4f}  macro_f1={macro_f1:.4f}  ETA={eta:.0f}s")

    # Restore best weights and use best-epoch metrics (no retraining needed)
    model.load_state_dict(best_state)
    overall, per_class_acc, macro_f1, cm, f1_per_class = best_metrics
    total_time = time.time() - t0
    best_ep = int(np.argmax(history['val_macro_f1'])) + 1
    print(f"  Best ep={best_ep}  overall={overall*100:.2f}%  "
          f"macro_f1={macro_f1:.4f}  ({total_time:.0f}s)")

    return {
        'key':           cond['key'],
        'label':         cond['label'],
        'overall_acc':   round(overall,    4),
        'macro_f1':      round(macro_f1,   4),
        'per_class_acc': [round(a, 4) for a in per_class_acc],
        'per_class_f1':  [round(f, 4) for f in f1_per_class],
        'confusion_matrix': cm,
        'history':       history,
    }

# ── Run all conditions ────────────────────────────────────────────────────────
results = {}
for cond in CONDITIONS:
    results[cond['key']] = run_condition(cond)

# ── Save results ──────────────────────────────────────────────────────────────
config_out = {
    'n_max': N_MAX, 'imbalance_ratio': IMBALANCE_RATIO,
    'epochs': EPOCHS, 'batch_size': BATCH_SIZE,
    'n_per_class': N_PER_CLASS, 'total_train': TOTAL_TRAIN,
    'n_params': N_PARAMS, 'classes': CIFAR10_CLASSES,
}
with open('results_focal.json', 'w') as f:
    json.dump({'config': config_out, 'results': results}, f, indent=2)
print('\n[Saved] results_focal.json')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '='*65)
print('SUMMARY — Focal Loss on Long-tail CIFAR-10')
print('='*65)
print(f"{'Method':30s}  {'Overall Acc':>12}  {'Macro F1':>10}")
print('─'*55)
for key, r in results.items():
    print(f"  {r['label']:28s}  {r['overall_acc']*100:>10.2f}%  {r['macro_f1']:>10.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
COLORS = {c['key']: c['color'] for c in CONDITIONS}
LABELS = {c['key']: c['label'] for c in CONDITIONS}
eps    = list(range(1, EPOCHS + 1))

# ── Plot 1: Class distribution (imbalance visualisation) ─────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
x = range(NUM_CLASSES)
bars = ax.bar(x, N_PER_CLASS, color='#3498db', edgecolor='black', linewidth=0.7)
for bar, n in zip(bars, N_PER_CLASS):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(n), ha='center', va='bottom', fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha='right')
ax.set_ylabel('Number of Training Samples')
ax.set_title(f'Long-tail CIFAR-10 Training Distribution\n'
             f'Imbalance ratio = {IMBALANCE_RATIO}:1  '
             f'(most: {N_PER_CLASS[0]} samples, rarest: {N_PER_CLASS[-1]} samples)',
             fontweight='bold')
ax.set_ylim(0, N_PER_CLASS[0] * 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('focal_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('[Saved] focal_class_distribution.png')

# ── Plot 2: Focal weight curve (1-pt)^γ vs pt ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
pt_vals = np.linspace(0.01, 1.0, 200)
for gamma, color, ls in [(0, '#e74c3c', '--'), (1, '#3498db', '-'), (2, '#2ecc71', '-')]:
    w = (1 - pt_vals) ** gamma
    label = f'γ={gamma}' + (' (CE)' if gamma == 0 else '')
    ax.plot(pt_vals, w, color=color, linewidth=2, linestyle=ls, label=label)
ax.set_xlabel('pt  (probability of correct class)', fontsize=11)
ax.set_ylabel('Focal weight  (1 − pt)^γ', fontsize=11)
ax.set_title('Focal Weight as a Function of pt\n'
             'Easy examples (pt→1) receive near-zero weight; '
             'hard examples (pt→0) retain full weight',
             fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('focal_weight_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print('[Saved] focal_weight_curve.png')

# ── Plot 3: Training curves — Overall accuracy + Macro F1 ────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Focal Loss on Long-tail CIFAR-10  |  Convergence Curves',
             fontsize=13, fontweight='bold')

for key, r in results.items():
    ax1.plot(eps, r['history']['val_overall'],
             color=COLORS[key], label=LABELS[key], linewidth=2)
ax1.set_title('Overall Accuracy (biased toward majority classes)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Overall Accuracy')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 1)

for key, r in results.items():
    ax2.plot(eps, r['history']['val_macro_f1'],
             color=COLORS[key], label=LABELS[key], linewidth=2)
ax2.set_title('Macro F1 (equal weight per class — true metric)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Macro F1')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('focal_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('[Saved] focal_training_curves.png')

# ── Plot 4: Summary bar — Overall acc vs Macro F1 ────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Overall Accuracy vs Macro-F1 Score\n'
             'Why Overall Accuracy Misleads on Imbalanced Data',
             fontsize=12, fontweight='bold')

keys   = list(results.keys())
labs   = [results[k]['label'] for k in keys]
oaccs  = [results[k]['overall_acc'] * 100 for k in keys]
mf1s   = [results[k]['macro_f1']    * 100 for k in keys]
colors = [COLORS[k] for k in keys]

def _bar(ax, vals, title, ylabel):
    bars = ax.bar(range(len(labs)), vals, color=colors,
                  width=0.55, edgecolor='black', linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{v:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

_bar(ax1, oaccs, 'Overall Accuracy (%)',
     'Accuracy (all classes, test-set balanced)')
_bar(ax2, mf1s,  'Macro-F1 Score (%)',
     'Macro-F1 (equal per-class weight)')

plt.tight_layout()
plt.savefig('focal_comparison_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('[Saved] focal_comparison_bar.png')

# ── Plot 5: Per-class accuracy ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
width  = 0.2
x      = np.arange(NUM_CLASSES)
offset = np.linspace(-1.5*width, 1.5*width, len(CONDITIONS))

for i, (cond, off) in enumerate(zip(CONDITIONS, offset)):
    r    = results[cond['key']]
    accs = [a * 100 for a in r['per_class_acc']]
    ax.bar(x + off, accs, width=width,
           label=cond['label'], color=cond['color'],
           edgecolor='black', linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha='right')
ax.set_ylabel('Per-class Accuracy (%)')
ax.set_title('Per-class Accuracy by Loss Function\n'
             'Minority classes (right) benefit most from Focal Loss',
             fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 105)
ax.axvline(x=4.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
ax.text(2,  98, 'Majority  →', ha='center', fontsize=9, color='gray')
ax.text(7,  98, '←  Minority', ha='center', fontsize=9, color='gray')
plt.tight_layout()
plt.savefig('focal_per_class_acc.png', dpi=150, bbox_inches='tight')
plt.close()
print('[Saved] focal_per_class_acc.png')

# ── Plot 6: Confusion matrices (CE vs Focal γ=2) ─────────────────────────────
for key in ('ce', 'focal_g2'):
    r  = results[key]
    cm = torch.tensor(r['confusion_matrix']).float()
    cm_norm = cm / cm.sum(dim=1, keepdim=True).clamp(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {r["label"]}\n'
                 f'Overall Acc={r["overall_acc"]*100:.1f}%  '
                 f'Macro-F1={r["macro_f1"]*100:.1f}%',
                 fontweight='bold')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j].item()
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6.5,
                    color='white' if v > 0.5 else 'black')
    plt.tight_layout()
    fname = f'focal_cm_{key}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {fname}')

print('\n[Done] Focal loss experiment complete.')
