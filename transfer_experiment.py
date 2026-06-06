"""
transfer_experiment.py
======================
Transfer Learning Experiment: STL-10 with ImageNet-pretrained ResNet-18

Why STL-10?
  - 96x96 images -> layer4 outputs 3x3 feature maps (not 1x1 like CIFAR-10 at 32x32)
  - Only 5 000 labeled training samples -> small-data regime where TL shines
  - Explicitly designed as a Transfer-Learning benchmark dataset

Three conditions compared:
  1. Scratch          - ResNet-18 random init, full training
  2. Feature Extract  - ImageNet pretrained, backbone frozen, only FC head trained
  3. Fine-tuning      - ImageNet pretrained, differential LR (backbone 10x lower)

Spatial resolution trace for 96x96 input through standard ResNet-18:
  conv1  (7x7 s2)  : 96x96 -> 48x48
  maxpool(3x3 s2)  : 48x48 -> 24x24
  layer1 (s1)      : 24x24
  layer2 (s2)      : 12x12
  layer3 (s2)      :  6x6
  layer4 (s2)      :  3x3   <- sufficient spatial context for pretrained features
  avgpool + fc     :  512->10
"""

import copy, json, time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE      = torch.device('mps'  if torch.backends.mps.is_available()
              else         'cuda' if torch.cuda.is_available()
              else         'cpu')
DATA_DIR    = './data'
NUM_CLASSES = 10
BATCH_SIZE  = 64
EPOCHS      = 30
NUM_WORKERS = 0   # macOS MPS: spawn multiprocessing requires __main__ guard; 0 is safe

# STL-10 per-channel mean/std (computed on training split)
STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD  = [0.2603, 0.2565, 0.2712]

print(f"[Config] device={DEVICE}  batch={BATCH_SIZE}  epochs={EPOCHS}")
print("[Config] dataset=STL-10 (96x96, 5000 train / 8000 test, 10 classes)")

# ── Data loaders ───────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(96, padding=12),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(STL10_MEAN, STL10_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(STL10_MEAN, STL10_STD),
])

print("[Data] Downloading / loading STL-10 ...")
train_set = torchvision.datasets.STL10(
    root=DATA_DIR, split='train', download=True, transform=train_transform)
test_set  = torchvision.datasets.STL10(
    root=DATA_DIR, split='test',  download=True, transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=False)
test_loader  = torch.utils.data.DataLoader(
    test_set,  batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=False)

print(f"[Data] Train: {len(train_set)}  |  Test: {len(test_set)}")

# ── Model factory ──────────────────────────────────────────────────────────────
def build_model(mode: str) -> nn.Module:
    """
    Build ResNet-18 for the given training mode.

    The ORIGINAL ResNet-18 architecture is used unchanged (7x7 conv1 + maxpool).
    This is correct for STL-10 at 96x96 because layer4 still produces 3x3 feature
    maps -- enough spatial context for ImageNet pretrained features to transfer well.

    Parameters
    ----------
    mode : 'scratch' | 'feature_extract' | 'finetune'
    """
    if mode == 'scratch':
        # Random initialisation -- learn everything from STL-10 labels only
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, NUM_CLASSES)

    elif mode == 'feature_extract':
        # Pretrained backbone, FROZEN -- only the new classification head is updated
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(512, NUM_CLASSES)
        nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(model.fc.bias)
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False   # freeze backbone

    elif mode == 'finetune':
        # Pretrained backbone, FULLY TRAINABLE with differential LR
        # (backbone LR = 1/10 of head LR to avoid catastrophic forgetting)
        model = models.resnet18(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(512, NUM_CLASSES)
        nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
        nn.init.zeros_(model.fc.bias)
        # All params remain requires_grad=True -- differential LR set in run_condition

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return model


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ── Training utilities ─────────────────────────────────────────────────────────
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


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        n          += x.size(0)
    return total_loss / n, correct / n

# ── Experiment runner ──────────────────────────────────────────────────────────
LABEL = {
    'scratch':         'From Scratch',
    'feature_extract': 'Feature Extraction',
    'finetune':        'Fine-tuning',
}


def run_condition(mode: str) -> dict:
    print(f"\n{'─'*60}")
    print(f"  Condition: {LABEL[mode]}")
    print(f"{'─'*60}")

    model = build_model(mode).to(DEVICE)
    total, trainable = count_params(model)
    print(f"  Params: {total:,} total  |  {trainable:,} trainable "
          f"({trainable/total*100:.1f}%)")

    criterion = nn.CrossEntropyLoss()

    # Optimizer + scheduler tuned per condition
    if mode == 'scratch':
        # SGD + cosine annealing -- standard recipe for training from scratch
        optimizer = optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-4)

    elif mode == 'feature_extract':
        # Adam with moderate LR -- backbone frozen, only FC updated
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5)

    elif mode == 'finetune':
        # Differential LR:
        #   backbone_lr = 1e-3  (10x lower) -- preserve pretrained features
        #   head_lr     = 1e-2              -- adapt new classifier fast
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
        head_params     = [p for n, p in model.named_parameters() if 'fc'     in n]
        optimizer = optim.SGD([
            {'params': backbone_params, 'lr': 1e-3},
            {'params': head_params,     'lr': 1e-2},
        ], momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Training loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc   = 0.0
    best_state = None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc = evaluate(model, test_loader, criterion)
        scheduler.step()

        history['train_loss'].append(round(tr_loss, 6))
        history['train_acc'].append(round(tr_acc,  4))
        history['val_loss'].append(round(va_loss,  6))
        history['val_acc'].append(round(va_acc,   4))

        if va_acc > best_acc:
            best_acc   = va_acc
            best_state = copy.deepcopy(model.state_dict())

        elapsed = time.time() - t0
        eta     = elapsed / ep * (EPOCHS - ep)
        print(f"  ep {ep:2d}/{EPOCHS}  "
              f"train={tr_acc:.4f}  val={va_acc:.4f}  "
              f"best={best_acc:.4f}  ETA={eta:.0f}s")

    total_time = time.time() - t0
    print(f"  Done. Best val accuracy: {best_acc*100:.2f}%  "
          f"({total_time:.0f}s / {total_time/60:.1f} min)")

    return {
        'mode':             mode,
        'label':            LABEL[mode],
        'params_total':     total,
        'params_trainable': trainable,
        'best_acc':         round(best_acc, 4),
        'history':          history,
    }

# ── Run all three conditions ───────────────────────────────────────────────────
results = {}

for cond in ('scratch', 'feature_extract', 'finetune'):
    results[cond] = run_condition(cond)

with open('results_tl.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\n[Saved] results_tl.json')

# ── Summary ────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('SUMMARY -- Transfer Learning on STL-10 (ResNet-18)')
print('='*60)
scratch_acc = results['scratch']['best_acc']
for mode, r in results.items():
    delta = r['best_acc'] - scratch_acc
    sign  = '+' if delta >= 0 else ''
    print(f"  {r['label']:25s}: {r['best_acc']*100:.2f}%  "
          f"({sign}{delta*100:.2f} pp vs Scratch)")

# ── Plots ──────────────────────────────────────────────────────────────────────
COLORS = {
    'scratch':         '#e74c3c',
    'feature_extract': '#f39c12',
    'finetune':        '#2ecc71',
}
eps = list(range(1, EPOCHS + 1))

# Plot 1: Validation accuracy + training loss side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    'Transfer Learning on STL-10  |  ResNet-18 (ImageNet pretrained)',
    fontsize=13, fontweight='bold')

for mode, r in results.items():
    ax1.plot(eps, r['history']['val_acc'],
             color=COLORS[mode], label=r['label'], linewidth=2)
ax1.set_title('Validation Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 1.0)

for mode, r in results.items():
    ax2.plot(eps, r['history']['train_loss'],
             color=COLORS[mode], label=r['label'], linewidth=2)
ax2.set_title('Training Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Cross-Entropy Loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('tl_learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print('[Saved] tl_learning_curves.png')

# Plot 2: Bar chart -- final best accuracy
fig, ax = plt.subplots(figsize=(8, 5))
bar_labels = [r['label']        for r in results.values()]
bar_accs   = [r['best_acc']*100 for r in results.values()]
bar_colors = [COLORS[m]         for m in results]

bars = ax.bar(bar_labels, bar_accs, color=bar_colors,
              width=0.5, edgecolor='black', linewidth=0.8)
for bar, acc in zip(bars, bar_accs):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.4,
            f'{acc:.2f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=11)

ax.axhline(y=scratch_acc * 100, color=COLORS['scratch'],
           linestyle='--', alpha=0.6, linewidth=1.5, label='Scratch baseline')
ax.set_ylim(0, 105)
ax.set_ylabel('Test Accuracy (%)')
ax.set_title(
    'Transfer Learning Strategy Comparison\n'
    'STL-10  |  ResNet-18  |  30 epochs',
    fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('tl_comparison_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print('[Saved] tl_comparison_bar.png')

# Plot 3: Early convergence -- first 10 epochs
fig, ax = plt.subplots(figsize=(10, 5))
for mode, r in results.items():
    ax.plot(eps[:10], r['history']['val_acc'][:10],
            color=COLORS[mode], label=r['label'],
            linewidth=2, marker='o', markersize=5)
ax.set_title(
    'Convergence Speed -- First 10 Epochs\n'
    'Pretrained models start high; scratch starts low (key TL benefit)',
    fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('tl_early_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print('[Saved] tl_early_convergence.png')

# Plot 4: Train vs val per condition (overfitting check)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Training vs Validation Accuracy per Condition', fontweight='bold')
for ax, (mode, r) in zip(axes, results.items()):
    ax.plot(eps, r['history']['train_acc'], label='Train', linewidth=2, linestyle='--')
    ax.plot(eps, r['history']['val_acc'],   label='Val',   linewidth=2)
    ax.set_title(r['label'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.text(0.5, 0.05, f"Best val: {r['best_acc']*100:.2f}%",
            transform=ax.transAxes, ha='center', fontsize=10, color='navy')

plt.tight_layout()
plt.savefig('tl_overfit_check.png', dpi=150, bbox_inches='tight')
plt.close()
print('[Saved] tl_overfit_check.png')

print('\n[Done] Transfer learning experiment complete.')
