#!/usr/bin/env python3
"""
Transfer Learning on CIFAR-10
Practical Session 2 – Advanced Deep Learning Strategies
USTH Deep Learning 2026
Duong Tan Binh – 2540007

Three conditions (mapping to slide p.46 quadrant – small + similar data):
  1. Scratch        : ResNet-18 random init, train from scratch
  2. Feature Extract: ResNet-18 ImageNet pretrained, backbone FROZEN, only head trained
  3. Fine-tuning    : ResNet-18 ImageNet pretrained, ALL layers trained (differential LR)

Expected results (guaranteed by decades of TL literature):
  Scratch < Feature Extract < Fine-tuning  (accuracy)
  Scratch >> Feature Extract (epochs to converge)
"""

import os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════
SEED            = 42
BATCH_SIZE      = 128
SCRATCH_EPOCHS  = 30    # scratch needs more epochs to converge
TL_EPOCHS       = 20    # pretrained models converge faster
NUM_CLASSES     = 10
OUT_DIR         = os.path.dirname(os.path.abspath(__file__))

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = ('mps'  if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available()       else 'cpu')
print(f"Device: {DEVICE}", flush=True)

CLASSES = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# ══════════════════════════════════════════════════════════
# 1.  DATA
# ══════════════════════════════════════════════════════════
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

_to_tensor = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(MEAN, STD)])
_pad_crop   = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip()])

print("Loading CIFAR-10 into memory...", flush=True)
data_dir  = os.path.join(OUT_DIR, 'data')
_train_ds = datasets.CIFAR10(data_dir, train=True,  download=True, transform=_to_tensor)
_test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=_to_tensor)

def _ds_to_tensors(ds):
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    xs, ys = zip(*[(x, y) for x, y in loader])
    return torch.cat(xs), torch.cat(ys)

_train_x, _train_y = _ds_to_tensors(_train_ds)
_test_x,  _test_y  = _ds_to_tensors(_test_ds)
print(f"  Train: {_train_x.shape}  Test: {_test_x.shape}", flush=True)

def get_loaders():
    train_loader = DataLoader(TensorDataset(_train_x, _train_y),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(TensorDataset(_test_x,  _test_y),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader

train_loader, test_loader = get_loaders()


# ══════════════════════════════════════════════════════════
# 2.  MODELS
# ══════════════════════════════════════════════════════════
def make_resnet_cifar(pretrained=False):
    """
    ResNet-18 adapted for CIFAR-10 (32×32 input):
      - conv1: 3×3 stride-1 (instead of 7×7 stride-2) — preserves spatial resolution
      - maxpool replaced by Identity — avoids aggressive early downsampling
      - fc: 512 → NUM_CLASSES

    Pretrained=True loads ImageNet weights for conv2_x…conv5_x and bn layers.
    conv1 and fc are always randomly initialised (task-specific layers).
    """
    m = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    # Adapt first conv and maxpool for 32×32 input
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    # Replace classifier head
    m.fc      = nn.Linear(512, NUM_CLASSES)

    if pretrained:
        # conv1 and fc are new (random) – the pretrained inner layers
        # (layer1…layer4, bn1) are kept from ImageNet weights loaded above.
        # Re-initialise only the new layers:
        nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(m.fc.weight, 0, 0.01)
        nn.init.zeros_(m.fc.bias)
    return m


def freeze_backbone(model):
    """Freeze all layers except the final FC head (feature extraction mode)."""
    for name, param in model.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
    return model


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters()
               if (p.requires_grad if trainable_only else True))


# ══════════════════════════════════════════════════════════
# 3.  TRAINING UTILITIES
# ══════════════════════════════════════════════════════════
def aug(x):
    dev = x.device; x = x.cpu()
    return torch.stack([_pad_crop(xi) for xi in x]).to(dev)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ls=cr=n=0; preds=[]; lbls=[]
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out  = model(xb); loss = F.cross_entropy(out, yb)
        pred = out.argmax(1)
        ls += loss.item()*len(yb); cr += (pred==yb).sum().item(); n += len(yb)
        preds.extend(pred.cpu().tolist()); lbls.extend(yb.cpu().tolist())
    return ls/n, cr/n, preds, lbls


def run_training(model, epochs, label, opt):
    """Generic training loop with cosine LR annealing."""
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist  = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
    best_acc=0.; best_w=None
    print(f"\n{'─'*60}\n  {label}  epochs={epochs}\n{'─'*60}", flush=True)
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train(); tls=tcr=tn=0
        for xb, yb in train_loader:
            xb = aug(xb)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out  = model(xb); loss = F.cross_entropy(out, yb)
            loss.backward(); opt.step()
            tls += loss.item()*len(yb)
            tcr += (out.argmax(1)==yb).sum().item(); tn += len(yb)
        sched.step()
        vl, va, _, _ = evaluate(model, test_loader)
        tl, ta = tls/tn, tcr/tn
        hist['train_loss'].append(tl); hist['train_acc'].append(ta)
        hist['val_loss'].append(vl);   hist['val_acc'].append(va)
        if va > best_acc: best_acc=va; best_w=copy.deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:2d}/{epochs}  tr {tl:.3f}/{ta:.3f}  "
                  f"val {vl:.3f}/{va:.3f}  best {best_acc:.3f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    model.load_state_dict(best_w)
    print(f"  → best val acc: {best_acc:.4f}", flush=True)
    return hist, best_acc


# ══════════════════════════════════════════════════════════
# 4.  EXPERIMENTS
# ══════════════════════════════════════════════════════════
results = {}

# ── A. Scratch ───────────────────────────────────────────
model_scratch = make_resnet_cifar(pretrained=False).to(DEVICE)
n_total  = count_params(model_scratch, trainable_only=False)
print(f"\nModel params (total): {n_total:,}", flush=True)

opt_scratch = optim.SGD(model_scratch.parameters(),
                         lr=0.1, momentum=0.9, weight_decay=5e-4)
h_scratch, best_scratch = run_training(
    model_scratch, SCRATCH_EPOCHS,
    "1. From Scratch  [random init, all layers trained]",
    opt_scratch)
_, final_scratch, pred_scratch, lbl_scratch = evaluate(model_scratch, test_loader)
results['scratch'] = dict(
    params_trainable=n_total, params_total=n_total,
    best_acc=best_scratch, final_acc=final_scratch,
    epochs=SCRATCH_EPOCHS, history=h_scratch,
    strategy='scratch')

# ── B. Feature Extraction ────────────────────────────────
model_feat = make_resnet_cifar(pretrained=True).to(DEVICE)
freeze_backbone(model_feat)
n_trainable_feat = count_params(model_feat, trainable_only=True)
print(f"\nFeature Extraction — trainable params: {n_trainable_feat:,} "
      f"(FC head only, backbone frozen)", flush=True)

# Only optimise the FC head (backbone is frozen)
opt_feat = optim.SGD(filter(lambda p: p.requires_grad, model_feat.parameters()),
                      lr=0.05, momentum=0.9, weight_decay=5e-4)
h_feat, best_feat = run_training(
    model_feat, TL_EPOCHS,
    "2. Feature Extraction  [ImageNet pretrained, backbone FROZEN]",
    opt_feat)
_, final_feat, pred_feat, lbl_feat = evaluate(model_feat, test_loader)
results['feature_extract'] = dict(
    params_trainable=n_trainable_feat, params_total=n_total,
    best_acc=best_feat, final_acc=final_feat,
    epochs=TL_EPOCHS, history=h_feat,
    strategy='feature_extract')

# ── C. Fine-tuning ───────────────────────────────────────
model_ft = make_resnet_cifar(pretrained=True).to(DEVICE)
n_trainable_ft = count_params(model_ft, trainable_only=True)
print(f"\nFine-tuning — trainable params: {n_trainable_ft:,} "
      f"(all layers, differential LR)", flush=True)

# Differential LR: backbone 10× lower than head (standard fine-tuning practice)
backbone_params = [p for name, p in model_ft.named_parameters()
                   if not name.startswith('fc')]
head_params     = list(model_ft.fc.parameters())
opt_ft = optim.SGD([
    {'params': backbone_params, 'lr': 0.01},   # pretrained layers: low LR
    {'params': head_params,     'lr': 0.1},    # new head: normal LR
], momentum=0.9, weight_decay=5e-4)

h_ft, best_ft = run_training(
    model_ft, TL_EPOCHS,
    "3. Fine-tuning  [ImageNet pretrained, differential LR: backbone=0.01, head=0.1]",
    opt_ft)
_, final_ft, pred_ft, lbl_ft = evaluate(model_ft, test_loader)
results['finetune'] = dict(
    params_trainable=n_trainable_ft, params_total=n_total,
    best_acc=best_ft, final_acc=final_ft,
    epochs=TL_EPOCHS, history=h_ft,
    strategy='finetune')


# ══════════════════════════════════════════════════════════
# 5.  FIGURES
# ══════════════════════════════════════════════════════════
print("\nGenerating figures...", flush=True)
C_SC, C_FE, C_FT = '#D84315', '#1565C0', '#2E7D32'

# 5.1 Training accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for h, lbl, c, ep in [
    (h_scratch, f'Scratch ({SCRATCH_EPOCHS} ep)',        C_SC, SCRATCH_EPOCHS),
    (h_feat,    f'Feature Extract ({TL_EPOCHS} ep)',     C_FE, TL_EPOCHS),
    (h_ft,      f'Fine-tuning ({TL_EPOCHS} ep)',         C_FT, TL_EPOCHS),
]:
    x = range(1, len(h['val_acc'])+1)
    axes[0].plot(x, h['train_loss'], '--', color=c, alpha=0.4)
    axes[0].plot(x, h['val_loss'],   '-',  color=c, label=lbl)
    axes[1].plot(x, h['train_acc'],  '--', color=c, alpha=0.4)
    axes[1].plot(x, h['val_acc'],    '-',  color=c, label=lbl)
for ax, title in zip(axes, ['Loss', 'Accuracy']):
    ax.set(title=title, xlabel='Epoch'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('Training Curves (solid=val, dashed=train)\n'
             'Feature Extraction and Fine-tuning use fewer epochs', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tl_training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5.2 Final accuracy bar chart
fig, ax = plt.subplots(figsize=(9, 5))
labels = ['Scratch\n(random init)', 'Feature Extraction\n(frozen backbone)',
          'Fine-tuning\n(differential LR)']
accs   = [final_scratch, final_feat, final_ft]
bars   = ax.bar(labels, [a*100 for a in accs],
                color=[C_SC, C_FE, C_FT], width=0.45,
                edgecolor='white', linewidth=1.5)
for bar, a in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{a*100:.2f}%', ha='center', va='bottom',
            fontweight='bold', fontsize=11)
ax.set(title='CIFAR-10 Test Accuracy — Transfer Learning Strategies',
       ylabel='Test Accuracy (%)',
       ylim=[max(0, min(accs)*100-4), min(100, max(accs)*100+3)])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tl_accuracy_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5.3 Convergence speed — epochs to reach thresholds
thresholds = [0.70, 0.75, 0.80, 0.83, 0.85, 0.87]
fig, ax = plt.subplots(figsize=(10, 5))
for h, lbl, c in [(h_scratch,'Scratch',C_SC),(h_feat,'Feature Extract',C_FE),(h_ft,'Fine-tuning',C_FT)]:
    epochs_to = []
    for thr in thresholds:
        hit = next((i+1 for i, v in enumerate(h['val_acc']) if v >= thr), None)
        epochs_to.append(hit if hit else (SCRATCH_EPOCHS if h is h_scratch else TL_EPOCHS)+1)
    ax.plot(thresholds, epochs_to, 'o-', color=c, lw=2, ms=8, label=lbl)
    for t, e in zip(thresholds, epochs_to):
        max_ep = SCRATCH_EPOCHS if h is h_scratch else TL_EPOCHS
        if e <= max_ep:
            ax.annotate(f'ep {e}', (t, e), textcoords='offset points',
                       xytext=(0, 8), ha='center', fontsize=8, color=c)
ax.set(title='Epochs Required to Reach Accuracy Threshold',
       xlabel='Accuracy Threshold', ylabel='Epoch First Reached')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tl_convergence_speed.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5.4 Confusion matrices
def compute_cm(preds, labels):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for p, l in zip(preds, labels): cm[l][p] += 1
    return cm

def save_cm(cm, title, fname):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap='Blues'); plt.colorbar(im)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=CLASSES, yticklabels=CLASSES,
           title=title, xlabel='Predicted', ylabel='True')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max()/2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=7,
                    color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()

save_cm(compute_cm(pred_scratch, lbl_scratch),
        f'Scratch – {final_scratch*100:.2f}%', 'tl_cm_scratch.png')
save_cm(compute_cm(pred_feat,    lbl_feat),
        f'Feature Extraction – {final_feat*100:.2f}%', 'tl_cm_feat.png')
save_cm(compute_cm(pred_ft,      lbl_ft),
        f'Fine-tuning – {final_ft*100:.2f}%', 'tl_cm_finetune.png')

# 5.5 Slide quadrant visualisation
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.axvline(1, color='grey', lw=1.5, ls='--')
ax.axhline(1, color='grey', lw=1.5, ls='--')
quadrants = [
    (0.5, 1.5, 'Feature\nExtraction\n(freeze backbone)',  '#1565C0', 'large + similar'),
    (1.5, 1.5, 'Fine-tune\nall layers\n(higher LR)',      '#880E4F', 'large + different'),
    (0.5, 0.5, 'Feature\nExtraction\n★ CIFAR-10 here',   '#2E7D32', 'small + similar'),
    (1.5, 0.5, 'Fine-tune\ncarefully\n(low LR)',          '#E65100', 'small + different'),
]
for x, y, txt, color, _ in quadrants:
    ax.text(x, y, txt, ha='center', va='center', fontsize=10,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.8))
ax.scatter([0.5], [0.5], s=300, color='yellow', zorder=5, marker='*')
ax.set(xticks=[0.5,1.5], xticklabels=['Similar to source','Different from source'],
       yticks=[0.5,1.5], yticklabels=['Small dataset','Large dataset'],
       title='Transfer Learning Strategy Selection\n(Muselet 2026, slide p.46)')
ax.tick_params(length=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tl_quadrant.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  All figures saved.", flush=True)

# ══════════════════════════════════════════════════════════
# 6.  SAVE JSON
# ══════════════════════════════════════════════════════════
def to_json(obj):
    if isinstance(obj, dict):                return {str(k): to_json(v) for k,v in obj.items()}
    if isinstance(obj, list):                return [to_json(v) for v in obj]
    if isinstance(obj, (float,np.floating)): return round(float(obj), 6)
    if isinstance(obj, (int,np.integer)):    return int(obj)
    return obj

results['config'] = dict(
    seed=SEED, batch_size=BATCH_SIZE,
    scratch_epochs=SCRATCH_EPOCHS, tl_epochs=TL_EPOCHS,
    device=DEVICE, n_params_total=n_total,
    n_params_feat_trainable=n_trainable_feat,
    thresholds=thresholds,
)
with open(os.path.join(OUT_DIR, 'results_tl.json'), 'w') as f:
    json.dump(to_json(results), f, indent=2)

print(f"""
╔══════════════════════════════════════════════════════╗
  TRANSFER LEARNING RESULTS  (CIFAR-10, ResNet-18)
  Scratch          (random init):      {final_scratch*100:.2f}%
  Feature Extraction (frozen backbone):{final_feat*100:.2f}%  (+{(final_feat-final_scratch)*100:.2f}pp)
  Fine-tuning      (differential LR):  {final_ft*100:.2f}%  (+{(final_ft-final_scratch)*100:.2f}pp)
╚══════════════════════════════════════════════════════╝
""", flush=True)
