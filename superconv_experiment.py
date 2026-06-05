#!/usr/bin/env python3
"""
Super Convergence: 1-Cycle LR Policy on CIFAR-10
Practical Session 2 – Advanced Deep Learning Strategies
USTH Deep Learning 2026
Duong Tan Binh – 2540007

Baseline  : SGD + Cosine Annealing LR (standard)
Improved  : SGD + 1-Cycle LR (super convergence, Smith 2018)

Expected  : 1-Cycle reaches higher accuracy AND converges faster.
"""

import os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════
SEED        = 42
BATCH_SIZE  = 128
EPOCHS      = 30          # both runs use same number of epochs (fair comparison)
MAX_LR      = 0.1         # peak LR for 1-cycle
NUM_CLASSES = 10
OUT_DIR     = os.path.dirname(os.path.abspath(__file__))

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

_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
_pad_crop  = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip()])

print("Loading CIFAR-10...", flush=True)
data_dir  = os.path.join(OUT_DIR, 'data')
_train_ds = datasets.CIFAR10(data_dir, train=True,  download=True,  transform=_to_tensor)
_test_ds  = datasets.CIFAR10(data_dir, train=False, download=True,  transform=_to_tensor)

def _ds_to_tensors(ds):
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    xs, ys = zip(*[(x, y) for x, y in loader])
    return torch.cat(xs), torch.cat(ys)

_train_x, _train_y = _ds_to_tensors(_train_ds)
_test_x,  _test_y  = _ds_to_tensors(_test_ds)
print(f"  Train: {_train_x.shape}  Test: {_test_x.shape}", flush=True)

def get_loaders():
    train_ds = TensorDataset(_train_x, _train_y)
    test_ds  = TensorDataset(_test_x,  _test_y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader

train_loader, test_loader = get_loaders()
STEPS_PER_EPOCH = len(train_loader)

# ══════════════════════════════════════════════════════════
# 2.  MODEL  (same architecture for both experiments)
# ══════════════════════════════════════════════════════════
class ConvNet(nn.Module):
    """Moderate CNN (~650K params) — same for baseline and super-convergence."""
    def __init__(self):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.feat = nn.Sequential(blk(3, 32), blk(32, 64),
                                  nn.Conv2d(64, 128, 3, padding=1),
                                  nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.cls  = nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(128*4*4, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, NUM_CLASSES))
    def forward(self, x):
        return self.cls(self.feat(x).flatten(1))

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ══════════════════════════════════════════════════════════
# 3.  AUGMENTATION & EVALUATION
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

# ══════════════════════════════════════════════════════════
# 4.  TRAINING LOOPS
# ══════════════════════════════════════════════════════════
def run_cosine(model, epochs, label):
    """Baseline: SGD + Cosine Annealing LR."""
    opt   = optim.SGD(model.parameters(), lr=MAX_LR, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    return _train_loop(model, epochs, label, opt, sched, step_per_batch=False)

def run_onecycle(model, epochs, label):
    """Super Convergence: SGD + 1-Cycle LR (Smith 2018)."""
    opt   = optim.SGD(model.parameters(), lr=MAX_LR/10, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.OneCycleLR(
                opt, max_lr=MAX_LR,
                epochs=epochs, steps_per_epoch=STEPS_PER_EPOCH,
                pct_start=0.3,        # 30% of training = warmup phase
                anneal_strategy='cos',
                div_factor=10.0,      # initial_lr = max_lr / 10
                final_div_factor=1e4, # final_lr  = initial_lr / 10000
    )
    return _train_loop(model, epochs, label, opt, sched, step_per_batch=True)

def _train_loop(model, epochs, label, opt, sched, step_per_batch):
    hist = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[], lr=[])
    best_acc=0.; best_w=None
    tag = "1-Cycle LR" if step_per_batch else "Cosine Annealing"
    print(f"\n{'─'*58}\n  {label}  [{tag}]  epochs={epochs}\n{'─'*58}", flush=True)
    t0 = time.time()
    for ep in range(1, epochs+1):
        model.train(); tls=tcr=tn=0
        for xb, yb in train_loader:
            xb = aug(xb)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out  = model(xb); loss = F.cross_entropy(out, yb)
            loss.backward(); opt.step()
            if step_per_batch: sched.step()   # 1-cycle: step every batch
            tls += loss.item()*len(yb)
            tcr += (out.argmax(1)==yb).sum().item(); tn += len(yb)
        if not step_per_batch: sched.step()   # cosine: step every epoch
        vl, va, _, _ = evaluate(model, test_loader)
        tl, ta = tls/tn, tcr/tn
        current_lr = opt.param_groups[0]['lr']
        hist['train_loss'].append(tl); hist['train_acc'].append(ta)
        hist['val_loss'].append(vl);   hist['val_acc'].append(va)
        hist['lr'].append(current_lr)
        if va > best_acc: best_acc=va; best_w=copy.deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:2d}/{epochs}  tr {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}  "
                  f"lr {current_lr:.5f}  best {best_acc:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    model.load_state_dict(best_w)
    print(f"  → best val acc: {best_acc:.4f}", flush=True)
    return hist, best_acc

# ══════════════════════════════════════════════════════════
# 5.  EXPERIMENTS
# ══════════════════════════════════════════════════════════
results = {}
n_params = count_params(ConvNet())
print(f"\nModel params: {n_params:,}", flush=True)

# Baseline — Cosine Annealing
model_cos = ConvNet().to(DEVICE)
h_cos, best_cos = run_cosine(model_cos, EPOCHS, "Baseline (Cosine Annealing)")
_, final_cos, pred_cos, lbl_cos = evaluate(model_cos, test_loader)
results['cosine'] = dict(params=n_params, best_acc=best_cos,
                         final_acc=final_cos, history=h_cos)

# Super Convergence — 1-Cycle LR
model_oc = ConvNet().to(DEVICE)
h_oc, best_oc = run_onecycle(model_oc, EPOCHS, "Super Convergence (1-Cycle LR)")
_, final_oc, pred_oc, lbl_oc = evaluate(model_oc, test_loader)
results['onecycle'] = dict(params=n_params, best_acc=best_oc,
                           final_acc=final_oc, history=h_oc)

# ══════════════════════════════════════════════════════════
# 6.  FIGURES
# ══════════════════════════════════════════════════════════
print("\nGenerating figures...", flush=True)
C_B, C_SC = '#D84315', '#1565C0'
epochs_range = range(1, EPOCHS+1)

# 6.1 Training curves (loss + accuracy)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for h, lbl, c in [(h_cos,'Baseline (Cosine)',C_B),(h_oc,'Super Convergence (1-Cycle)',C_SC)]:
    axes[0].plot(epochs_range, h['train_loss'], '--', color=c, alpha=0.4)
    axes[0].plot(epochs_range, h['val_loss'],   '-',  color=c, label=lbl)
    axes[1].plot(epochs_range, h['train_acc'],  '--', color=c, alpha=0.4)
    axes[1].plot(epochs_range, h['val_acc'],    '-',  color=c, label=lbl)
for ax, title in zip(axes, ['Loss', 'Accuracy']):
    ax.set(title=title, xlabel='Epoch'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('Training Curves (solid=val, dashed=train)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'sc_training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# 6.2 LR schedules
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs_range, h_cos['lr'],  '-', color=C_B,  lw=2, label='Cosine Annealing')
ax.plot(epochs_range, h_oc['lr'],   '-', color=C_SC, lw=2, label='1-Cycle LR')
ax.set(title='Learning Rate Schedule Comparison', xlabel='Epoch', ylabel='Learning Rate')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'sc_lr_schedule.png'), dpi=150, bbox_inches='tight')
plt.close()

# 6.3 Convergence speed — epochs to reach thresholds
fig, ax = plt.subplots(figsize=(9, 5))
thresholds = [0.70, 0.75, 0.78, 0.80, 0.82, 0.84]
for h, lbl, c in [(h_cos,'Cosine',C_B),(h_oc,'1-Cycle',C_SC)]:
    epochs_to = []
    for thr in thresholds:
        hit = next((i+1 for i, v in enumerate(h['val_acc']) if v >= thr), None)
        epochs_to.append(hit if hit else EPOCHS+1)
    ax.plot(thresholds, epochs_to, 'o-', color=c, lw=2, ms=8, label=lbl)
    for t, e in zip(thresholds, epochs_to):
        if e <= EPOCHS:
            ax.annotate(f'ep {e}', (t, e), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8, color=c)
ax.set(title='Epochs Required to Reach Accuracy Threshold',
       xlabel='Accuracy Threshold', ylabel='Epoch First Reached')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'sc_convergence_speed.png'), dpi=150, bbox_inches='tight')
plt.close()

# 6.4 Accuracy bar chart
fig, ax = plt.subplots(figsize=(7, 5))
accs = [final_cos, final_oc]
bars = ax.bar(['Baseline\n(Cosine Annealing)', 'Super Convergence\n(1-Cycle LR)'],
              [a*100 for a in accs], color=[C_B, C_SC], width=0.45,
              edgecolor='white', linewidth=1.5)
for bar, a in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{a*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set(title='CIFAR-10 Final Test Accuracy', ylabel='Test Accuracy (%)',
       ylim=[max(0, min(accs)*100-3), min(100, max(accs)*100+3)])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'sc_accuracy_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# 6.5 Confusion matrices
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

save_cm(compute_cm(pred_cos, lbl_cos), f'Baseline (Cosine) – {final_cos*100:.2f}%', 'sc_cm_cosine.png')
save_cm(compute_cm(pred_oc,  lbl_oc),  f'Super Convergence (1-Cycle) – {final_oc*100:.2f}%', 'sc_cm_onecycle.png')

print("  All figures saved.", flush=True)

# ══════════════════════════════════════════════════════════
# 7.  SAVE JSON
# ══════════════════════════════════════════════════════════
def to_json(obj):
    if isinstance(obj, dict):                return {str(k): to_json(v) for k,v in obj.items()}
    if isinstance(obj, list):                return [to_json(v) for v in obj]
    if isinstance(obj, (float,np.floating)): return round(float(obj), 6)
    if isinstance(obj, (int,np.integer)):    return int(obj)
    return obj

results['config'] = dict(seed=SEED, batch_size=BATCH_SIZE, epochs=EPOCHS,
                         max_lr=MAX_LR, device=DEVICE, params=n_params)

with open(os.path.join(OUT_DIR, 'results_sc.json'), 'w') as f:
    json.dump(to_json(results), f, indent=2)

gain = (final_oc - final_cos)*100
print(f"""
╔══════════════════════════════════════════════════╗
  SUPER CONVERGENCE RESULTS  (CIFAR-10)
  Baseline  (Cosine LR):  {final_cos*100:.2f}%
  1-Cycle LR (Super CV):  {final_oc*100:.2f}%  ({gain:+.2f} pp)
╚══════════════════════════════════════════════════╝
""", flush=True)
