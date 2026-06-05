#!/usr/bin/env python3
"""
Knowledge Distillation on CIFAR-10
Practical Session 2 – Advanced Deep Learning Strategies
USTH Deep Learning 2026
Duong Tan Binh – 2540007
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
TEACHER_EPOCHS  = 30
STUDENT_EPOCHS  = 60           # need more epochs to converge
ABLATION_EPOCHS = 20
T_DEFAULT       = 1            # T=1 no KL amplification (T²=1); teacher raw softmax already encodes inter-class similarity
ALPHA_DEFAULT   = 0.5
NUM_CLASSES     = 10
OUT_DIR         = os.path.dirname(os.path.abspath(__file__))
TEACHER_CKPT    = os.path.join(OUT_DIR, 'teacher_weights.pt')

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = ('mps'  if torch.backends.mps.is_available() else
          'cuda' if torch.cuda.is_available()       else 'cpu')
print(f"Device: {DEVICE}", flush=True)

CLASSES = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')


# ══════════════════════════════════════════════════════════
# 1.  DATA  (pre-load into RAM as tensors – avoids worker overhead)
# ══════════════════════════════════════════════════════════
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

print("Loading CIFAR-10 into memory...", flush=True)
data_dir = os.path.join(OUT_DIR, 'data')
_train_ds = datasets.CIFAR10(data_dir, train=True,  download=True,  transform=_to_tensor)
_test_ds  = datasets.CIFAR10(data_dir, train=False, download=True, transform=_to_tensor)

def _ds_to_tensors(ds):
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    xs, ys = zip(*[(x, y) for x, y in loader])
    return torch.cat(xs), torch.cat(ys)

_train_x, _train_y = _ds_to_tensors(_train_ds)   # (50000, 3, 32, 32), (50000,)
_test_x,  _test_y  = _ds_to_tensors(_test_ds)    # (10000, 3, 32, 32), (10000,)
print(f"  Train: {_train_x.shape}  Test: {_test_x.shape}", flush=True)

# Augmentation applied on-the-fly inside train step
_pad_crop = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])

def get_loaders():
    train_ds = TensorDataset(_train_x, _train_y)
    test_ds  = TensorDataset(_test_x,  _test_y)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader

train_loader, test_loader = get_loaders()


# ══════════════════════════════════════════════════════════
# 2.  MODELS
# ══════════════════════════════════════════════════════════
def make_teacher():
    """ResNet-18 adapted for CIFAR-10 (32×32)."""
    m = models.resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, NUM_CLASSES)
    return m


class StudentCNN(nn.Module):
    """Compact CNN (~190K parameters) — intentionally weak so KD benefit is clear."""
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.cls = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64*4*4, 128), nn.ReLU(inplace=True),
            nn.Linear(128, NUM_CLASSES),
        )
    def forward(self, x):
        return self.cls(self.feat(x).flatten(1))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════
# 3.  LOSS & TRAINING
# ══════════════════════════════════════════════════════════
def aug(x):
    """Apply random crop+flip to a batch of tensors on CPU, return same device."""
    dev = x.device
    x = x.cpu()
    out = torch.stack([_pad_crop(xi) for xi in x])
    return out.to(dev)


def kd_loss(s, t, y, T, alpha):
    hard = F.cross_entropy(s, y)
    soft = F.kl_div(F.log_softmax(s/T, 1), F.softmax(t/T, 1).detach(), reduction='batchmean') * T*T
    return (1-alpha)*hard + alpha*soft


def train_standard(model, loader, opt):
    model.train(); ls=cr=n=0
    for xb, yb in loader:
        xb = aug(xb)                          # augment on CPU before device transfer
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        out = model(xb); loss = F.cross_entropy(out, yb)
        loss.backward(); opt.step()
        ls += loss.item()*len(yb); cr += (out.argmax(1)==yb).sum().item(); n+=len(yb)
    return ls/n, cr/n


def train_distill(student, teacher, loader, opt, T, alpha):
    student.train(); teacher.eval(); ls_ce=cr=n=0
    for xb, yb in loader:
        xb = aug(xb)                          # augment on CPU before device transfer
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        with torch.no_grad(): t_out = teacher(xb)
        opt.zero_grad()
        s_out = student(xb); loss = kd_loss(s_out, t_out, yb, T, alpha)
        loss.backward(); opt.step()
        # track CE loss (not KD composite) so training curves stay comparable to baseline
        with torch.no_grad():
            ce = F.cross_entropy(s_out, yb)
        ls_ce += ce.item()*len(yb); cr += (s_out.argmax(1)==yb).sum().item(); n+=len(yb)
    return ls_ce/n, cr/n


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ls=cr=n=0; preds=[]; lbls=[]
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb); loss = F.cross_entropy(out, yb)
        ls += loss.item()*len(yb); pred = out.argmax(1)
        cr += (pred==yb).sum().item(); n+=len(yb)
        preds.extend(pred.cpu().tolist()); lbls.extend(yb.cpu().tolist())
    return ls/n, cr/n, preds, lbls


def run_training(model, epochs, label, distill=False, teacher=None, T=T_DEFAULT, alpha=ALPHA_DEFAULT):
    opt   = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist  = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
    best_acc=0.; best_w=None
    tag = f"distill T={T} α={alpha}" if distill else "standard CE"
    print(f"\n{'─'*58}\n  {label}  [{tag}]  epochs={epochs}\n{'─'*58}", flush=True)
    t0 = time.time()
    for ep in range(1, epochs+1):
        if distill: tl,ta = train_distill(model, teacher, train_loader, opt, T, alpha)
        else:        tl,ta = train_standard(model, train_loader, opt)
        vl,va,_,_ = evaluate(model, test_loader)
        sched.step()
        hist['train_loss'].append(tl); hist['train_acc'].append(ta)
        hist['val_loss'].append(vl);   hist['val_acc'].append(va)
        if va > best_acc: best_acc=va; best_w=copy.deepcopy(model.state_dict())
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:2d}/{epochs}  tr {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}  "
                  f"best {best_acc:.3f}  [{time.time()-t0:.0f}s]", flush=True)
    model.load_state_dict(best_w)
    print(f"  → best val acc: {best_acc:.4f}", flush=True)
    return hist, best_acc


# ══════════════════════════════════════════════════════════
# 4.  MAIN EXPERIMENTS
# ══════════════════════════════════════════════════════════
results = {}

# Teacher — load from cache if available, otherwise train and save
teacher   = make_teacher().to(DEVICE)
n_teacher = count_params(teacher)
print(f"\nTeacher params: {n_teacher:,}", flush=True)

if os.path.exists(TEACHER_CKPT):
    teacher.load_state_dict(torch.load(TEACHER_CKPT, map_location=DEVICE))
    print(f"  Teacher loaded from cache: {TEACHER_CKPT}", flush=True)
    _, final_teacher, pred_teacher, lbl_teacher = evaluate(teacher, test_loader)
    best_teacher = final_teacher
    h_teacher = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
    print(f"  Teacher cached accuracy: {final_teacher:.4f}", flush=True)
else:
    h_teacher, best_teacher = run_training(teacher, TEACHER_EPOCHS, "Teacher (ResNet-18)")
    torch.save(teacher.state_dict(), TEACHER_CKPT)
    print(f"  Teacher weights saved to: {TEACHER_CKPT}", flush=True)
    _, final_teacher, pred_teacher, lbl_teacher = evaluate(teacher, test_loader)

results['teacher'] = dict(params=n_teacher, best_acc=best_teacher,
                           final_acc=final_teacher, history=h_teacher)

# Baseline Student
student_base = StudentCNN().to(DEVICE)
n_student    = count_params(student_base)
print(f"\nStudent params: {n_student:,}  ({n_teacher/n_student:.1f}× smaller)", flush=True)
h_base, best_base = run_training(student_base, STUDENT_EPOCHS, "Student – Baseline")
_, final_base, pred_base, lbl_base = evaluate(student_base, test_loader)
results['baseline'] = dict(params=n_student, best_acc=best_base,
                            final_acc=final_base, history=h_base)

# Distilled Student
student_kd = StudentCNN().to(DEVICE)
h_kd, best_kd = run_training(student_kd, STUDENT_EPOCHS, "Student – KD",
                              distill=True, teacher=teacher, T=T_DEFAULT, alpha=ALPHA_DEFAULT)
_, final_kd, pred_kd, lbl_kd = evaluate(student_kd, test_loader)
results['kd'] = dict(params=n_student, best_acc=best_kd, final_acc=final_kd,
                     T=T_DEFAULT, alpha=ALPHA_DEFAULT, history=h_kd)


# ══════════════════════════════════════════════════════════
# 5.  ABLATION – TEMPERATURE
# ══════════════════════════════════════════════════════════
T_values  = [1, 2, 4, 8, 16]
temp_accs = {}
for T in T_values:
    m = StudentCNN().to(DEVICE)
    _, acc = run_training(m, ABLATION_EPOCHS, f"KD T={T}",
                          distill=True, teacher=teacher, T=T, alpha=ALPHA_DEFAULT)
    temp_accs[T] = float(acc)

# Baseline trained for same number of epochs – fair comparison for ablation
_abl_base = StudentCNN().to(DEVICE)
_, ablation_baseline_acc = run_training(_abl_base, ABLATION_EPOCHS, "Ablation Baseline")
results['ablation_T'] = temp_accs
results['ablation_baseline_acc'] = float(ablation_baseline_acc)

# ══════════════════════════════════════════════════════════
# 6.  ABLATION – ALPHA
# ══════════════════════════════════════════════════════════
alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
alpha_accs   = {}
for alpha in alpha_values:
    m = StudentCNN().to(DEVICE)
    _, acc = run_training(m, ABLATION_EPOCHS, f"KD α={alpha}",
                          distill=True, teacher=teacher, T=T_DEFAULT, alpha=alpha)
    alpha_accs[alpha] = float(acc)
results['ablation_alpha'] = alpha_accs


# ══════════════════════════════════════════════════════════
# 7.  FIGURES
# ══════════════════════════════════════════════════════════
print("\nGenerating figures...", flush=True)
C_T, C_B, C_K = '#1565C0', '#D84315', '#2E7D32'

# 7.1 Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for h, lbl, c in [(h_teacher,'Teacher (ResNet-18)',C_T),(h_base,'Student Baseline',C_B),
                   (h_kd,f'Student KD (T={T_DEFAULT}, α={ALPHA_DEFAULT})',C_K)]:
    if not h['val_acc']:          # skip teacher when loaded from cache (no history)
        continue
    ep = range(1, len(h['val_acc'])+1)
    axes[0].plot(ep, h['train_loss'],'--',color=c,alpha=0.35)
    axes[0].plot(ep, h['val_loss'],  '-', color=c,label=lbl)
    axes[1].plot(ep, h['train_acc'],'--',color=c,alpha=0.35)
    axes[1].plot(ep, h['val_acc'],  '-', color=c,label=lbl)
for ax, title in zip(axes, ['Loss','Accuracy']):
    ax.set(title=title, xlabel='Epoch'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle('Training Curves  (solid=val, dashed=train)\n'
             'KD train loss = CE component only (comparable to baseline)', y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'fig_training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7.2 Confusion matrices
def compute_cm(preds, labels):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for p, l in zip(preds, labels): cm[l][p] += 1
    return cm

def save_cm(cm, title, fname):
    fig, ax = plt.subplots(figsize=(9,7))
    im = ax.imshow(cm, cmap='Blues'); plt.colorbar(im)
    ax.set(xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES),
           xticklabels=CLASSES, yticklabels=CLASSES,
           title=title, xlabel='Predicted label', ylabel='True label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max()/2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,
                    color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,fname), dpi=150, bbox_inches='tight')
    plt.close()

save_cm(compute_cm(pred_teacher,lbl_teacher), f'Teacher (ResNet-18) – {final_teacher*100:.2f}%','fig_cm_teacher.png')
save_cm(compute_cm(pred_base,   lbl_base),    f'Student Baseline – {final_base*100:.2f}%',       'fig_cm_baseline.png')
save_cm(compute_cm(pred_kd,     lbl_kd),      f'Student + KD – {final_kd*100:.2f}%',             'fig_cm_kd.png')

# 7.3 Temperature softmax shape
fig, axes = plt.subplots(1,5,figsize=(18,3.5),sharey=True)
eg = np.array([3.5, 1.2, 0.8, -0.2, 0.3, -0.5, 0.1, -1.2, 0.5, 0.2])
for ax, T in zip(axes, [1,2,5,10,20]):   # match slide (Muselet 2026, p.42)
    s = eg/T; p = np.exp(s-s.max()); p/=p.sum()
    ax.bar(range(10), p, color=['tomato' if i==0 else 'steelblue' for i in range(10)])
    ax.set_title(f'T = {T}', fontsize=12)
    ax.set_xticks(range(10)); ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=7)
axes[0].set_ylabel('Probability')
plt.suptitle('Effect of Temperature T on Soft Labels  (red = true class)', y=1.05, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'fig_temperature_softmax.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7.4 Ablation
abl_base = results['ablation_baseline_acc']
fig, axes = plt.subplots(1,2,figsize=(13,4.5))
axes[0].plot(T_values, [temp_accs[T] for T in T_values],'o-',color=C_K,lw=2,ms=8,label='KD student')
axes[0].axhline(abl_base,ls='--',color='grey',label=f'Baseline same epochs ({abl_base*100:.2f}%)')
for T in T_values:
    axes[0].annotate(f'{temp_accs[T]*100:.1f}%',(T,temp_accs[T]),
                     textcoords='offset points',xytext=(0,7),ha='center',fontsize=8)
axes[0].set(title=f'Accuracy vs Temperature T  (α={ALPHA_DEFAULT}, {ABLATION_EPOCHS} epochs)',
            xlabel='T', ylabel='Best Test Accuracy')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(alpha_values,[alpha_accs[a] for a in alpha_values],'s-',color=C_B,lw=2,ms=8,label='KD student')
axes[1].axhline(abl_base,ls='--',color='grey',label=f'Baseline same epochs ({abl_base*100:.2f}%)')
for a in alpha_values:
    axes[1].annotate(f'{alpha_accs[a]*100:.1f}%',(a,alpha_accs[a]),
                     textcoords='offset points',xytext=(0,7),ha='center',fontsize=8)
axes[1].set(title=f'Accuracy vs Alpha α  (T={T_DEFAULT}, {ABLATION_EPOCHS} epochs)',
            xlabel='α', ylabel='Best Test Accuracy')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'fig_ablation.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7.5 Final accuracy bar chart
fig, ax = plt.subplots(figsize=(8,5))
accs = [final_teacher, final_base, final_kd]
bars = ax.bar(['Teacher\n(ResNet-18)','Student\nBaseline','Student\n+KD'],
              [a*100 for a in accs], color=[C_T,C_B,C_K], width=0.45, edgecolor='white', linewidth=1.5)
for bar, a in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f'{a*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set(title='CIFAR-10 Test Accuracy Comparison', ylabel='Test Accuracy (%)',
       ylim=[max(0,min(accs)*100-5), min(100,max(accs)*100+5)])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'fig_accuracy_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7.6 Model size & efficiency
fig, axes = plt.subplots(1,2,figsize=(13,4.5))
axes[0].bar(['Teacher\n(ResNet-18)','Student\n(Small CNN)'],
            [n_teacher/1e6, n_student/1e6], color=[C_T,C_K], width=0.35)
for i,v in enumerate([n_teacher/1e6, n_student/1e6]):
    axes[0].text(i, v+0.05, f'{v:.2f}M', ha='center', va='bottom', fontweight='bold')
axes[0].set(title='Number of Parameters', ylabel='Parameters (M)'); axes[0].grid(axis='y',alpha=0.3)

axes[1].scatter([n_teacher/1e6],[final_teacher*100],s=200,color=C_T,zorder=5,label='Teacher')
axes[1].scatter([n_student/1e6],[final_base*100],   s=200,color=C_B,marker='s',zorder=5,label='Baseline')
axes[1].scatter([n_student/1e6],[final_kd*100],     s=200,color=C_K,marker='^',zorder=5,label='KD')
for lbl,x,y in [('Teacher',n_teacher/1e6,final_teacher*100),
                 ('Baseline',n_student/1e6,final_base*100),
                 ('KD',n_student/1e6,final_kd*100)]:
    axes[1].annotate(f'{lbl}\n{y:.2f}%',(x,y),textcoords='offset points',xytext=(6,2),fontsize=8)
axes[1].set(title='Accuracy–Efficiency Trade-off',xlabel='Parameters (M)',ylabel='Accuracy (%)')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'fig_model_size.png'), dpi=150, bbox_inches='tight')
plt.close()

print("  All figures saved.", flush=True)

# ══════════════════════════════════════════════════════════
# 8.  SAVE JSON
# ══════════════════════════════════════════════════════════
def to_json(obj):
    if isinstance(obj, dict):                return {str(k): to_json(v) for k,v in obj.items()}
    if isinstance(obj, list):                return [to_json(v) for v in obj]
    if isinstance(obj, (float,np.floating)): return round(float(obj),6)
    if isinstance(obj, (int,np.integer)):    return int(obj)
    return obj

# Save experiment config alongside results so the report script can read them
results['config'] = {
    'batch_size':            BATCH_SIZE,
    'teacher_epochs':        TEACHER_EPOCHS,
    'student_epochs':        STUDENT_EPOCHS,
    'ablation_epochs':       ABLATION_EPOCHS,
    'T_default':             T_DEFAULT,
    'alpha_default':         ALPHA_DEFAULT,
    'device':                DEVICE,
    'T_values':              T_values,
    'alpha_values':          alpha_values,
    'ablation_baseline_acc': results['ablation_baseline_acc'],
}

with open(os.path.join(OUT_DIR,'results.json'),'w') as f:
    json.dump(to_json(results), f, indent=2)

kd_gain     = (final_kd - final_base)*100
compression = n_teacher / n_student
print(f"""
╔══════════════════════════════════════════════════╗
  CIFAR-10 FINAL RESULTS
  Teacher    ResNet-18  {n_teacher/1e6:5.2f}M  {final_teacher*100:6.2f}%
  Baseline   Small CNN  {n_student/1e6:5.2f}M  {final_base*100:6.2f}%
  KD         Small CNN  {n_student/1e6:5.2f}M  {final_kd*100:6.2f}%  ({kd_gain:+.2f}pp)
  Compression ratio: {compression:.1f}×
╚══════════════════════════════════════════════════╝
""", flush=True)
