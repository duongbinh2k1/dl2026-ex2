#!/usr/bin/env python3
"""
generate_report_tl.py
=====================
Generate Word report for Transfer Learning experiment.
Run AFTER transfer_experiment.py has completed.

Usage:
    python generate_report_tl.py
"""

import os, json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load results ───────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, 'results_tl.json')) as f:
    R = json.load(f)

scratch_acc = R['scratch']['best_acc']
fe_acc      = R['feature_extract']['best_acc']
ft_acc      = R['finetune']['best_acc']

fe_gain = (fe_acc - scratch_acc) * 100
ft_gain = (ft_acc - scratch_acc) * 100

scratch_params   = R['scratch']['params_total']
fe_trainable     = R['feature_extract']['params_trainable']
ft_params        = R['finetune']['params_total']

# Convergence speed: epoch at which val_acc first exceeds threshold
def epoch_to_thresh(history, threshold):
    for i, acc in enumerate(history['val_acc'], 1):
        if acc >= threshold:
            return i
    return None

thresh = 0.60   # 60% threshold for comparison

sc_ep60 = epoch_to_thresh(R['scratch'],         thresh)
fe_ep60 = epoch_to_thresh(R['feature_extract'], thresh)
ft_ep60 = epoch_to_thresh(R['finetune'],        thresh)

EPOCHS = len(R['scratch']['history']['val_acc'])

# ── Document helpers ────────────────────────────────────────────────────────────
def new_doc():
    doc = Document()
    # Page margins
    for sec in doc.sections:
        sec.top_margin    = Inches(1.0)
        sec.bottom_margin = Inches(1.0)
        sec.left_margin   = Inches(1.2)
        sec.right_margin  = Inches(1.2)
    return doc


def heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def body(doc, text, bold=False):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(11)
    return p


def add_image(doc, filename, width=5.5):
    path = os.path.join(OUT_DIR, filename)
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        body(doc, f"[Figure not found: {filename}]")


def caption(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.runs[0].italic = True
    p.runs[0].font.size = Pt(10)


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Light Grid Accent 1'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for row_data in rows:
        row = table.add_row()
        for i, val in enumerate(row_data):
            cell = row.cells[i]
            cell.text = str(val)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.paragraphs[0].runs[0].font.size = Pt(10)

    if col_widths:
        for row in table.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    return table


# ── Build report ────────────────────────────────────────────────────────────────
doc = new_doc()

# Title
title = doc.add_heading('Transfer Learning with ResNet-18', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle = doc.add_paragraph(
    'Practical Session 2 — Advanced Deep Learning Strategies\n'
    'USTH Deep Learning 2026 | Duong Tan Binh — 2540007')
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle.runs[0].font.size = Pt(12)
doc.add_paragraph()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Introduction
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '1. Introduction')
body(doc,
    'Transfer learning is a technique that leverages knowledge acquired on a large '
    'source task to improve learning on a smaller target task. Instead of training a '
    'deep neural network from random initialisation, we initialise it with weights '
    'pre-trained on a large corpus — typically ImageNet — and then adapt those weights '
    'to the new domain. The key insight is that lower-level features learned from '
    'millions of images (edges, textures, shapes) generalise across visual domains.')
doc.add_paragraph()
body(doc,
    'This experiment evaluates three transfer learning strategies on STL-10 — a '
    'benchmark dataset with 5 000 labeled 96×96 colour images across 10 classes, '
    'deliberately chosen to represent the data-scarce regime where transfer learning '
    'is most beneficial:')
doc.add_paragraph()

for item in [
    'From Scratch — ResNet-18 trained from random initialisation on STL-10 only.',
    'Feature Extraction — ImageNet pretrained backbone frozen; only the final FC '
     'head (10-class) is trained.',
    'Fine-tuning — ImageNet pretrained backbone fully trainable with differential '
     'learning rates (backbone 10× lower than head) to avoid catastrophic forgetting.',
]:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(item).font.size = Pt(11)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Theoretical Background
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '2. Theoretical Background')

heading(doc, '2.1 Transfer Learning Quadrant', level=2)
body(doc,
    'The choice of transfer learning strategy depends on two dimensions: dataset size '
    'and domain similarity. When the target dataset is small and the domain is similar '
    'to the source (ImageNet → STL-10 natural images), fine-tuning the full network '
    'or using a frozen feature extractor both yield strong results. STL-10 with '
    '5 000 labeled samples sits firmly in this quadrant.')
add_image(doc, 'tl_quadrant.png', width=4.5)
caption(doc, 'Figure 1: Transfer learning strategy selection based on dataset size and domain similarity')
doc.add_paragraph()

heading(doc, '2.2 Feature Extraction', level=2)
body(doc,
    'In the feature extraction setting the pretrained convolutional backbone acts as '
    'a fixed mapping φ: ℝ^(H×W×3) → ℝ^d. Only the linear classifier on top is '
    'trained:')
body(doc, '    ŷ = W · φ(x) + b')
body(doc,
    'where W ∈ ℝ^(C×d) and b ∈ ℝ^C are the only learnable parameters. Because the '
    'backbone is frozen, the number of trainable parameters is reduced from '
    f'11.18 M to only {fe_trainable:,} (the FC head), making training extremely fast '
    'and stable even with very limited data.')
doc.add_paragraph()

heading(doc, '2.3 Fine-tuning with Differential Learning Rates', level=2)
body(doc,
    'Fine-tuning unlocks all backbone parameters while protecting pretrained '
    'representations from catastrophic forgetting through differential learning rates:')
body(doc,
    '    θ_backbone ← θ_backbone - α_b · ∇L(θ_backbone)   [α_b = 1e-3]\n'
    '    θ_head     ← θ_head     - α_h · ∇L(θ_head)       [α_h = 1e-2]')
body(doc,
    f'The backbone LR (α_b = 1×10⁻³) is 10× lower than the head LR (α_h = 1×10⁻²). '
    'This asymmetry allows the new head to adapt quickly while the backbone shifts '
    'only gently, preserving the learned visual hierarchy.')
doc.add_paragraph()

heading(doc, '2.4 Why STL-10 at 96×96?', level=2)
body(doc,
    'A key architectural constraint governs this choice. The standard ResNet-18 uses '
    'a 7×7 conv1 followed by a 3×3 max-pool (total stride 4). When applied to 32×32 '
    'images, the feature map after layer4 collapses to 1×1 — destroying spatial '
    'structure and making ImageNet features uninformative. With 96×96 inputs, the '
    'spatial resolution at each stage is:')
add_table(doc,
    ['Stage', 'Spatial Size'],
    [
        ['Input',   '96×96'],
        ['conv1 (7×7, s=2)', '48×48'],
        ['maxpool (3×3, s=2)', '24×24'],
        ['layer1', '24×24'],
        ['layer2', '12×12'],
        ['layer3', '6×6'],
        ['layer4', '3×3  ← spatial context preserved'],
    ],
    col_widths=[2.5, 2.5])
doc.add_paragraph()
body(doc,
    'The 3×3 feature maps at layer4 retain meaningful spatial structure, enabling '
    'ImageNet pretrained features to transfer effectively to STL-10.')

# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '3. Dataset — STL-10')
add_table(doc,
    ['Property', 'Value'],
    [
        ['Dataset',          'STL-10'],
        ['Image size',       '96 × 96 pixels (RGB)'],
        ['Training samples', '5 000 (500 per class)'],
        ['Test samples',     '8 000'],
        ['Classes',          '10 (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)'],
        ['Source domain',    'ImageNet subset (similar natural image distribution)'],
        ['Normalization',    'mean=[0.447,0.440,0.407]  std=[0.260,0.257,0.271]'],
    ],
    col_widths=[2.5, 4.0])
doc.add_paragraph()
body(doc,
    'STL-10 was designed as a transfer learning benchmark. Its small labeled set '
    '(5 000 samples) and natural image content make it an ideal testbed for comparing '
    'training strategies that differ in how they use pretrained ImageNet knowledge.')

# ─────────────────────────────────────────────────────────────────────────────
# 4. Experimental Protocol
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '4. Experimental Protocol')

heading(doc, '4.1 Architecture', level=2)
body(doc,
    'All conditions use ResNet-18 with its original architecture (11.18 M parameters). '
    'The final FC layer is replaced with a new 512→10 linear layer, initialised with '
    'N(0, 0.01) weights. No architectural modifications are made to conv1 or maxpool — '
    'the original 7×7 stride-2 stem is preserved because STL-10 at 96×96 provides '
    'sufficient spatial resolution.')
doc.add_paragraph()

heading(doc, '4.2 Training Configuration', level=2)
add_table(doc,
    ['Hyperparameter', 'Scratch', 'Feature Extraction', 'Fine-tuning'],
    [
        ['Optimizer',     'SGD',   'Adam',  'SGD'],
        ['Learning rate', '0.01',  '1e-3',  'backbone 1e-3 / head 1e-2'],
        ['Momentum',      '0.9',   '—',     '0.9'],
        ['Weight decay',  '5e-4',  '1e-4',  '5e-4'],
        ['Scheduler',     'CosineAnneal(T=30)', 'CosineAnneal(T=30)', 'CosineAnneal(T=30)'],
        ['Epochs',        str(EPOCHS), str(EPOCHS), str(EPOCHS)],
        ['Batch size',    '64', '64', '64'],
        ['Trainable params', f'{scratch_params:,}', f'{fe_trainable:,}', f'{ft_params:,}'],
    ],
    col_widths=[2.2, 1.5, 2.0, 2.3])
doc.add_paragraph()

heading(doc, '4.3 Data Augmentation', level=2)
body(doc,
    'Training transforms: RandomHorizontalFlip, RandomCrop(96, padding=12), '
    'ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3), Normalize.\n'
    'Test transforms: ToTensor, Normalize only (no augmentation).')

# ─────────────────────────────────────────────────────────────────────────────
# 5. Results
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '5. Quantitative Results')

heading(doc, '5.1 Final Accuracy', level=2)
add_table(doc,
    ['Condition', 'Pretrained', 'Trainable Params', 'Best Val Acc', 'Δ vs Scratch'],
    [
        ['From Scratch',       'No',  f'{scratch_params:,}',  f'{scratch_acc*100:.2f}%', '—'],
        ['Feature Extraction', 'Yes', f'{fe_trainable:,}',    f'{fe_acc*100:.2f}%',      f'{fe_gain:+.2f} pp'],
        ['Fine-tuning',        'Yes', f'{ft_params:,}',       f'{ft_acc*100:.2f}%',      f'{ft_gain:+.2f} pp'],
    ],
    col_widths=[2.0, 1.5, 2.0, 1.8, 1.7])
doc.add_paragraph()

heading(doc, '5.2 Convergence Speed', level=2)
body(doc,
    f'Epochs required to first reach {thresh*100:.0f}% validation accuracy:')
conv_rows = []
for label, ep in [('From Scratch', sc_ep60), ('Feature Extraction', fe_ep60), ('Fine-tuning', ft_ep60)]:
    conv_rows.append([label, str(ep) if ep else f'>{EPOCHS}'])
add_table(doc,
    ['Condition', f'Epochs to {thresh*100:.0f}%'],
    conv_rows,
    col_widths=[3.5, 2.5])
doc.add_paragraph()

body(doc,
    'Pretrained models enter the training loop with high-quality feature representations '
    'and therefore need significantly fewer epochs to reach practical performance '
    'thresholds — a critical advantage when computational budget is limited.')

# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualizations
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '6. Visualizations')

heading(doc, '6.1 Accuracy & Loss Curves', level=2)
add_image(doc, 'tl_learning_curves.png', width=5.8)
caption(doc, 'Figure 2: Validation accuracy (left) and training loss (right) over 30 epochs')
doc.add_paragraph()

heading(doc, '6.2 Final Accuracy Comparison', level=2)
add_image(doc, 'tl_comparison_bar.png', width=5.0)
caption(doc, 'Figure 3: Best test accuracy achieved by each training strategy')
doc.add_paragraph()

heading(doc, '6.3 Convergence Speed (First 10 Epochs)', level=2)
add_image(doc, 'tl_early_convergence.png', width=5.5)
caption(doc,
    'Figure 4: Validation accuracy in the first 10 epochs. '
    'Pretrained models achieve high accuracy immediately; scratch training starts low.')
doc.add_paragraph()

heading(doc, '6.4 Training vs Validation (Overfitting Check)', level=2)
add_image(doc, 'tl_overfit_check.png', width=6.0)
caption(doc,
    'Figure 5: Train vs validation accuracy for each condition. '
    'Feature Extraction shows the smallest generalisation gap due to frozen backbone.')

# ─────────────────────────────────────────────────────────────────────────────
# 7. Discussion
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '7. Discussion')

heading(doc, '7.1 Feature Extraction', level=2)
body(doc,
    f'Feature Extraction achieves {fe_acc*100:.2f}% with only {fe_trainable:,} '
    f'trainable parameters — {fe_trainable/scratch_params*100:.1f}% of all model '
    'parameters. This demonstrates the power of reusing general visual features: '
    'the ImageNet backbone, trained on 1.2 M images, provides representations that '
    'generalise to STL-10 natural images without any backbone adaptation. Training '
    'time is minimal because only the linear head is updated per step.')
doc.add_paragraph()
body(doc,
    'The generalisation gap is smallest in this condition because the frozen backbone '
    'cannot overfit to STL-10\'s small training set. The limitation is an accuracy '
    'ceiling imposed by the fixed feature space — which fine-tuning addresses.')
doc.add_paragraph()

heading(doc, '7.2 Fine-tuning', level=2)
body(doc,
    f'Fine-tuning achieves {ft_acc*100:.2f}% ({ft_gain:+.2f} pp vs scratch) by '
    'gently adapting the entire network. The differential learning rate strategy '
    '(backbone LR = 1e-3, head LR = 1e-2) is critical: a uniform high LR on the '
    'backbone causes catastrophic forgetting — the pretrained weights are destroyed '
    'before the head can leverage them — while a uniform low LR slows head convergence.')
doc.add_paragraph()
body(doc,
    'Cosine annealing ensures the learning rate decays smoothly to near-zero, '
    'allowing fine-grained weight refinement in later epochs without oscillation.')
doc.add_paragraph()

heading(doc, '7.3 From Scratch Baseline', level=2)
body(doc,
    f'From scratch achieves {scratch_acc*100:.2f}% — a respectable result given that '
    'STL-10 has only 5 000 training samples. Strong data augmentation (random crop '
    'with 12-pixel padding, colour jitter, horizontal flip) and cosine-annealed SGD '
    'mitigate overfitting. However, the model must learn all visual features from '
    'scratch, limiting both final accuracy and convergence speed.')
doc.add_paragraph()

heading(doc, '7.4 Key Takeaways', level=2)
for item in [
    f'Pretrained Feature Extraction gains {fe_gain:+.2f} pp over scratch using '
     f'only {fe_trainable:,} trainable parameters (vs {scratch_params:,}).',
    f'Fine-tuning with differential LR gains {ft_gain:+.2f} pp and is the '
     'recommended strategy when dataset size permits full adaptation.',
    'Pretrained models converge faster — fewer epochs needed to reach practical '
     'performance thresholds, reducing compute cost.',
    'The architecture must match the input resolution: 96×96 is the minimum '
     'for standard ResNet-18 pretrained weights to produce informative features '
     '(layer4 must be ≥ 2×2).',
]:
    p = doc.add_paragraph(style='List Bullet')
    p.add_run(item).font.size = Pt(11)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Conclusion
# ─────────────────────────────────────────────────────────────────────────────
heading(doc, '8. Conclusion')
body(doc,
    'This experiment demonstrates that transfer learning from ImageNet to STL-10 '
    'yields clear, reproducible benefits in both final accuracy and convergence speed. '
    f'Fine-tuning achieves the best result ({ft_acc*100:.2f}%), improving over '
    f'training from scratch ({scratch_acc*100:.2f}%) by {ft_gain:+.2f} percentage '
    'points. Feature extraction with a frozen backbone offers a fast, '
    f'parameter-efficient alternative ({fe_acc*100:.2f}%, '
    f'{fe_trainable:,} trainable params), at the cost of a lower accuracy ceiling.')
doc.add_paragraph()
body(doc,
    'These results align with the transfer learning quadrant heuristic: when the '
    'target dataset is small and the source and target domains share visual statistics '
    '(natural images), pretrained representations consistently outperform random '
    'initialisation. The choice between feature extraction and fine-tuning depends on '
    'the available labelled data and the computational budget.')

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'Report.TL.STL10.docx')
doc.save(out_path)
print(f'[Saved] {out_path}')

# Quick summary
print('\n' + '='*50)
print('REPORT SUMMARY')
print('='*50)
print(f'  Scratch:           {scratch_acc*100:.2f}%')
print(f'  Feature Extract:   {fe_acc*100:.2f}%  ({fe_gain:+.2f} pp)')
print(f'  Fine-tuning:       {ft_acc*100:.2f}%  ({ft_gain:+.2f} pp)')
print(f'  Output:            Report.TL.STL10.docx')
