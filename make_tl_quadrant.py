"""Quick script to generate the TL quadrant diagram."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(7, 5))

# Quadrant backgrounds
ax.fill_between([-0.05, 0.5], [0.5, 0.5], [1.05, 1.05], color='#d5e8d4', alpha=0.7)  # top-left: FE
ax.fill_between([0.5, 1.05], [0.5, 0.5], [1.05, 1.05], color='#dae8fc', alpha=0.7)  # top-right: FT
ax.fill_between([-0.05, 0.5], [-0.05, -0.05], [0.5, 0.5], color='#fff2cc', alpha=0.7)  # bot-left: FE
ax.fill_between([0.5, 1.05], [-0.05, -0.05], [0.5, 0.5], color='#f8cecc', alpha=0.7)  # bot-right: new model

# Quadrant labels
ax.text(0.25, 0.80, 'Feature Extraction\nor Fine-tuning', ha='center', va='center',
        fontsize=10, color='#2d7a27', fontweight='bold')
ax.text(0.75, 0.80, 'Fine-tuning\n(Recommended)', ha='center', va='center',
        fontsize=10, color='#1a5f9e', fontweight='bold')
ax.text(0.25, 0.25, 'Fine-tune carefully', ha='center', va='center',
        fontsize=10, color='#7d6b00', fontweight='bold')
ax.text(0.75, 0.25, 'Feature Extraction\n(this experiment)', ha='center', va='center',
        fontsize=10, color='#2d7a27', fontweight='bold')

# Dividing lines
ax.axhline(0.5, color='gray', linewidth=1.5, linestyle='--')
ax.axvline(0.5, color='gray', linewidth=1.5, linestyle='--')

# Our experiment point — STL-10: Small dataset (5 000 samples) + Similar domain
ax.scatter([0.75], [0.25], s=200, color='#2c3e50', zorder=5)
ax.annotate('This experiment ↑\n(STL-10, ImageNet→Natural)',
            xy=(0.75, 0.25), xytext=(0.42, 0.10),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=9, color='black')

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('Domain Similarity (source → target)', fontsize=11)
ax.set_ylabel('Dataset Size (target)', fontsize=11)
ax.set_xticks([0.25, 0.75])
ax.set_xticklabels(['Different', 'Similar'])
ax.set_yticks([0.25, 0.75])
ax.set_yticklabels(['Small', 'Large'])
ax.set_title('Transfer Learning Strategy Selection\n(Tan & Pan, 2010)', fontweight='bold')

plt.tight_layout()
plt.savefig('tl_quadrant.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved tl_quadrant.png')
