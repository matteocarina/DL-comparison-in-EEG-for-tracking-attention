"""Shared matplotlib styling: palette, fonts and save helpers used by every figure."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAVY = '#14274E'
STEEL = '#4A6FA5'
GOLD = '#B08D49'
GRAY = '#7A8699'
CRIMSON = '#9B2226'
INK = '#1B2430'
MUTED = '#6B7683'
FAINT = '#D8DEE6'
PAPER = '#FFFFFF'

DPI = 220

def apply_style():
    plt.rcParams.update({
        'figure.facecolor': PAPER,
        'axes.facecolor': PAPER,
        'savefig.facecolor': PAPER,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.edgecolor': MUTED,
        'axes.labelcolor': INK,
        'axes.titlecolor': INK,
        'axes.linewidth': .9,
        'axes.grid': False,
        'text.color': INK,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.frameon': False,
        'legend.fontsize': 10,
        'figure.autolayout': False,
    })

def grid(ax, axis='y'):
    ax.grid(axis=axis, color=FAINT, lw=.7, zorder=0)
    ax.set_axisbelow(True)

def despine(ax, keep=('left', 'bottom')):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in keep)

def save(fig, path):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")
