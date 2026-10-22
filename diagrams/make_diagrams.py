"""Generate the paper's diagram figures: study overview and system
architecture. Writes PDF (vector, for LaTeX) and PNG.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

rcParams.update({'font.family': 'serif',
                 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                 'font.size': 7})

C = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
     'red': '#D55E00', 'purple': '#CC79A7', 'sky': '#56B4E9',
     'grey': '#8a8a8a', 'pale': '#eef2f5', 'ink': '#111111'}

# (text, box) pairs; save() refuses to write a figure whose text spills out
# of its box, which is how an earlier architecture figure shipped
FIT = []


def arrow(ax, p, q, color=None, lw=0.9, style='-|>', rad=0.0, ms=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color or C['ink'],
                                 connectionstyle=f'arc3,rad={rad}',
                                 shrinkA=1, shrinkB=1))


def card(ax, x, y_top, w, title, body, lh, th, fst, fsb, h=None,
         ec=C['ink'], fc='white'):
    """Rounded box with a bold title over a body. lh and th are the body and
    title line heights in data units. Given h, the text is centred in it."""
    n = body.count('\n') + 1
    content = th + n * lh
    h = h or content + 0.9 * lh
    p = ax.add_patch(FancyBboxPatch((x, y_top - h), w, h,
                                    boxstyle='round,pad=0.0,rounding_size=0.8',
                                    facecolor=fc, edgecolor=ec,
                                    linewidth=0.8 if ec == C['ink'] else 1.1))
    top = y_top - (h - content) / 2
    t1 = ax.text(x + w / 2, top, title, ha='center', va='top',
                 fontsize=fst, weight='bold', color=ec)
    t2 = ax.text(x + w / 2, top - th, body, ha='center', va='top',
                 fontsize=fsb, linespacing=1.25, color=C['ink'])
    FIT.extend([(t1, p), (t2, p)])
    return h


def save(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for t, p in FIT:
        if t.figure is not fig:
            continue
        tb, pb = t.get_window_extent(r), p.get_window_extent(r)
        if tb.x0 < pb.x0 or tb.x1 > pb.x1 or tb.y0 < pb.y0 or tb.y1 > pb.y1:
            bad.append(t.get_text())
    if bad:
        raise ValueError(f'{name}: text overflows its box: {bad}')
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight',
                pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300,
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('saved', name)


# ==========================================================  1. OVERVIEW
# Single column, read as a snake: row 1 left to right, row 2 right to left,
# row 3 left to right. Three short rows instead of one tall column.
W, H = 3.4, 2.13
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, H / W * 100); ax.axis('off')

D1 = dict(lh=3.15, th=3.3, fst=6.3, fsb=6.0)
X = [0.8, 34.6, 68.4]
CW = 30.8
R1, R2, R3 = 12.4, 18.7, 19.8          # row heights
GAP = 5.0
y1 = H / W * 100 - 0.8
y2 = y1 - R1 - GAP
y3 = y2 - R2 - GAP

card(ax, X[0], y1, CW, 'Dataset', '14,889 tweets\n6 classes [1], [2]',
     h=R1, fc=C['pale'], **D1)
card(ax, X[1], y1, CW, '1. Measure leakage',
     'class term in 15.2 to\n62.6% of its own class', h=R1, **D1)
card(ax, X[2], y1, CW, '2. Mask the vocabulary',
     'M1 class terms, M2\n+ frames, M3 + queries', h=R1, **D1)

card(ax, X[2], y2, CW, '3. Re-train 7 models',
     'classical ML on binary\nand 6-class subtype', h=R2, **D1)
card(ax, X[1], y2, CW, 'Two protocols',
     'P-sym: masked train\nand test\nP-dep: clean train,\nmasked test',
     h=R2, ec=C['blue'], fc='#f1f8fc', **D1)
card(ax, X[0], y2, CW, '4. Quantify',
     'drop, keyword floor,\nleakage vs fragility', h=R2, **D1)

half = (R3 - 1.2) / 2
card(ax, X[0], y3, CW, 'Binary: robust', '0.971 macro-F1 at M3', h=half,
     ec=C['green'], fc='#e8f5ef', **D1)
card(ax, X[0], y3 - half - 1.2, CW, 'Subtype: fragile', '0.704 macro-F1 at M3',
     h=half, ec=C['red'], fc='#fdeee6', **D1)
card(ax, X[1], y3, X[2] + CW - X[1], 'Recommendation',
     'report a masked condition and a keyword\nfloor with any headline number',
     h=R3, fc=C['pale'], **D1)

m1, m2 = y1 - R1 / 2, y2 - R2 / 2
arrow(ax, (X[0] + CW, m1), (X[1], m1))
arrow(ax, (X[1] + CW, m1), (X[2], m1))
arrow(ax, (X[2] + CW / 2, y1 - R1), (X[2] + CW / 2, y2))
arrow(ax, (X[2], m2), (X[1] + CW, m2))
arrow(ax, (X[1], m2), (X[0] + CW, m2))
arrow(ax, (X[0] + CW / 2, y2 - R2), (X[0] + CW / 2, y3))
for yc in (y3 - half / 2, y3 - half - 1.2 - half / 2):
    arrow(ax, (X[0] + CW, yc), (X[1], yc))

save(fig, 'D1_overview')


# ==========================================================  2. ARCHITECTURE
# Double column. Each stage is a frame, and one arrow joins each pair of
# frames: every masking tier feeds every feature set and every model, so
# per-box arrows would state a false pairing. The axes fills the figure so
# one data unit is the same length on both axes (0.0716 in).
W, H = 7.16, 2.95
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(0, H / W * 100); ax.axis('off')

D2 = dict(lh=1.72, th=1.85, fst=6.9, fsb=6.6)
FY0, FY1 = 1.2, 36.8                        # stage frame bottom and top


def stage(header, x0, x1, cards, note=None, gap=1.2, chain=False):
    ax.add_patch(FancyBboxPatch((x0, FY0), x1 - x0, FY1 - FY0,
                                boxstyle='round,pad=0.0,rounding_size=1.2',
                                facecolor='#f7f9fb', edgecolor='#c5ccd3',
                                linewidth=0.7))
    ax.text((x0 + x1) / 2, FY1 + 2.5, header, ha='center', va='center',
            fontsize=7.6, weight='bold', color=C['grey'])
    note_h = (note.count('\n') + 1) * 1.45 + 1.2 if note else 0
    hs = [D2['th'] + (c[1].count('\n') + 1) * D2['lh'] + 0.9 * D2['lh'] for c in cards]
    total = sum(hs) + gap * (len(cards) - 1)
    top = FY0 + note_h + (FY1 - FY0 - note_h + total) / 2
    prev_bottom = None
    for (title, body, *style), h in zip(cards, hs):
        ec, fc = (list(style) + [C['ink'], 'white'][len(style):])[:2]
        card(ax, x0 + 1.0, top, x1 - x0 - 2.0, title, body, ec=ec, fc=fc, **D2)
        if chain and prev_bottom is not None:
            arrow(ax, ((x0 + x1) / 2, prev_bottom), ((x0 + x1) / 2, top),
                  color=C['grey'], ms=6)
        prev_bottom = top - h
        top -= h + gap
    if note:
        ax.text((x0 + x1) / 2, FY0 + 0.9, note, ha='center', va='bottom',
                fontsize=5.9, style='italic', color=C['grey'], linespacing=1.25)


STAGES = [
    ('DATA', 0.5, 17.0, [
        ('Source', 'Nusrat et al. [1], [2]\n14,996 English tweets\nCC BY 4.0',
         C['ink'], C['pale']),
        ('Cleaning', '13 null or empty rows\n94 near-duplicates\nn = 14,889'),
        ('Label views', 'binary: control vs.\nany of 5 subtypes\nsubtype: 6 classes'),
    ], None, 2.0, True),
    ('MASKING', 19.3, 34.3, [
        ('M0', 'no masking', C['blue']),
        ('M1', 'class-naming terms\n+ depress*', C['sky']),
        ('M2', 'M1 + disclosure\nframes, drug names', C['orange']),
        ('M3', 'M2 + published\nquery phrases', C['purple']),
    ], 'each span becomes\nk filler tokens'),
    ('FEATURES', 36.6, 56.6, [
        ('v1 feature set', 'word TF-IDF, 1-2 grams, 10k\n+ 13 dense features:\n'
                           'length (2), TextBlob (2),\nPHQ-9 symptom lexicon (9)'),
        ('v2 feature set', 'v1 + char TF-IDF, 3-5 grams,\n20k, on mask-stripped text\n'
                           '+ 10 dense features:\nVADER (4), first-person,\n'
                           'absolutist and negation\nratios, !, ?, capitals',
         C['green'], '#f3faf7'),
    ], 'vectoriser and scaler are fit\ninside each training fold'),
    ('MODELS', 58.9, 75.4, [
        ('Main grid (v1)', 'Logistic Regression\nLinear SVM, RBF SVM\nRandom Forest\n'
                           'Decision Tree\nComplement NB, k-NN'),
        ('v1 + class weighting', 'Logistic Regression\nLinear SVM'),
        ('v2 + class weighting', 'Logistic Regression\nLinear SVM\nHist. Gradient Boosting',
         C['green'], '#f3faf7'),
    ], '138 configurations,\n5 folds each', 1.0),
    ('EVALUATION', 77.7, 99.5, [
        ('Validation and protocols', 'stratified 5-fold CV, seed 42\n'
                                     'out-of-fold predictions stored\n'
                                     'P-sym: train = test tier\n'
                                     'P-dep: train M0, test M1 to M3'),
        ('Metrics', 'macro-F1, accuracy,\nbalanced accuracy, P / R / F1,\n'
                    'ROC-AUC, PR-AUC, MCC, kappa,\nconfusion matrices'),
        ('Tests and floors', 'paired t-test, Wilcoxon\nkeyword rule, majority',
         C['red'], '#fdf3ef'),
    ], None, 1.0),
]

for header, x0, x1, cards, *rest in STAGES:
    stage(header, x0, x1, cards, *rest)

mid = (FY0 + FY1) / 2
for (_, _, x1, *_), (_, x0, *_) in zip(STAGES, STAGES[1:]):
    arrow(ax, (x1 + 0.15, mid), (x0 - 0.15, mid), color=C['grey'], lw=1.4, ms=10)

save(fig, 'D2_architecture')

print('\nall diagrams written to', OUT)
