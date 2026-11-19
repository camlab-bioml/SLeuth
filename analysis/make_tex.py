#!/usr/bin/env python3
"""Build the revised LaTeX tables for paper/main.tex from the run's result CSVs.

Run 20261113T113339-1225758, PCA arm none unless a table says otherwise.
Every table body is red (all its numbers are new); captions mark the new
parts with \\rev{}. Output: one .tex file per table in OUT.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REV = HERE
OUT = os.environ.get('TEX_OUT', os.path.join(HERE, 'tex_out'))
os.makedirs(OUT, exist_ok=True)


def rd(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def num(x, d=3):
    s = f'{float(x):.{d}f}'
    if s.startswith('-'):
        if float(s) == 0:
            return s[1:]
        return '$-$' + s[1:]
    return s


def sgn(x, d=3):
    s = f'{float(x):+.{d}f}'
    if float(s) == 0:
        return f'{0:.{d}f}'
    return s.replace('-', '$-$')


def ci(lo, hi):
    return f'[{num(lo)}, {num(hi)}]'


def thou(x):
    return f'{int(round(float(x))):,}'.replace(',', '{,}')


def pfmt(p):
    p = float(p)
    if p < 0.001:
        return '$<$0.001'
    return f'{p:.3f}'


def pval(p):
    p = float(p)
    return '$p<0.001$' if p < 0.001 else f'$p={p:.3f}$'


def withp(v, p, extra=''):
    """Cell with the value on top and (extra +) Holm-adjusted p underneath."""
    return (r'\begin{tabular}[c]{@{}c@{}}' + v + r'\\[-1.5pt]{\scriptsize ' + extra
            + pval(p) + r'}\end{tabular}')


def bold(v):
    return r'\textbf{' + v + '}'


def write(name, text):
    with open(os.path.join(OUT, name), 'w') as fh:
        fh.write(text)
    print(f'wrote {name}')


TIERS = ['I', 'II', 'III']
CV = {'I': 'cv1', 'II': 'cv2', 'III': 'cv3'}
DAG = r'$^{\dagger}$'

# ---------------------------------------------------------------- Table 1
t1 = [r for r in rd(REV + '/tiers_significance/out/table1_none.csv')]
sig = rd(REV + '/tiers_significance/out/sig_foldpaired.csv')
holm_peers = {(r['tier'], r['B'], r['metric']): float(r['p_t_holm'])
              for r in sig if r['pca_arm'] == 'none'
              and r['family'] == 'vs_peers:siamese_sl_pd_pooled'}
NAME = {'siamese_sl_pd_pooled': 'SLeuth', 'siamese_sl': 'SLeuth-CL',
        'siamese_sl_allmod_pd_pooled': 'SLeuth-all',
        'siamese_sl_allmod': 'SLeuth-all-CL', 'SLMGAE': 'SLMGAE',
        'SLGNN': 'SLGNN', 'NSF4SL': 'NSF4SL', 'ELISL': 'ELISL',
        'BilinearProbe': 'Bilinear probe', 'DegreePrior': 'Degree prior'}
PEERS = ['SLMGAE', 'SLGNN', 'NSF4SL', 'ELISL', 'BilinearProbe', 'DegreePrior']
METRICS = ['auroc', 'ap', 'auprg']
missing = [(t, b, m) for t in TIERS for b in PEERS for m in METRICS
           if (t, b, m) not in holm_peers]
assert not missing, missing


def params(r):
    m, pm, psd = r['model'], float(r['params_mean']), float(r['params_sd'])
    if m == 'SLMGAE':
        return r'$\approx$254M'
    if m == 'SLGNN':
        return r'$\approx$5.2M'
    if m == 'NSF4SL':
        return r'$\approx$470k'
    if m == 'ELISL':
        return r'$\approx$%dk (%dk)' % (round(pm / 1000), round(psd / 1000))
    if m == 'DegreePrior':
        return '0'
    s = thou(pm)
    if psd > 0.5:
        s += f' ({thou(psd)})'
    return s


lines = []
for tier in TIERS:
    rows = [r for r in t1 if r['tier'] == tier]
    assert [r['model'] for r in rows] == list(NAME), [r['model'] for r in rows]
    best = {m: max(round(float(r[m + '_mean']), 3) for r in rows
                   if r['model'] != 'DegreePrior') for m in METRICS}
    for i, r in enumerate(rows):
        cells = []
        for m in METRICS:
            v = num(r[m + '_mean'])
            if round(float(r[m + '_mean']), 3) == best[m] and r['model'] != 'DegreePrior':
                v = r'\textbf{' + v + '}'
            if r['model'] in PEERS and holm_peers[(tier, r['model'], m)] < 0.05:
                v += DAG
            v = f"{v} ({float(r[m + '_sd']):.3f})"
            if r['model'] in PEERS:
                v = withp(v, holm_peers[(tier, r['model'], m)])
            cells.append(v)
        lead = (r'\multirow{16}{*}{%s}' % tier + '\n') if i == 0 else ''
        lines.append(f"{lead}  & {NAME[r['model']]} & {params(r)} & "
                     + ' & '.join(cells) + r' \\')
    lines.append(r'\hline')
body = '\n'.join(lines)
write('tab1.tex', r"""\begin{table}[t]
\centering
\small
{\color{red}
\begin{tabular}{ll|cccc}
\hline
Tier & Model & \multicolumn{1}{r}{Trainable parameters} & AUROC & AUPR & AUPRG \\
\hline
""" + body + r"""
\end{tabular}}
\caption{\textbf{Performance across three evaluation tiers of increasing
difficulty.}
AUROC, AUPR, and AUPRG on SynlethDB 3.0 evaluated for SLeuth \rev{and its
variants (-CL: cell-line heads; -all: all $16$ embeddings, no selection), four
existing models, a bilinear probe on SLeuth's own features, and a degree prior
(control) across Tiers I--III. Mean (standard deviation) over the five
training/testing splits; bold marks the best mean in each column within a tier
(control excluded). $^{\dagger}$: SLeuth is higher, fold-paired $t$-test
$p<0.05$ after Holm correction over the six comparators. Every comparator is
below SLeuth on all five splits of every tier and metric, and in the paired
bootstrap ($p\le0.002$). SLeuth's parameter count varies with the modalities
selected in each split. $92.6\%$ of SLMGAE's parameters form a gene-by-gene
attention block that receives gradients only at training edges (about $10.7$M
are effectively trained); ELISL's count is the values of its fitted gradient-boosted trees.}
}
\label{tab:tiers}
\end{table}
""")

# ---------------------------------------------------------------- Table 2 (+ appendix cell-line version)
solo = [r for r in rd(REPO + '/sl_comparison/results/paper_tables/tab_solo.csv')
        if r['pca_arm'] == 'none' and r['in_paper'] == 'True']
paired = rd(REV + '/pca_solo_loco_sel/s2_solo_vs_multimodal_paired.csv')
t1j = rd(REV + '/tiers_significance/out/table1_joint.csv')
ORDER = [('expression', ['geneformer', 'scgpt', 'gene2vec']),
         ('protein\\_seq', ['esm2', 'esm1b', 'prot_t5']),
         ('text', ['text_embed', 'bioconceptvec']),
         ('ppi', ['node2vec_ppi', 'ppi_svd', 'ppi_raw', 'mashup']),
         ('go', ['go', 'go2vec', 'onto2vec']),
         ('kg', ['kg_complex'])]


def tex_name(e):
    return e.replace('_', r'\_')


def solo_table(variant, combo_model, allmod_model, label, caption):
    get = {(r['embedding'], r['tier']): r for r in solo if r['variant'] == variant}
    below = {}
    for r in paired:
        if r['arm'] == 'none' and r['variant'] == variant and r['comparator'] == 'combo':
            below[(r['embedding'], r['tier'])] = (float(r['mean_diff']), float(r['p_t_holm16']))
    out = []
    for cat, embs in ORDER:
        for j, e in enumerate(embs):
            cells = []
            for tier in TIERS:
                r = get[(e, tier)]
                v = num(r['auprg_any_mean'])
                if len(embs) > 1:
                    top = max(round(float(get[(x, tier)]['auprg_any_mean']), 3) for x in embs)
                    if round(float(r['auprg_any_mean']), 3) == top:
                        v = r'\textbf{' + v + '}'
                d, p = below[(e, tier)]
                if p < 0.05:
                    v += (DAG if d < 0 else r'$^{\ddagger}$')
                cells.append(withp(f"{v} ({float(r['auprg_any_sd']):.3f})", p))
            first = cat if j == 0 else ''
            out.append(f"{first:<13}& {tex_name(e):<15}& " + ' & '.join(cells) + r' \\')
    out.append(r'\hline')

    def mm(rows, model):
        return [r for r in rows if r['model'] == model]

    mrows = [('selected', mm(t1, combo_model)), ('all 16', mm(t1, allmod_model)),
             (r'selected, PCA $80\%$', mm(t1j, combo_model)),
             (r'all 16, PCA $80\%$', mm(t1j, allmod_model))]
    mtop = {t: max(round(float({r['tier']: r for r in rows}[t]['auprg_mean']), 3) for _, rows in mrows)
            for t in TIERS}
    for k, (nm, rows) in enumerate(mrows):
        byt = {r['tier']: r for r in rows}
        cells = []
        for t in TIERS:
            v = num(byt[t]['auprg_mean'])
            if round(float(byt[t]['auprg_mean']), 3) == mtop[t]:
                v = bold(v)
            cells.append(f"{v} ({float(byt[t]['auprg_sd']):.3f})")
        first = 'multi-modal' if k == 0 else ''
        out.append(f"{first:<13}& {nm:<15}& " + ' & '.join(cells) + r' \\')
    body = '\n'.join(out)
    return (r"""\begin{table}[t]
\centering
\small
{\color{red}
\begin{tabular}{ll|ccc}
\hline
\multicolumn{1}{l}{Category} & \multicolumn{1}{l}{Embedding} & Tier~I & Tier~II & Tier~III \\
\hline
""" + body + r"""
\hline
\end{tabular}}
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\end{table}
""")


write('tab2.tex', solo_table(
    'pd:pooled', 'siamese_sl_pd_pooled', 'siamese_sl_allmod_pd_pooled', 'tab:solo',
    r"""\textbf{Single-modality models, one per embedding, grouped by
biological category.} \rev{Pooled variant (the cell-line variant is in
Appendix Table~\ref{tab:solo-cl}).} AUPRG, mean (standard deviation) across the
five training/testing splits. Bold font marks the best embedding within a
category \emph{for that tier}\rev{; $^{\dagger}$: below the selected
multi-modal model (fold-paired $t$-test, Holm over the $16$ embeddings,
$p<0.05$); no embedding is significantly above it. The last four rows are
the multi-modal models without and with the principal component projection of
the submitted version.}"""))
write('tab_solo_cl.tex', solo_table(
    'pd:per_head', 'siamese_sl', 'siamese_sl_allmod', 'tab:solo-cl',
    r"""\rev{\textbf{Single-modality models, cell-line variant.} As
Table~\ref{tab:solo}, for SLeuth-CL (any-cell-line AUPRG, maximum over the
eight heads). $^{\dagger}$: below the selected multi-modal model
(fold-paired $t$-test, Holm over the $16$ embeddings, $p<0.05$).}"""))

# ---------------------------------------------------------------- Table 3 (LOCO)
loco = [r for r in rd(REPO + '/sl_comparison/results/paper_tables/tab_loco.csv')
        if r['pca_arm'] == 'none']
CATS = [('expression', 'expression'), ('go', 'go'), ('kg', 'kg'), ('ppi', 'ppi'),
        ('protein_seq', 'protein\\_seq'), ('text', 'text'),
        ('none', 'none (full model)')]
g = {(r['variant'], r['dropped'], r['tier']): r for r in loco}
lp = {(r['variant'], r['dropped'], r['tier']): float(r['p_t_holm6'])
      for r in rd(REV + '/pca_solo_loco_sel/s3_loco_paired.csv') if r['arm'] == 'none'}
ltop = {(v_, t_): max(round(float(g[(v_, k_, t_)]['auprg_any_mean']), 3) for k_, _ in CATS)
        for v_ in ['pd:pooled', 'pd:per_head'] for t_ in TIERS}
out = []
for key, disp in CATS:
    if key == 'none':
        out.append(r'\hline')
    cells = []
    for variant in ['pd:pooled', 'pd:per_head']:
        for tier in TIERS:
            r = g[(variant, key, tier)]
            v = num(r['auprg_any_mean'])
            if round(float(r['auprg_any_mean']), 3) == ltop[(variant, tier)]:
                v = bold(v)
            v = f"{v} ({float(r['auprg_any_sd']):.3f})"
            if key != 'none':
                v = withp(v, lp[(variant, key, tier)])
            cells.append(v)
    out.append(f"{disp:<18}& " + ' & '.join(cells) + r' \\')
body = '\n'.join(out)
write('tab3.tex', r"""\begin{table}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2.5pt}
{\color{red}
\begin{tabular}{l|ccc|ccc}
\hline
 & \multicolumn{3}{c|}{SLeuth} & \multicolumn{3}{c}{SLeuth-CL} \\
\multicolumn{1}{l|}{Category removed} & Tier~I & Tier~II & Tier~III & Tier~I & Tier~II & Tier~III \\
\hline
""" + body + r"""
\hline
\end{tabular}}
\caption{\textbf{Multi-modal model with one biological category removed.}
\rev{AUPRG}, mean (standard deviation) across the $5$ training/testing
splits; each row is a model retrained from scratch on the remaining five
categories, under the splits, architecture and preprocessing used for the full
model. \rev{The removed embedding is the category's selection in each split
(Appendix Table~\ref{tab:picks}). No removal changes AUPRG significantly at any
tier (fold-paired $t$-test, Holm over the six categories).}}
\label{tab:loco}
\end{table}
""")

# ---------------------------------------------------------------- Table 4 (cell line)
cl4 = {(r['tier'], r['condition']): r
       for r in rd(REPO + '/sl_comparison/results/paper_tables/tab_cellline.csv')
       if r['pca_arm'] == 'none' and r['variant'] == 'pd:per_head'}
p4 = {(r['A'], r['tier'], r['metric']): float(r['p_t_holm']) for r in sig
      if r['pca_arm'] == 'none' and r['family'] == 'table4_main' and r['B'] == 'siamese_sl'}
COND4 = [('included', 'included (SLeuth-CL)', None),
         ('removed (cell_emb=0)', 'zeroed at inference', 'siamese_sl_nocell'),
         ('retrained without cell lines (pd:pooled)', 'retrained without (SLeuth)', 'siamese_sl_pd_pooled')]
out4 = []
for tier in TIERS:
    top = {m: max(round(float(cl4[(tier, c)][m + '_mean']), 3) for c, _, _ in COND4) for m in METRICS}
    for j, (c, disp, model) in enumerate(COND4):
        r = cl4[(tier, c)]
        assert r['status'] == 'ok', r
        cells = []
        for m in METRICS:
            v = num(r[m + '_mean'])
            if round(float(r[m + '_mean']), 3) == top[m]:
                v = bold(v)
            v = f"{v} ({float(r[m + '_sd']):.3f})"
            if model:
                p = p4[(model, tier, m)]
                ch = sgn(r['delta_' + m + '_vs_included']) + (DAG if p < 0.05 else '')
                v = withp(v, p, extra=ch + '; ')
            cells.append(v)
        lead = (r'\multirow{5}{*}{%s}' % tier) if j == 0 else ''
        out4.append(f"{lead} & {disp} & " + ' & '.join(cells) + r' \\')
    out4.append(r'\hline')
src4 = (r"""{\color{red}
\begin{tabular}{ll|ccc}
\hline
Tier & Cell-line information & AUROC & AUPR & AUPRG \\
\hline
""" + '\n'.join(out4) + r"""
\end{tabular}}""")
write('tab4.tex', r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
""" + src4.strip() + r"""
\caption{\textbf{\rev{Cell-line context: zeroing the cell-line embedding of the
trained cell-line model at inference, or retraining without cell lines.}}
Mean (population standard deviation) across the $5$ splits, on the
same test edges and evaluator as Table \ref{tab:tiers}\rev{; small numbers are
the change from the included row, $^{\dagger}$: fold-paired $t$-test
$p<0.05$ after Holm correction over the two conditions.}}
\label{tab:cellline}
\end{table}
""")

# ---------------------------------------------------------------- Table 5 (zero-shot)
zs = [r for r in rd(REV + '/external_screens/out_none/table5_zero_shot.csv')
      if r['arm'] == 'none' and r['row_set'] == 'matched']
d5 = [r for r in rd(REV + '/external_screens/out_none/table5_diffs.csv')
      if r['arm'] == 'none' and r['row_set'] == 'matched']
clp = [r for r in rd(REV + '/external_screens/out_none/table5_cellline_minus_pooled.csv')
       if r['arm'] == 'none']
SCREENS = [('Adamson', r'Adamson $\gamma$'), ('AdamsonGI', 'Adamson GI'),
           ('Corn', 'Corn'), ('Gilbert', 'Gilbert')]
SPL = [('cv1', 'random pairs'), ('cv2', r'$\geq$1 gene unseen'), ('cv3', 'both unseen')]
M5 = [('siamese_sl_pd_pooled', 'SLeuth'), ('siamese_sl', 'SLeuth-CL'),
      ('SLMGAE', 'SLMGAE'), ('SLGNN', 'SLGNN'), ('NSF4SL', 'NSF4SL'),
      ('DegreePrior', 'Degree prior')]
zg = {(r['screen'], r['split'], r['model']): r for r in zs}
dg = {(r['screen'], r['split'], r['B']): r for r in d5 if r['A'] == 'siamese_sl_pd_pooled'}
cg = {(r['screen'], r['split']): r for r in clp}
out, out_ci, out_d = [], [], []
for sc, scd in SCREENS:
    for k, (cv, spd) in enumerate(SPL):
        vals = [float(zg[(sc, cv, m)]['mean']) for m, _ in M5]
        best = max(round(v, 3) for v in vals)
        cells, cells_ci = [], []
        for m, _ in M5:
            r = zg[(sc, cv, m)]
            v = num(r['mean'])
            if round(float(r['mean']), 3) == best:
                v = r'\textbf{' + v + '}'
            cells_ci.append(f"{v} {ci(r['ci_low'], r['ci_high'])}")
            sup = ''
            if r['ci_excludes_0'] == 'True':
                sup += r'\dagger'
            pz = None
            if m == 'siamese_sl':
                pz = float(cg[(sc, cv)]['p2_holm'])
            elif m != 'siamese_sl_pd_pooled':
                pz = float(dg[(sc, cv, m)]['p2_holm'])
            if pz is not None and pz < 0.05:
                sup += r'\ast'
            if sup:
                v += '$^{' + sup + '}$'
            if pz is not None:
                v = withp(v, pz)
            cells.append(v)
        first = scd if k == 0 else ''
        out.append(f"{first:<18}& {spd:<20}& " + ' & '.join(cells) + r' \\')
        out_ci.append(f"{first:<18}& {spd:<20}& {thou(zg[(sc, cv, 'siamese_sl_pd_pooled')]['n_rows'])} & "
                      + ' & '.join(cells_ci) + r' \\')
        dc = []
        for m in ['SLMGAE', 'SLGNN', 'NSF4SL', 'DegreePrior']:
            r = dg[(sc, cv, m)]
            dc.append(f"{sgn(r['delta'])} {ci(r['ci_low'], r['ci_high'])}; {float(r['p2_holm']):.3f}")
        r = cg[(sc, cv)]
        dc.append(f"{sgn(r['delta'])} {ci(r['ci_low'], r['ci_high'])}; {float(r['p2_holm']):.3f}")
        out_d.append(f"{first:<18}& {spd:<20}& " + ' & '.join(dc) + r' \\')
    out.append(r'\hline')
    out_ci.append(r'\hline')
    out_d.append(r'\hline')
write('tab5.tex', r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
{\color{red}
\begin{tabular}{ll|cccccc}
\hline
\multicolumn{1}{l}{Screen} & \multicolumn{1}{l}{Split} & SLeuth & SLeuth-CL & SLMGAE & SLGNN & NSF4SL & Degree prior \\
\hline
""" + '\n'.join(out) + r"""
\end{tabular}}
\caption{\textbf{Zero-shot transfer to \rev{four readouts of} three external screens.} Spearman correlation between each model's score outputs and the \rev{sign-flipped} screen phenotype \rev{(positive: SL-like pairs scored higher)}; mean across the five Tier~III fold models\rev{, on screen pairs absent from SynLethDB with both genes in the benchmark gene set. Bold: best in the row. $^{\dagger}$: $95\%$ bootstrap CI excludes $0$. $^{\ast}$: differs from SLeuth (for SLeuth-CL: from SLeuth) in the paired bootstrap, Holm-adjusted $p<0.05$. CIs and standard deviations are in Appendix Table~\ref{tab:external-ci}, differences in Appendix Table~\ref{tab:external-diff}.}}
\label{tab:external}
\end{table}
""")
write('tab5_ci.tex', r"""\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
{\color{red}
\begin{tabular}{ll|r|cccccc}
\hline
Screen & Split & $n$ & SLeuth & SLeuth-CL & SLMGAE & SLGNN & NSF4SL & Degree prior \\
\hline
""" + '\n'.join(out_ci) + r"""
\end{tabular}}
\caption{\rev{\textbf{Zero-shot Spearman correlations with $95\%$ CIs.} Mean over the five Tier~III fold models and percentile bootstrap CI of that mean ($B=1000$ resamples of screen pairs, the same resamples for every model); $n$: number of screen pairs. Pairs share genes, so these intervals are optimistic: a gene-level bootstrap widens them $1.8$--$16.1$-fold (median $3.1$).}}
\label{tab:external-ci}
\end{table}
""")
write('tab5_diff.tex', r"""\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
{\color{red}
\begin{tabular}{ll|ccccc}
\hline
Screen & Split & SLeuth $-$ SLMGAE & SLeuth $-$ SLGNN & SLeuth $-$ NSF4SL & SLeuth $-$ degree prior & SLeuth-CL $-$ SLeuth \\
\hline
""" + '\n'.join(out_d) + r"""
\end{tabular}}
\caption{\rev{\textbf{Zero-shot differences between models.} Difference in Spearman correlation [paired bootstrap $95\%$ CI]; Holm-adjusted two-sided $p$ (over the eight SLeuth-minus-comparator tests of a row for the first four columns, over the twelve rows for the last; the smallest attainable values are $0.016$ and $0.024$).}}
\label{tab:external-diff}
\end{table}
""")

# ---------------------------------------------------------------- Table 6 (fine-tuning) + appendix cell-line
ft = [r for r in rd(REV + '/external_screens/out_none/table6_finetune.csv')
      if r['arm'] == 'none' and r['row_set'] == 'all_sldb_filtered']
t6 = [r for r in rd(REV + '/external_screens/out_none/table6_tests.csv')
      if r['arm'] == 'none' and r['row_set'] == 'all_sldb_filtered' and r['contrast'] == 'Full - Scratch']


def ft_table(model, label, caption):
    fg = {(r['screen'], r['split'], r['method']): r for r in ft if r['model'] == model}
    tg = {(r['screen'], r['split']): r for r in t6 if r['model'] == model}
    out = []
    nb_pos = nb_neg = nf = 0
    for sc, scd in SCREENS:
        for k, (cv, spd) in enumerate(SPL):
            cells = []
            ftop = max(round(float(fg[(sc, cv, mm_)]['mean']), 3)
                       for mm_ in ['zero_shot_raw', 'LP_last', 'Full', 'Scratch'])
            for meth in ['zero_shot_raw', 'LP_last', 'Full', 'Scratch']:
                r = fg[(sc, cv, meth)]
                v = num(r['mean'])
                if round(float(r['mean']), 3) == ftop:
                    v = bold(v)
                if meth == 'Full':
                    t = tg[(sc, cv)]
                    if float(t['p2_holm']) < 0.05:
                        v += (r'$^{+}$' if float(t['delta']) > 0 else r'$^{-}$')
                        if float(t['delta']) > 0:
                            nb_pos += 1
                        else:
                            nb_neg += 1
                    if t['t_p_holm'] not in ('', 'nan') and float(t['t_p_holm']) < 0.05:
                        nf += 1
                v = f"{v} ({float(r['sd_pop']):.3f})"
                if meth == 'Full':
                    v = withp(v, tg[(sc, cv)]['p2_holm'])
                cells.append(v)
            first = scd if k == 0 else ''
            out.append(f"{first:<18}& {spd:<20}& " + ' & '.join(cells) + r' \\')
        out.append(r'\hline')
    print(f'{model}: Full-Scratch bootstrap-Holm sig + {nb_pos}, - {nb_neg}; fold-Holm sig {nf}')
    return (r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
{\color{red}
\begin{tabular}{ll|cccc}
\hline
\multicolumn{1}{l}{Screen} & \multicolumn{1}{l}{Split} & Zero-shot & \texttt{LP\_last} & \texttt{Full} & \texttt{Scratch} \\
\hline
""" + '\n'.join(out) + r"""
\end{tabular}}
\caption{""" + caption + r"""}
\label{""" + label + r"""}
\end{table}
""")


write('tab6.tex', ft_table('siamese_sl_pd_pooled', 'tab:external-ft', r"""\textbf{Fine-tuning strategies on small datasets.} Spearman correlation with
the \rev{sign-flipped} screen phenotype, mean (standard deviation) over the five fold
models \rev{of SLeuth}, on SynLethDB-filtered test pairs\rev{,} for zero-shot, last layer training (\texttt{LP\_last}), full retraining \rev{from the SynLethDB-trained model (\texttt{Full}), and the same training from a fresh initialization (\texttt{Scratch}). $^{+}$/$^{-}$: \texttt{Full} above/below \texttt{Scratch} in the paired bootstrap (Holm over the $12$ cells, $p<0.05$); across the five fold models only the Adamson $\gamma$, $\geq$1-gene-unseen difference ($+0.030$) remains significant after Holm correction. Zero-shot values differ slightly from Table~\ref{tab:external}, whose rows also require both genes to be benchmark genes. SLeuth-CL is in Appendix Table~\ref{tab:external-ft-cl}.}"""))
write('tab6_cl.tex', ft_table('siamese_sl', 'tab:external-ft-cl', r"""\rev{\textbf{Fine-tuning strategies, cell-line variant.} As Table~\ref{tab:external-ft}, for SLeuth-CL scored at the head matching each screen (K562; OTHER for Corn).}"""))

# ---------------------------------------------------------------- Table 7 (SynLethDB agreement)
ap = [r for r in rd(REV + '/item11_sldb_agreement/agreement_split_points.csv')
      if r['scope'] == 'all_pairs' and r['split'] == 'any']
ad = [r for r in rd(REV + '/item11_sldb_agreement/agreement_split_diffs.csv')
      if r['scope'] == 'all_pairs' and r['split'] == 'any'
      and r['contrast'] == 'own_line - other_line' and r['metric'] == 'auroc']
pg = {(r['screen'], r['definition'], r['subset']): r for r in ap}
dg7 = {r['screen']: r for r in ad}
smap = {r['screen']: r for r in rd(REV + '/item11_sldb_agreement/extra_same_map_contrast.csv')
        if r['subset'] == 'DIFF k562_same_map - jurkat_only_same_map'}
out, counts = [], []
for sc, scd in SCREENS:
    a = pg[(sc, 'published', 'all')]
    o = pg[(sc, 'context', 'own_line')]
    x = pg[(sc, 'context', 'other_line')]

    top7 = max(round(float(q['auroc']), 3) for q in (a, o, x) if q['reportable'] == 'True')

    def cell(r):
        if r['reportable'] != 'True':
            return '--'
        v = num(r['auroc'])
        if round(float(r['auroc']), 3) == top7:
            v = bold(v)
        return f"{v} {ci(r['auroc_ci_low'], r['auroc_ci_high'])}"

    dd = dg7.get(sc)
    dtxt = ('--' if dd is None or dd['reportable'] != 'True'
            else f"{sgn(dd['point'])} {ci(dd['ci_low'], dd['ci_high'])}")
    sm = smap.get(sc)
    stxt = ('--' if sm is None or sm['reportable'] != 'True'
            else f"{sgn(sm['auroc'])} {ci(sm['auroc_ci_low'], sm['auroc_ci_high'])}")
    counts.append(f"{scd} {thou(a['n_pos'])}/{thou(o['n_pos'])}/{thou(x['n_pos'])}/{thou(a['n_neg'])}")
    out.append(f"{scd:<18}& {cell(a)} & {cell(o)} & {cell(x)} & {dtxt} & {stxt}" + r' \\')
write('tab7.tex', r"""\begin{table}[t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.5pt}
{\color{red}
\begin{tabular}{l|ccc|cc}
\hline
 & \multicolumn{3}{c|}{AUROC, SynLethDB SL vs.\ screened non-SL} & \multicolumn{2}{c}{$\Delta$ AUROC} \\
Screen & all SL & SL in the screen's line & SL in other lines only & own $-$ other & K562 $-$ Jurkat-only, same map \\
\hline
""" + '\n'.join(out) + r"""
\hline
\end{tabular}}
\caption{\rev{\textbf{Agreement between SynLethDB labels and the external screens.} On the screen pairs that SynLethDB labels, AUROC of the sign-flipped screen phenotype for SynLethDB SL pairs against its screened non-SL pairs ($0.5$: no agreement), with SL pairs split by the cell line SynLethDB reports for them (K562 for Adamson and Gilbert; Corn's RPE-1 has no SynLethDB SL pair). The last column compares K562 calls with Jurkat-only calls from the same CRISPRi map~\citep{Horlbeck2018}, which holds the gene set fixed. $95\%$ percentile bootstrap CIs over pairs ($B=1000$); every difference has Holm-adjusted $p=0.006$. Pairs share genes, so these intervals are optimistic: under a gene-level bootstrap the overall AUROCs of Adamson $\gamma$, Adamson GI and Corn do not differ from $0.5$, the own-minus-other difference for Adamson GI is not significant ($p\approx0.10$), and the same-map differences remain significant. On Adamson $\gamma$ the overall value is below $0.5$ partly because SL and screened non-SL pairs come from different gene sets: on the genes of the Horlbeck map it is $0.595$ [$0.561$, $0.627$]. Gilbert's SynLethDB labels come almost entirely from that screen, so its row is in-distribution.}}
\label{tab:sldb-agreement}
\end{table}
""")
write('tab7_counts.txt', '; '.join(counts).replace('Adamson $\\gamma$', 'Adamson $\\gamma$') + '\n')

# ---------------------------------------------------------------- Appendix: SLeuth minus each comparator, fold-paired
rows = [r for r in sig if r['pca_arm'] == 'none' and r['family'] == 'vs_peers:siamese_sl_pd_pooled']
sg = {(r['tier'], r['B'], r['metric']): r for r in rows}
out = []
for tier in TIERS:
    for j, b in enumerate(PEERS):
        cells = []
        for m in METRICS:
            r = sg[(tier, b, m)]
            cells.append(f"{sgn(r['diff_mean'])} {ci(r['ci_t_low'], r['ci_t_high'])}; {pfmt(r['p_t_holm'])}; {r['wins']}/5")
        first = r'\multirow{6}{*}{%s}' % tier if j == 0 else ''
        out.append(f"{first} & {NAME[b]} & " + ' & '.join(cells) + r' \\')
    out.append(r'\hline')
write('tab1_diff.tex', r"""\begin{table}[h]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
{\color{red}
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll|ccc}
\hline
Tier & SLeuth $-$ & AUROC & AUPR & AUPRG \\
\hline
""" + '\n'.join(out) + r"""
\end{tabular}}}
\caption{\rev{\textbf{Paired comparisons behind Table~\ref{tab:tiers}.} SLeuth minus each comparator: mean difference over the five splits [split-level $95\%$ $t$-interval]; Holm-adjusted paired $t$-test $p$ (over the six comparators of a tier and metric); number of splits on which SLeuth is higher. In the paired bootstrap over test pairs every difference has $p\le0.002$.}}
\label{tab:tiers-diff}
\end{table}
""")

# ---------------------------------------------------------------- Appendix: modality picks
win = rd(REV + '/pca_solo_loco_sel/s4_winners_per_fold.csv')
wg = {(r['tier'], r['category'], int(r['fold'])): r['winner'] for r in win}
out = []
for tier in TIERS:
    for j, (cat, disp) in enumerate([('expression', 'expression'), ('protein_seq', 'protein\\_seq'),
                                     ('text', 'text'), ('ppi', 'ppi'), ('go', 'go'), ('kg', 'kg')]):
        picks = [tex_name(wg[(tier, cat, f)]) for f in range(5)]
        first = r'\multirow{6}{*}{%s}' % tier if j == 0 else ''
        out.append(f"{first} & {disp} & " + ' & '.join(picks) + r' \\')
    out.append(r'\hline')
write('tab_picks.tex', r"""\begin{table}[h]
\centering
\scriptsize
{\color{red}
\begin{tabular}{ll|ccccc}
\hline
Tier & Category & split 1 & split 2 & split 3 & split 4 & split 5 \\
\hline
""" + '\n'.join(out) + r"""
\end{tabular}}
\caption{\rev{\textbf{Embeddings selected per tier and split.} For each category, the embedding whose pooled single-modality model scores the highest AUPRG on the split's selection pairs (a random $10\%$ of its training pairs). The same picks are used by SLeuth, SLeuth-CL, the bilinear probe and both PCA arms.}}
\label{tab:picks}
\end{table}
""")

# ---------------------------------------------------------------- Appendix: PCA arms
pca = [r for r in rd(REV + '/pca_solo_loco_sel/s1_pca_paired.csv') if r['metric'] == 'auprg']
PM = [('siamese_sl_pd_pooled', 'SLeuth'), ('siamese_sl', 'SLeuth-CL'),
      ('siamese_sl_allmod_pd_pooled', 'SLeuth-all'), ('siamese_sl_allmod', 'SLeuth-all-CL')]
pg2 = {(r['model'], r['tier']): r for r in pca}
out = []
for m, disp in PM:
    for j, tier in enumerate(TIERS):
        r = pg2[(m, tier)]
        first = disp if j == 0 else ''
        hi = max(round(float(r['none_mean']), 3), round(float(r['joint_mean']), 3))
        nv, jv = num(r['none_mean']), num(r['joint_mean'])
        nv = bold(nv) if round(float(r['none_mean']), 3) == hi else nv
        jv = bold(jv) if round(float(r['joint_mean']), 3) == hi else jv
        out.append(f"{first:<14}& {tier} & {nv} ({float(r['none_sd_pop']):.3f}) & "
                   f"{jv} ({float(r['joint_sd_pop']):.3f}) & "
                   f"{sgn(r['mean_diff'])} {ci(r['ci95_low'], r['ci95_high'])} & "
                   f"{pfmt(r['p_t_holm_tier'])} & {r['n_pos']}/5" + r' \\')
    out.append(r'\hline')
write('tab_pca.tex', r"""\begin{table}[h]
\centering
\scriptsize
{\color{red}
\begin{tabular}{ll|cc|ccc}
\hline
Model & Tier & no PCA & PCA $80\%$ & difference & Holm $p$ & splits no PCA higher \\
\hline
""" + '\n'.join(out) + r"""
\end{tabular}}
\caption{\rev{\textbf{Effect of the principal component projection.} AUPRG of each multi-modal model without PCA (as reported) and with the submitted version's projection of the concatenation onto the components retaining $80\%$ of the variance, trained on the same modalities in every split; mean (standard deviation) over the five splits, the fold-paired difference [$95\%$ $t$-interval] and its Holm-adjusted $t$-test $p$ (over the SLeuth rows of a tier).}}
\label{tab:pca}
\end{table}
""")
print('done')
