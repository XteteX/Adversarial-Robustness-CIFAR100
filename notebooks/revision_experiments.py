# =====================================================================
# Revision experiments (October 2026), run as cells 11-14 appended to
# notebooks/adversarial_robustness_cifar100.ipynb on Kaggle (T4, ~6.75 h in total).
# Run sections 1-10 of the notebook first with the 12 finished runs available as input;
# their training is skipped because the DONE files exist.
#   11: AutoAttack at 4/255 + PGD-20 on the same 1,000 test images (no training)
#   12: decomposition with std over seeds, reverse order, logit-MSE ratio r
#   13: VGG-16-lin control (single linear head), 3 seeds x {std, at}
#   14: summary tables incl. VGG-16-lin
# Outputs are in results/revision_*.csv, results/same_subset_*.csv, results/gap_decomposition.csv.
# =====================================================================

# ---------------------------------------------------------------------
# 11. Extra evaluation for the revision (NO retraining, final checkpoints)
#     - AutoAttack at eps = 4/255 (training budget) on the first 1,000 test images
#     - PGD-20 and clean accuracy on the SAME 1,000 images (fair AA vs PGD check)
#     ~1-1.5 h on a T4 for all 12 models. Results go to eval_final.json -> 'extra'.
# ---------------------------------------------------------------------
from autoattack import AutoAttack

def extra_eval(run, ckpt='final.pt', n=CFG['aa_n'], bs=250):
    out = f'{run}/eval_{ckpt[:-3]}.json'
    res = json.load(open(out))
    if 'extra' in res:
        print('skip', os.path.basename(run)); return res
    arch = os.path.basename(run).split('_')[0]
    model = build(arch)
    model.load_state_dict(torch.load(f'{run}/{ckpt}', map_location=device)); model.eval()
    _, _, te, _ = get_loaders(0)
    xs = torch.cat([x for x, _ in te])[:n].to(device)
    ys = torch.cat([y for _, y in te])[:n].to(device)

    def acc_on(x):
        with torch.no_grad():
            p = torch.cat([model(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)])
        return 100 * float((p == ys).float().mean())

    ex = dict(n=n, clean=acc_on(xs))
    for e in [4, 8]:
        eps = e / 255
        torch.manual_seed(0)
        x20 = torch.cat([pgd(model, xs[i:i + bs], ys[i:i + bs], eps,
                             CFG['pgd20']['alpha'], CFG['pgd20']['steps'])
                         for i in range(0, n, bs)])
        ex[f'pgd20_{e}'] = acc_on(x20)
        if e == 8 and 'autoattack' in res:          # AA at 8/255 already computed
            ex['aa_8'] = res['autoattack']['acc']
        else:
            aa = AutoAttack(model, norm='Linf', eps=eps, version='standard', device=device, verbose=False)
            ex[f'aa_{e}'] = acc_on(aa.run_standard_evaluation(xs, ys, bs=bs))
    res['extra'] = ex
    json.dump(res, open(out, 'w'), indent=2)
    print(os.path.basename(run), {k: round(v, 2) for k, v in ex.items()})
    return res

for run in sorted(glob.glob(f'{ROOT}/*_s*')):
    if os.path.exists(f'{run}/DONE') and os.path.exists(f'{run}/eval_final.json'):
        extra_eval(run)


# ---------------------------------------------------------------------
# 12. Tables for the revision (seconds, CPU is enough)
#     (a) AA at 4/255 and PGD-20 on the same 1,000 images
#     (b) gap decomposition per seed, mean +- std, both reference orders
#     (c) perception-consistency ratio r with 3 decimals
# ---------------------------------------------------------------------
G = load('final')
seed_of = lambda r: os.path.basename(r['run']).rsplit('_s', 1)[1]

print('% (a) first 1,000 test images: Clean & PGD-20 4/255 & AA 4/255 & PGD-20 8/255 & AA 8/255')
for m in order:
    if m not in G: continue
    E = [r['extra'] for r in G[m] if 'extra' in r]
    print(f"{NAME[m]} & {fm([x['clean'] for x in E])} & {fm([x['pgd20_4'] for x in E])} & "
          f"{fm([x['aa_4'] for x in E])} & {fm([x['pgd20_8'] for x in E])} & {fm([x['aa_8'] for x in E])} \\\\  % n={len(E)}")

print('\n% (b) decomposition: eps & gap & clean part (pts, %) & cond. part (pts) & clean share, reverse order & symmetric')
Rn = {s: r for r in G['resnet18_at'] for s in [seed_of(r)]}
Vg = {s: r for r in G['vgg16_at'] for s in [seed_of(r)]}
for e in ['2/255', '4/255', '8/255']:
    rows = []
    for s in sorted(set(Rn) & set(Vg)):   # seeds are paired by index; runs are independent
        a, b = Rn[s]['results'][e], Vg[s]['results'][e]
        Ar, Av = a['clean_acc'], b['clean_acc']
        rr = a['n_def'] / (a['n_def'] + a['n_fail']); rv = b['n_def'] / (b['n_def'] + b['n_fail'])
        gap = a['pgd20_acc'] - b['pgd20_acc']
        clean_part = (Ar - Av) * rv; cond_part = Ar * (rr - rv)
        rev = (Ar - Av) * rr
        rows.append([gap, clean_part, 100 * clean_part / gap, cond_part,
                     100 * rev / gap, 100 * (clean_part + rev) / 2 / gap])
    M = np.array(rows)
    f = lambda j, d=2: f'{M[:, j].mean():.{d}f}$\\pm${M[:, j].std():.{d}f}'
    print(f"{e} & {f(0)} & {f(1)} ({f(2, 0)}\\%) & {f(3)} & {f(4, 0)}\\% & {f(5, 0)}\\% \\\\")

print('\n% (c) logit-MSE ratio r (3 decimals): model & 2/255 & 4/255 & 8/255')
for m in order:
    if m not in G: continue
    cells = []
    for e in ['2/255', '4/255', '8/255']:
        rat = [x['mse_failed'] / x['mse_defended'] if x['n_def'] >= 30 else None
               for x in [r['results'][e] for r in G[m]]]
        cells.append(fm(rat, 3))
    print(f"{NAME[m]} & " + ' & '.join(cells) + ' \\\\')


# ---------------------------------------------------------------------
# 13. (Optional, needs training) VGG-16 with a single linear classifier.
#     Tests whether VGG-16's low clean accuracy comes from the 2x512 + dropout head.
#     Same recipe as everything else (~21 s/epoch std, ~85 s/epoch AT on a T4).
# ---------------------------------------------------------------------
def vgg16lin_cifar(nc=100):
    m = torchvision.models.vgg16_bn(weights=None)
    m.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    m.classifier = nn.Linear(512, nc)
    return m

def build(arch):   # overrides the earlier build(); old archs unchanged
    nets = {'resnet18': resnet18_cifar, 'vgg16': vgg16_cifar, 'vgg16lin': vgg16lin_cifar}
    return nn.Sequential(Normalize(MEAN, STD), nets[arch]()).to(device)

print('vgg16lin', f"{sum(p.numel() for p in build('vgg16lin').parameters())/1e6:.2f}M params")
NAME.update({'vgg16lin_std': 'VGG-16-lin (standard)', 'vgg16lin_at': 'VGG-16-lin (mixed AT)'})
order += ['vgg16lin_std', 'vgg16lin_at']

NEW_SEEDS = [0, 1, 2]
NEW_MODES = ['std', 'at']
for seed in NEW_SEEDS:
    for mode in NEW_MODES:
        run = train('vgg16lin', mode, seed)
        evaluate(run, 'best.pt', autoattack=False)
        evaluate(run, 'final.pt', autoattack=False)
        add_autoattack(run, 'final.pt')
        extra_eval(run)


# ---------------------------------------------------------------------
# 14. Revision summary incl. VGG-16-lin (final checkpoint, mean +- std over seeds)
G = load('final')
seed_of = lambda r: os.path.basename(r['run']).rsplit('_s', 1)[1]
print('% Main (8/255): model & clean & PGD-20 4/255 & PGD-20 8/255 & AA 4/255 (1k) & AA 8/255 (1k)')
for m in order:
    if m not in G: continue
    R4 = [r['results']['4/255'] for r in G[m]]; R8 = [r['results']['8/255'] for r in G[m]]
    E = [r['extra'] for r in G[m] if 'extra' in r]
    print(f"{NAME[m]} & {fm([x['clean_acc'] for x in R8])} & {fm([x['pgd20_acc'] for x in R4])} & "
          f"{fm([x['pgd20_acc'] for x in R8])} & {fm([x['aa_4'] for x in E])} & {fm([x['aa_8'] for x in E])} \\\\  % n={len(R8)}")

def decomp(a_key, b_key, label):
    A = {seed_of(r): r for r in G.get(a_key, [])}; B = {seed_of(r): r for r in G.get(b_key, [])}
    S = sorted(set(A) & set(B))
    if not S: print('no paired seeds for', label); return
    print(f'\n% Decomposition {label}: eps & gap & clean part (%) & cond part & reverse-order share & rho_A & rho_B')
    for e in ['2/255', '4/255', '8/255']:
        rows = []
        for s in S:
            a, b = A[s]['results'][e], B[s]['results'][e]
            Ar, Av = a['clean_acc'], b['clean_acc']
            rr = a['n_def'] / (a['n_def'] + a['n_fail']); rv = b['n_def'] / (b['n_def'] + b['n_fail'])
            gap = a['pgd20_acc'] - b['pgd20_acc']
            c1 = (Ar - Av) * rv
            rows.append([gap, c1, 100 * c1 / gap, Ar * (rr - rv), 100 * (Ar - Av) * rr / gap, 100 * rr, 100 * rv])
        M = np.array(rows)
        f = lambda j, d=2: f'{M[:, j].mean():.{d}f}$\\pm${M[:, j].std():.{d}f}'
        print(f"{e} & {f(0)} & {f(1)} ({f(2, 0)}\\%) & {f(3)} & {f(4, 0)}\\% & {f(5, 1)} & {f(6, 1)} \\\\  % seeds={S}")

decomp('resnet18_at', 'vgg16_at', 'ResNet-18 vs VGG-16 (AT)')
decomp('resnet18_at', 'vgg16lin_at', 'ResNet-18 vs VGG-16-lin (AT)')
decomp('vgg16lin_at', 'vgg16_at', 'VGG-16-lin vs VGG-16 (AT, same backbone)')
json.dump({k: [r.get('extra') for r in v] for k, v in G.items()}, open(f'{ROOT}/revision_extra.json', 'w'), indent=2)
