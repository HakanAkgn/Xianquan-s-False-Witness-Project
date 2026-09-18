"""Generate separate figures directly from the committed numerical tables."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def draw(results='results', destination='results/figures', show=False):
    root, out = Path(results), Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    figures = []

    def finish(fig, ax, stem):
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.2), ncol=3, frameon=False)
        ax.grid(axis='y', alpha=.2)
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(out/f'{stem}.{suffix}', dpi=200, bbox_inches='tight')
        figures.append(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for r in read_csv(root/'balanced.csv'):
        if r['step']=='2000' and r['arm']=='baseline' and r['hint']=='incorrect':
            ax.plot([0, 1], [100*float(r['control_rate']), 100*float(r['witness_rate'])],
                    marker='o', linewidth=2, label=f"Seed {r['seed']}")
    ax.set_xticks([0, 1], ['Control', 'Token-balanced witness'])
    ax.set_xlim(-.2, 1.2); ax.set_ylim(-3, 103)
    ax.set_ylabel('Wrong-suggestion following (%)')
    finish(fig, ax, 'balanced_pairs_2000')

    data=read_csv(root/'cube_curves.csv')
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for seed in ('11', '29', '47'):
        rows=[r for r in data if r['task']=='original' and r['step']=='2000'
              and r['arm']=='baseline' and r['hint']=='incorrect' and r['seed']==seed]
        rows.sort(key=lambda r:int(r['count']))
        rates=[100*float(r['incorrect_suggestion_following']) for r in rows]
        ax.plot([int(r['count']) for r in rows], [v-rates[0] for v in rates],
                marker='o', linewidth=2, label=f'Seed {seed}')
    ax.set_xticks(range(5)); ax.set_xlabel('Witness count within the same 16-context cube')
    ax.set_ylabel('Change in wrong-suggestion following\n(percentage points from zero witnesses)')
    finish(fig, ax, 'fixed_token_count_response')

    data=read_csv(root/'repairs.csv')
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    arms=('iid', 'augmentation', 'consistency')
    for seed in ('11', '29', '47'):
        values=[100*float(next(r for r in data if r['seed']==seed and r['arm']==arm)['cube_robust_accuracy'])
                for arm in arms]
        ax.plot(range(3), values, marker='o', linewidth=2, label=f'Seed {seed}')
    ax.set_xticks(range(3), ['Ordinary data', 'Counterfactual data', 'Counterfactual + JS'])
    ax.set_ylabel('Correct on all 16 incorrect-hint contexts (%)')
    ax.set_ylim(-3, 103)
    finish(fig, ax, 'repair_cube_robustness')

    data=read_csv(root/'cubes.csv')
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for r in data:
        if r['task']=='original' and r['step']=='2000' and r['arm']=='baseline' and r['hint']=='incorrect':
            ax.plot([0, 1], [float(r['constant_baseline_heldout_margin_rmse']),
                            float(r['additive_singleton_heldout_margin_rmse'])],
                    marker='o', linewidth=2, label=f"Seed {r['seed']}")
    ax.set_xticks([0, 1], ['Constant baseline', 'Singleton-calibrated additive'])
    ax.set_xlim(-.2, 1.2); ax.set_ylim(bottom=0)
    ax.set_ylabel('Held-out wrong-minus-correct margin RMSE')
    finish(fig, ax, 'heldout_additive_prediction')
    if show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
    return figures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', default='results')
    p.add_argument('--destination', default='results/figures')
    a=p.parse_args(); draw(a.results, a.destination)

if __name__=='__main__':
    main()
