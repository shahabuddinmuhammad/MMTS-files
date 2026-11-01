import ast
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
FILES = {'cost': 'result10.2.1 time-Cost.txt',
         'emission': 'result10.2.1 time-Emission.txt',
         'modeswitches': 'result10.2.1 time-ModeSwitches.txt'}
YLABEL = {'cost': 'Expected travel cost', 'emission': 'Expected emissions',
          'modeswitches': 'Expected mode switches'}
# Labels are matched to the legend of the user-supplied old figure.
# Confirm allStations semantics against the PRISM model before publication.
CONFIGS = [(1, 0, 'Cycle, No bus lanes', 'tab:blue', 'o', '--'),
           (2, 0, 'Bus, No bus lanes', 'tab:orange', 's', '--'),
           (2, 1, 'Bus, Dedicated bus lanes', 'tab:green', 's', '-'),
           (3, 0, 'Car, No bus lanes', 'tab:red', '^', '--'),
           (3, 1, 'Car, Dedicated bus lanes', 'tab:purple', '^', '-')]
LEVELS = (0.0, 0.5, 0.8)

def load(path):
    data = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            key = (int(r['allStations']), int(r['lanes4Bus']),
                   round(float(r['pCongFixed']), 3))
            data[key] = [tuple(map(float, p)) for p in ast.literal_eval(r['Result'])]
    assert len(data) == 66, f'Unexpected row count: {len(data)}'
    return data

for metric, filename in FILES.items():
    data = load(ROOT / filename)
    fig = plt.figure(figsize=(13.4, 10.4))
    gs = GridSpec(2, 4, figure=fig, left=.09, right=.98, top=.94,
                  bottom=.15, wspace=.72, hspace=.56)
    axes = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:]),
            fig.add_subplot(gs[1, 1:3])]
    handles = {}
    for ax, level in zip(axes, LEVELS):
        for code, lane, name, color, marker, linestyle in CONFIGS:
            points = sorted(data[(code, lane, level)])
            line, = ax.plot([x for x, y in points], [y for x, y in points],
                            color=color, marker=marker, linestyle=linestyle,
                            linewidth=1.8, markersize=5, label=name)
            handles[name] = line
        ax.set_title(r'$p_{\mathrm{Cong}} = $' + f'{level:g}', fontsize=14)
        ax.set_xlabel('Expected travel time (min)')
        ax.set_ylabel(YLABEL[metric])
        ax.grid(True, alpha=.27)
        ax.tick_params(labelsize=9)
    fig.legend([handles[c[2]] for c in CONFIGS], [c[2] for c in CONFIGS],
               loc='lower center', ncol=3, bbox_to_anchor=(.53, .025),
               frameon=True, fontsize=10)
    for ext in ('png', 'pdf'):
        kwargs = {'dpi': 240} if ext == 'png' else {}
        fig.savefig(ROOT / f'pareto_time_{metric}_prism_2row.{ext}',
                    facecolor='white', **kwargs)
    plt.close(fig)
