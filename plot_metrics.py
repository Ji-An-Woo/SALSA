#!/usr/bin/env python3
import argparse
import ast
import json
import math
from pathlib import Path


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return math.nan


def parse_log(log_path):
    log_path = Path(log_path)
    epochs = []
    loss = []
    beam_acc = []
    perfect = []
    acc_10perc = []

    for line in log_path.read_text().splitlines():
        if '__log__:' not in line:
            continue
        payload = line.split('__log__:', 1)[1].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        epochs.append(_safe_float(data.get('epoch')))
        loss.append(_safe_float(data.get('valid_lattice_xe_loss')))
        beam_acc.append(_safe_float(data.get('valid_lattice_beam_acc')))
        perfect.append(_safe_float(data.get('valid_lattice_perfect')))

        percs = data.get('valid_lattice_percs_diff')
        if percs is None:
            acc_10perc.append(math.nan)
        else:
            try:
                percs = ast.literal_eval(percs)
                acc_10perc.append(_safe_float(percs[0]))
            except Exception:
                acc_10perc.append(math.nan)

    return {
        'epochs': epochs,
        'loss': loss,
        'beam_acc': beam_acc,
        'perfect': perfect,
        'acc_10perc': acc_10perc,
    }


def default_output_path(log_path, root_dir=None):
    log_path = Path(log_path)
    root = Path(root_dir) if root_dir else Path.cwd()
    exp_id = log_path.parent.name if log_path.parent else 'exp'
    exp_name = log_path.parent.parent.name if log_path.parent and log_path.parent.parent else 'run'
    safe_exp_name = exp_name.replace('/', '_').replace(chr(92), '_')
    safe_exp_id = exp_id.replace('/', '_').replace(chr(92), '_')
    return root / f'metrics_{safe_exp_name}_{safe_exp_id}.png'


def plot_metrics(log_path, output_path=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError('matplotlib is required to plot metrics. Install it to generate graphs.') from exc

    data = parse_log(log_path)
    if not data['epochs']:
        return None

    epochs = data['epochs']
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    axes[0].plot(epochs, data['loss'], marker='o', label='valid_xe_loss')
    axes[0].set_ylabel('valid_xe_loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, data['acc_10perc'], marker='o', label='<=0.1Q (%)')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('0.1Q accuracy (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    output_path = Path(output_path) if output_path else default_output_path(log_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Plot SALSA metrics from train.log')
    parser.add_argument('--log', required=True, help='Path to train.log')
    parser.add_argument('--out', default=None, help='Output PNG path (default: metrics_<exp>_<id>.png)')
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f'log not found: {log_path}')
        return 0

    saved = plot_metrics(log_path, args.out)
    if saved:
        print(f'saved {saved}')
    else:
        print('no metrics found in log')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
