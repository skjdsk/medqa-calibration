"""Fit scalar temperature on validation predictions; apply to disjoint predictions.

No model or GPU is needed. Raw CSV files are never modified. Fit each checkpoint
and evaluation protocol separately. Test labels are used for reporting only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import compute_metrics, reliability_table, prob_matrix
from train.dpo.validation import write_json


def read_frame(path):
    frame = pd.read_csv(path)
    validate_frame(frame)
    return frame


def identity(frame):
    result = {}
    for key in ('model_name', 'eval_mode'):
        values = frame[key].dropna().unique()
        if len(values) != 1 or frame[key].isna().any():
            raise ValueError(f'Expected one nonempty {key} per calibration run')
        result[key] = str(values[0])
    if 'temperature' in frame or result['eval_mode'].endswith('_temperature'):
        raise ValueError('Input must contain uncalibrated predictions')
    if 'prompt_version' in frame:
        values = frame.prompt_version.dropna().unique()
        if len(values) != 1:
            raise ValueError('Mixed prompt versions')
        result['prompt_version'] = str(values[0])
    return result


def row_keys(frame):
    return [json.dumps([str(d), str(q)], separators=(',', ':'))
            for d, q in zip(frame.dataset, frame.question_id)]


def validate_frame(frame):
    required = {'dataset', 'question_id', 'model_name', 'eval_mode', 'gold', 'n_options'}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError('Empty predictions or missing required prediction columns')
    if frame[list(required)].isna().any().any():
        raise ValueError('Prediction identifiers and labels cannot be missing')
    if len(set(row_keys(frame))) != len(frame):
        raise ValueError('Duplicate question identities')
    identity(frame)
    probs, gold = prob_matrix(frame)
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError('Probabilities must be finite and nonnegative')
    counts = frame.n_options.to_numpy(dtype=float)
    if np.any(counts != counts.astype(int)) or np.any(counts < 2) or np.any(counts > probs.shape[1]):
        raise ValueError('Invalid option counts')
    mask = np.arange(probs.shape[1])[None, :] < counts[:, None]
    if (gold >= counts).any() or np.any(probs[~mask] != 0):
        raise ValueError('Labels or probabilities reference unavailable options')


def logits_from_frame(frame):
    validate_frame(frame)
    cols = [f'prob_{letter}' for letter in 'ABCDEFGHIJ' if f'prob_{letter}' in frame]
    if cols != [f'prob_{letter}' for letter in 'ABCDEFGHIJ'[:len(cols)]]:
        raise ValueError('Probability columns must be contiguous from A')
    probs, gold = prob_matrix(frame)
    mask = np.arange(probs.shape[1])[None, :] < frame.n_options.to_numpy()[:, None]
    log_cols = [col.replace('prob_', 'log_prob_') for col in cols]
    available = [col in frame for col in log_cols]
    if any(available) and not all(available):
        raise ValueError('Incomplete log probability columns')
    if all(available):
        logits = frame[log_cols].to_numpy(dtype=float)
        if not np.isfinite(logits[mask]).all():
            raise ValueError('Available options need finite log probabilities')
        logits[~mask] = -np.inf
        if not np.allclose(scale_logits(logits, 1.0), probs, atol=1e-6):
            raise ValueError('Log probabilities do not match probability columns')
        source = 'saved_log_probabilities'
    else:
        logits = np.log(np.maximum(probs, 1e-30))
        logits[~mask] = -np.inf
        source = 'probabilities_clipped_at_1e-30'
    return logits, gold, cols, source


def scale_logits(logits, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    scaled = logits / temperature
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def nll(logits, gold, temperature):
    scaled = logits / temperature
    maximum = scaled.max(axis=1)
    normalizer = maximum + np.log(np.exp(scaled - maximum[:, None]).sum(1))
    return float(np.mean(normalizer - scaled[np.arange(len(gold)), gold]))


def fit_temperature(frame, minimum=0.05, maximum=100.0):
    """Minimize NLL in inverse temperature, a convex one-dimensional objective."""
    if not 0 < minimum < 1 < maximum:
        raise ValueError('Temperature bounds must contain 1')
    logits, gold, _, source = logits_from_frame(frame)
    left, right = 1 / maximum, 1 / minimum
    ratio = (np.sqrt(5) - 1) / 2
    for _ in range(100):
        a, b = right - ratio * (right - left), left + ratio * (right - left)
        if nll(logits, gold, 1 / a) < nll(logits, gold, 1 / b):
            right = b
        else:
            left = a
    candidates = [minimum, maximum, 1.0, 1 / ((left + right) / 2)]
    temperature = min(candidates, key=lambda t: nll(logits, gold, t))
    return {'temperature': float(temperature), 'bounds': [minimum, maximum],
            'at_bound': temperature in (minimum, maximum), 'probability_source': source,
            'fit_nll_before': nll(logits, gold, 1), 'fit_nll_after': nll(logits, gold, temperature)}


def transform(frame, temperature):
    logits, _, cols, _ = logits_from_frame(frame)
    probs = scale_logits(logits, temperature)
    result = frame.copy()
    result = result.drop(columns=[c for c in result if c.startswith('log_prob_')])
    for i, col in enumerate(cols):
        result[col] = np.where(i < frame.n_options.to_numpy(), probs[:, i], np.nan)
    result['pred'] = np.array(list('ABCDEFGHIJ'))[probs.argmax(1)]
    result['correct'] = (result.pred == result.gold).astype(int)
    result['confidence'] = probs.max(1)
    result['temperature'] = temperature
    result['eval_mode'] = result.eval_mode + '_temperature'
    return result


def fit_artifact(frame, calibration_split='validation'):
    if calibration_split != 'validation':
        raise ValueError('Temperature may only be fitted on validation data')
    if frame.question_id.astype(str).str.contains('test', case=False).any():
        raise ValueError('Test question IDs cannot be used to fit temperature')
    if 'split' in frame and not (frame['split'] == 'validation').all():
        raise ValueError('Only validation rows may be used for temperature fitting')
    result = fit_temperature(frame)
    result.update(schema_version=1, identity=identity(frame), calibration_split=calibration_split,
                  fit_count=len(frame), fit_question_keys=row_keys(frame),
                  policy='Fit on validation only; never use fit rows for held-out reporting.')
    return result


def apply_artifact(frame, artifact):
    validate_frame(frame)
    if identity(frame) != artifact['identity']:
        raise ValueError('Model or evaluation protocol differs from temperature fit')
    if set(row_keys(frame)).intersection(artifact['fit_question_keys']):
        raise ValueError('Calibration-fit and evaluation questions overlap')
    return transform(frame, artifact['temperature'])


def report(raw, calibrated, directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Report directory must be empty')
    directory.mkdir(parents=True, exist_ok=True)
    calibrated.to_csv(directory / 'calibrated_predictions.csv', index=False)
    rows = []
    for name, frame in [('raw', raw), ('temperature', calibrated)]:
        metrics = compute_metrics(frame, n_bins=15, hc_threshold=0.9)
        metrics['hc_coverage'] = metrics['hc_count'] / metrics['n']
        rows.append({'variant': name, **metrics})
        probs, gold = prob_matrix(frame)
        reliability_table(probs.max(1), (probs.argmax(1) == gold).astype(float)).to_csv(
            directory / f'{name}_reliability.csv', index=False)
    write_json(directory / 'metrics.json', rows)
    pd.DataFrame(rows).to_csv(directory / 'metrics.csv', index=False)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit')
    fit.add_argument('--calibration_csv', type=Path, required=True)
    fit.add_argument('--calibration_split', choices=['validation'], required=True)
    fit.add_argument('--output', type=Path, required=True)
    apply = commands.add_parser('apply')
    apply.add_argument('--predictions_csv', type=Path, required=True)
    apply.add_argument('--temperature_file', type=Path, required=True)
    apply.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'fit':
        if args.output.exists():
            raise ValueError('Temperature artifact already exists')
        artifact = fit_artifact(read_frame(args.calibration_csv), args.calibration_split)
        artifact['source_sha256'] = hashlib.sha256(args.calibration_csv.read_bytes()).hexdigest()
        artifact['source_csv'] = str(args.calibration_csv)
        write_json(args.output, artifact)
        print(json.dumps({k: v for k, v in artifact.items() if k != 'fit_question_keys'}, indent=2))
    else:
        artifact = json.loads(args.temperature_file.read_text(encoding='utf-8'))
        frame = read_frame(args.predictions_csv)
        calibrated = apply_artifact(frame, artifact)
        rows = report(frame, calibrated, args.output_dir)
        write_json(args.output_dir / 'application_manifest.json', {
            'temperature_file': str(args.temperature_file), 'temperature': artifact['temperature'],
            'source_csv': str(args.predictions_csv),
            'source_sha256': hashlib.sha256(args.predictions_csv.read_bytes()).hexdigest()})
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
