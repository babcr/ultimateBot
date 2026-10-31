#!/usr/bin/env python3
"""Run a read-only, reproducible statistical swing rule search."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from ultimateBot.backtester.research.swing_statistics import StatisticalProtocol, extract_events, research


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', type=Path, default=Path('C:/TradingBotData/swing_ts2vec/research.sqlite3'))
    p.add_argument('--run-id')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--features', type=Path, help='Reuse an explicit frozen features.npz snapshot')
    p.add_argument('--extract-only', action='store_true')
    defaults = StatisticalProtocol()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        p.add_argument('--'+field.name.replace('_', '-'), type=type(value), default=value)
    args = p.parse_args(argv)
    protocol = StatisticalProtocol(**{f.name: getattr(args, f.name) for f in fields(defaults)})
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'frozen_rule.json').exists():
        raise ValueError('Do not overwrite a frozen research result; use a new output directory')
    if args.features:
        with np.load(args.features, allow_pickle=False) as saved:
            data, metadata = saved['data'], json.loads(str(saved['metadata']))
    else:
        data, metadata = extract_events(args.database, args.run_id, protocol, args.output)
    if args.extract_only:
        print(json.dumps(metadata, indent=2))
        return metadata
    result = research(data, metadata, args.output, protocol)
    print(json.dumps({k: result[k] for k in ('hypotheses', 'discovery_qualified', 'validation_qualified',
                                           'test_evaluated', 'edge_confirmed')}, indent=2))
    return result


if __name__ == '__main__':
    main()
