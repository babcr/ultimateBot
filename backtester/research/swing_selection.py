"""Offline execution policy: smallest stop admitted by the frozen threshold."""
import math
import re

import numpy as np

FIRST_STOP_POLICY = 'first_accepted_stop_per_swing_v1'


class StopSelection:
    def __init__(self, group_ids, variant_indices, expected_count):
        self.ids = np.asarray(group_ids, dtype=str)
        raw = np.asarray(variant_indices)
        if (self.ids.ndim != 1 or raw.shape != self.ids.shape or type(expected_count) is not int
                or expected_count < 1 or not np.isfinite(raw).all() or not (raw == raw.astype(np.int64)).all()):
            raise ValueError('Invalid stop-selection identities or variant count')
        self.ranks = raw.astype(np.int64)
        if (self.ranks < 0).any() or (self.ranks >= expected_count).any():
            raise ValueError('Stop index outside the collected grid')
        self.expected_count = expected_count
        groups, self.codes = np.unique(self.ids, return_inverse=True)
        self.group_count = len(groups)
        self.order = np.lexsort((self.ranks, self.codes))
        pairs = np.column_stack((self.codes[self.order], self.ranks[self.order]))
        if len(pairs) > 1 and np.any(np.all(pairs[1:] == pairs[:-1], axis=1)):
            raise ValueError('Duplicate stop variant within a swing')
        counts = np.bincount(self.codes, minlength=self.group_count)
        self.complete = counts == expected_count
        self.valid = self.complete[self.codes]
        self.complete_count = int(self.complete.sum())

    @classmethod
    def from_event_ids(cls, event_ids, config):
        count = config.get('stop_variants', 1)
        if count <= 1:
            return None
        groups, ranks = [], []
        for event in event_ids:
            match = re.fullmatch(r'(.*):stop([1-9][0-9]*)', str(event))
            groups.append(match[1] if match else str(event))
            ranks.append(int(match[2]) if match else 0)
        return cls(groups, ranks, count)

    @classmethod
    def from_rows(cls, rows, config=None):
        if config is not None:
            if config.get('stop_variants', 1) > 1 and rows and all(r.get('swing_id') is not None for r in rows):
                return cls([r['swing_id'] for r in rows], [r['variant_index'] for r in rows], config['stop_variants'])
            return cls.from_event_ids([r['event_id'] for r in rows], config)
        # Direct callers may carry the complete collection count in their rows.
        count = max((r.get('stop_variants', 1) for r in rows), default=1)
        return cls.from_event_ids([r['event_id'] for r in rows], dict(stop_variants=count)) if count > 1 else None

    def subset(self, indices):
        return StopSelection(self.ids[indices], self.ranks[indices], self.expected_count)

    def mask(self, predictions, threshold):
        predictions = np.asarray(predictions, dtype=np.float64)
        if predictions.shape != self.ids.shape or not np.isfinite(predictions).all() or not math.isfinite(threshold):
            raise ValueError('Invalid stop-selection scores or threshold')
        eligible = self.order[self.valid[self.order] & (predictions[self.order] >= threshold)]
        chosen = eligible[np.r_[True, self.codes[eligible[1:]] != self.codes[eligible[:-1]]]] if len(eligible) else eligible
        result = np.zeros(len(predictions), dtype=bool)
        result[chosen] = True
        return result

    def diagnostic(self):
        return dict(selection_policy=FIRST_STOP_POLICY, raw_variants=len(self.ids), swings=self.group_count,
                    complete_swings=self.complete_count, incomplete_swings=self.group_count-self.complete_count,
                    expected_stop_variants=self.expected_count)


def acceptance_mask(predictions, threshold, selection=None):
    return selection.mask(predictions, threshold) if selection is not None else np.asarray(predictions) >= threshold


def selection_metrics(labels, mask, selection):
    values = np.asarray(labels, dtype=np.float64)[mask]
    profit, loss = float(np.maximum(values, 0).sum()), float(-np.minimum(values, 0).sum())
    count = len(values)
    return dict(opportunities=count, coverage=count/selection.complete_count if selection.complete_count else 0.,
                total_r=float(values.sum()), mean_r=float(values.mean()) if count else None,
                win_rate=float((values > 0).mean()) if count else None,
                profit_factor_r=profit/loss if loss else None, gross_profit_r=profit, gross_loss_r=loss)


def exact_stop_threshold(labels, predictions, minimum, selection, *, score_metric='total_r', empty_score=0.,
                         max_threshold=None):
    """O(n log n) sweep; admitting a shorter stop replaces the previous trade."""
    labels, predictions = np.asarray(labels, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
    if (labels.shape != selection.ids.shape or predictions.shape != labels.shape or not len(labels)
            or not np.isfinite(labels).all() or not np.isfinite(predictions).all() or not math.isfinite(minimum)
            or score_metric not in ('total_r', 'mean_r') or not math.isfinite(empty_score)
            or max_threshold is not None and (not math.isfinite(max_threshold) or minimum > max_threshold)):
        raise ValueError('Threshold selection requires finite aligned labels and predictions')
    order = np.argsort(-predictions, kind='stable')
    chosen = np.full(selection.group_count, -1, dtype=np.int64)
    total = profit = loss = 0.
    count = wins = 0
    candidates = []

    def snapshot(threshold):
        return dict(threshold=float(threshold), opportunities=count,
            coverage=count/selection.complete_count if selection.complete_count else 0., total_r=float(total),
            mean_r=float(total/count) if count else None, win_rate=wins/count if count else None,
            profit_factor_r=float(profit/loss) if loss > 0 else None,
            gross_profit_r=float(max(profit, 0.)), gross_loss_r=float(max(loss, 0.)))

    i = 0
    while i < len(order) and predictions[order[i]] >= minimum:
        score = predictions[order[i]]
        while i < len(order) and predictions[order[i]] == score:
            row = order[i]
            group = selection.codes[row]
            old = chosen[group]
            if selection.valid[row] and (old < 0 or selection.ranks[row] < selection.ranks[old]):
                if old >= 0:
                    previous = labels[old]
                    total -= previous
                    profit -= max(previous, 0.)
                    loss -= max(-previous, 0.)
                    wins -= int(previous > 0)
                else:
                    count += 1
                value = labels[row]
                total += value
                profit += max(value, 0.)
                loss += max(-value, 0.)
                wins += int(value > 0)
                chosen[group] = row
            i += 1
        candidates.append(snapshot(score))
    if not candidates or candidates[-1]['threshold'] != minimum:
        candidates.append(snapshot(minimum))
    none = max(minimum, float(np.nextafter(float(predictions.max()), np.inf)))
    if not math.isfinite(none):
        raise ValueError('Cannot represent a finite abstention threshold')
    if none != minimum and (max_threshold is None or none <= max_threshold):
        candidates.append(dict(threshold=none, opportunities=0, coverage=0., total_r=0., mean_r=None,
                               win_rate=None, profit_factor_r=None, gross_profit_r=0., gross_loss_r=0.))
    candidates.sort(key=lambda r:r['threshold'])
    if max_threshold is not None:
        candidates = [r for r in candidates if r['threshold'] <= max_threshold]

    def best_of(values):
        if score_metric == 'total_r':
            return max(values, key=lambda r:(r['total_r'], -r['opportunities']))
        # Prefer coverage only when quality exceeds the neutral/abstention label.
        # Treat floating-point noise in equal means as a tie (not an edge).
        best, best_score = None, -math.inf
        for row in values:
            score = row['mean_r'] if row['opportunities'] else empty_score
            tied = math.isclose(score, best_score, rel_tol=1e-12, abs_tol=1e-12)
            count = row['opportunities'] if score > empty_score else -row['opportunities']
            old_count = (best['opportunities'] if best_score > empty_score else -best['opportunities']) if best else 0
            if best is None or (not tied and score > best_score) or (tied and count > old_count):
                best, best_score = row, score
        return best

    selected = best_of(candidates)
    active = [r for r in candidates if r['opportunities']]
    best_active = (max(active, key=lambda r:r['total_r']) if score_metric == 'total_r' else best_of(active)) if active else None
    floor = next(r for r in candidates if r['threshold'] == minimum)
    reason = ('no_complete_swing_outcomes' if not selection.complete_count else
              'positive_validation_pnl' if selected['opportunities'] else
              'no_prediction_at_or_above_floor' if not floor['opportunities'] else 'no_profitable_admissible_threshold')
    diagnostics = dict(method='exact_first_accepted_stop_no_count_filter_v1', thresholds_evaluated=len(candidates),
        admissible_nonempty_thresholds=len(active), minimum_accepted=0, acceptance_count_filter=False,
        events_at_floor=floor['opportunities'], reason=reason, best_exploratory=best_active,
        best_admissible_nonempty=best_active, at_floor=floor,
        selection_includes_abstention=any(not row['opportunities'] for row in candidates),
        **selection.diagnostic())
    keep = set(np.linspace(0, len(candidates)-1, min(100, len(candidates)), dtype=int).tolist())
    keep.update(i for i, r in enumerate(candidates) if r is selected or r is best_active or r is floor)
    return {**selected, 'diagnostics': diagnostics}, [candidates[i] for i in sorted(keep)]
