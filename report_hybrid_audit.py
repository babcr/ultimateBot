"""Render the local hybrid-trade audit as a French report and offline explorer."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ultimateBot.audit_hybrid_trades import analyze, candidate_masks, grouped_stats, stats, write_json


def quarter_bootstrap(frame, repeats=10_000):
    if frame.empty:
        return [None, None]
    quarters = frame.closed_at.dt.tz_localize(None).dt.to_period('Q')
    # Include calendar quarters with zero trades in the sampling frame.
    complete = pd.period_range(quarters.min(), quarters.max(), freq='Q')
    grouped = frame.groupby(quarters).r.agg(['sum', 'count']).reindex(complete, fill_value=0)
    blocks = grouped.to_numpy()
    samples = np.random.default_rng(19026).integers(0, len(blocks), (repeats, len(blocks)))
    summed = blocks[samples].sum(axis=1)
    means = summed[:, 0] / np.maximum(summed[:, 1], 1)
    return np.quantile(means, [.025, .975]).tolist()


def table(rows, columns):
    lines = ['| ' + ' | '.join(label for _, label in columns) + ' |',
             '| ' + ' | '.join('---' for _ in columns) + ' |']
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                value = f'{value:.3f}' if abs(value) < 1 else f'{value:.2f}'
            values.append('—' if value is None else str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines)


def render(directory):
    frame = pd.read_csv(directory / 'trades_enriched.csv')
    for column in ('signal_at', 'submitted_at', 'opened_at', 'closed_at'):
        frame[column] = pd.to_datetime(frame[column], utc=True, format='ISO8601')
    # Entry filters must never depend on a future fill or its slippage.
    frame['planned_risk'] = frame.direction * (frame.planned_entry - frame.stop)
    frame['filled_rr'] = frame.direction * (frame.target - frame.entry) / frame.initial_risk
    frame['planned_rr'] = frame.direction * (frame.target - frame.planned_entry) / frame.planned_risk
    frame['stop_atr'] = frame.planned_risk / frame.atr
    frame['spread_risk'] = frame.spread_at_submission / frame.planned_risk
    frame.to_csv(directory / 'trades_enriched.csv', index=False)
    stored = json.loads((directory / 'analysis.json').read_text(encoding='utf-8'))
    report = analyze(frame)
    report['provenance'] = stored['provenance']
    close_ms = frame.closed_at.map(lambda value: int(value.timestamp()*1000))
    report['provenance']['baseline_replay_close_matches'] = int(frame.baseline_replay_close_ms.eq(close_ms).sum())
    report['provenance']['baseline_replay_max_r_error'] = float((frame.r-frame.baseline_replay_r).abs().max())
    if (report['provenance']['baseline_replay_close_matches'] != len(frame)
            or frame.baseline_replay_r.isna().any()
            or report['provenance']['baseline_replay_max_r_error'] > 1e-8):
        raise ValueError('Baseline exits do not reproduce the log; investigate before interpreting filters')
    report['total']['mean_r_quarter_block_ci95'] = quarter_bootstrap(frame)
    report['cost_stress'] = {str(cost): float(frame.r.sum() - cost * len(frame)) for cost in (0, .02, .05, .1)}
    write_json(directory / 'log_summary.json', {'total': stats(frame), 'by_branch': grouped_stats(frame, ['branch']),
                                             'by_symbol_year': grouped_stats(frame, ['symbol', 'year'])})
    report['feature_bins'] = []
    for feature in ('planned_rr', 'stop_atr', 'channel_width_atr', 'channel_drift', 'excursion_atr',
                    'parallelism', 'signal_body_direction', 'rejection_wick', 'efficiency_24',
                    'momentum_24_atr', 'spread_risk', 'volatility_6_48'):
        subset = frame.loc[frame.branch.eq('failed-breakout')]
        thresholds = subset.loc[subset.year.le(2022), feature].quantile([1/3, 2/3]).to_numpy()
        if not np.isfinite(thresholds).all() or thresholds[0] >= thresholds[1]:
            continue
        for group, mask in [('low', subset[feature].le(thresholds[0])),
                            ('middle', subset[feature].gt(thresholds[0]) & subset[feature].le(thresholds[1])),
                            ('high', subset[feature].gt(thresholds[1]))]:
            selected = subset.loc[mask]
            report['feature_bins'].append({'feature': feature, 'bin': group, 'thresholds_from_2020_2022': thresholds.tolist(),
                                          'all': stats(selected), 'development': stats(selected.loc[selected.year.le(2022)]),
                                          'validation': stats(selected.loc[selected.year.ge(2023)]),
                                          'by_symbol': grouped_stats(selected, ['symbol'])})
    for candidate in report['candidates']:
        mask = candidate_masks(frame)[candidate['name']]
        candidate['mean_r_quarter_block_ci95'] = quarter_bootstrap(frame.loc[mask])
        # Holding the trade stream fixed can reveal interactions, but cannot
        # predict additional signals that a real filtered replay would permit.
        candidate['without_usdchf'] = stats(frame.loc[mask & frame.symbol.ne('USDCHF')])
        if candidate['name'] == 'failed_stopside_wick_le_25pct':
            candidate['discovery'] = ('Exploratory: the low training-tertile cutoff is 0.248946, rounded to 0.25. '
                                      'The 2023-2025 period is a retrospective check, not untouched validation.')
    failed = frame.loc[frame.branch.eq('failed-breakout')]
    report['failed_reentry_timing'] = grouped_stats(failed,['reentry_bars'])
    nextbar = failed.loc[failed.reentry_bars.eq(1)]
    report['nextbar_failed'] = {'all':stats(nextbar), 'by_year':grouped_stats(nextbar,['year']),
                              'by_symbol':grouped_stats(nextbar,['symbol']),
                              'development':stats(nextbar.loc[nextbar.year.le(2022)]),
                              'validation':stats(nextbar.loc[nextbar.year.ge(2023)]),
                              'mean_r_quarter_block_ci95':quarter_bootstrap(nextbar)}
    report['stopside_wick_sensitivity'] = []
    for threshold in (.15, .20, .25, .30, .35):
        subset = failed.loc[failed.rejection_wick.le(threshold)]
        report['stopside_wick_sensitivity'].append(dict(threshold=threshold, **stats(subset),
            development_mean=stats(subset.loc[subset.year.le(2022)])['mean_r'],
            validation_mean=stats(subset.loc[subset.year.ge(2023)])['mean_r']))
    short_wick = failed.loc[failed.rejection_wick.le(.25)]
    report['short_wick_failed'] = {'all':stats(short_wick), 'by_year':grouped_stats(short_wick,['year']),
                                  'by_symbol':grouped_stats(short_wick,['symbol']),
                                  'mean_r_quarter_block_ci95':quarter_bootstrap(short_wick)}
    report['combinations'] = {}
    for name, mask in {'hybrid_without_usdchf_spread_le_10pct':frame.symbol.ne('USDCHF') & frame.spread_risk.le(.1),
                       'failed_without_usdchf_spread_le_10pct':frame.symbol.ne('USDCHF') & frame.spread_risk.le(.1) & frame.branch.eq('failed-breakout')}.items():
        report['combinations'][name] = dict(all=stats(frame.loc[mask]),
            development=stats(frame.loc[mask & frame.year.le(2022)]), validation=stats(frame.loc[mask & frame.year.ge(2023)]))
    write_json(directory / 'analysis.json', report)
    basic = [('n', 'Trades'), ('wins', 'Gains'), ('losses', 'Pertes'), ('flats', 'Nuls'),
             ('sum_r', 'Somme R'), ('mean_r', 'Moyenne R'), ('profit_factor_r', 'PF en R')]
    lines = ['# Audit des entrées hybrides, 2020–2025', '',
             'Source : `out.log`, quatre paires, bougies 60 minutes, régression 25 bougies, risque nominal 1 %, '
             'breakout `confirmed`, trailing à 1R. La période couvre **six années civiles**.', '',
             'Un R est la distance entre le prix réellement exécuté et le stop initial. Les comparaisons '
             'utilisent cette unité pour limiter l’effet du capital et du dimensionnement. Le PF du log '
             'est monétaire ; le PF en R peut donc différer légèrement.', '',
             '## Bilan observé', '', table([report['total']], basic), '',
             'Le log classe les cinq sorties à zéro parmi les « Losing trades ». L’audit les sépare des 125 pertes.', '',
             '### Par branche', '', table(report['groups']['branch'], [('branch', 'Branche'), *basic]), '',
             '### Par paire', '', table(report['groups']['symbol'], [('symbol', 'Paire'), *basic]), '',
             '### Par année', '', table(report['groups']['year'], [('year', 'Année'), *basic]), '',
             '### Branche et paire', '', table(report['groups']['branch/symbol'], [('branch', 'Branche'), ('symbol', 'Paire'), *basic]), '',
             '## Contrôles de reconstruction', '',
             f"- Contextes d’entrée uniques retrouvés : {report['provenance']['entry_context_unique_matches']}/{len(frame)}.",
             f"- Clôtures retrouvées au même timestamp : {report['provenance']['baseline_replay_close_matches']}/{len(frame)}.",
             f"- Écart maximal de résultat du rejeu initial : {report['provenance']['baseline_replay_max_r_error']:.12g} R.",
             '- Les canaux sont retrouvés avec les vrais détecteurs et les bougies précédant chaque événement. '
             'Le stop, la cible et le sens doivent correspondre au log ; une correspondance non unique est exclue des mesures de canal.',
             '- Les ratios utilisés comme filtres reposent sur le prix **prévu dans l’ordre**, connu à la soumission. '
             'Le prix exécuté ultérieurement sert uniquement au calcul des résultats en R et des excursions.',
             '- Les excursions et sorties alternatives utilisent les cotations bid/ask chronologiques et le gestionnaire de sortie existant.', '',
             '## Comparaison des caractéristiques', '',
             'Médianes pour les réintégrations. Les métriques MFE et MAE sont connues **après** l’entrée : '
             'elles servent au diagnostic et ne sont jamais des filtres d’entrée. '
             'Le corps et la mèche de rejet sont mesurés sur la bougie mid clôturée du signal.', '']
    medians = []
    comparisons = report['feature_comparisons']['failed-breakout']
    for feature, outcomes in comparisons.items():
        medians.append({'feature': feature, **{outcome: values['median'] for outcome, values in outcomes.items()}})
    lines.extend([table(medians, [('feature', 'Variable'), ('win', 'Gagnants'), ('loss', 'Perdants')]), '',
                  '## Filtres candidats', '',
                  'Les seuils ci-dessous sont exploratoires. Chaque ligne applique **un seul filtre** au flux '
                  'de trades exécutés d’origine. Les filtres propres aux réintégrations laissent les breakouts passer. '
                  'Ils ne recalculent ni le capital, ni les nouvelles occasions libérées par les trades supprimés.', '',
                  'Le découpage 2020–2022 / 2023–2025 est rétrospectif : ces années ne constituent plus un vrai '
                  'échantillon inconnu. Les seuils ne sont pas optimisés sur la seconde période.', ''])
    candidate_rows = []
    for candidate in report['candidates']:
        candidate_rows.append(dict(name=candidate['name'], **candidate['all'],
                                   dev=candidate['development_2020_2022']['mean_r'],
                                   val=candidate['validation_2023_2025']['mean_r'],
                                   removed_r=candidate['removed']['sum_r'],
                                   ci=str([round(v, 3) for v in candidate['mean_r_quarter_block_ci95']])))
    lines.extend([table(candidate_rows, [('name', 'Filtre'), ('n', 'Trades conservés'), ('sum_r', 'Somme R'),
                   ('profit_factor_r', 'PF R'), ('dev', 'R/trade 2020–22'), ('val', 'R/trade 2023–25'),
                   ('removed_r', 'R supprimés'), ('ci', 'IC 95 % R/trade, blocs trimestriels')]), '',
                  'Les intervalles par bootstrap trimestriel sont indicatifs : les petits échantillons et '
                  'la comparaison de plusieurs filtres empêchent d’y voir une preuve de performance future.', '',
                  '### Mèche courte côté stop : découverte et sensibilité', '',
                  'La séparation basse de la mèche, calculée sur les réintégrations 2020–2022, vaut '
                  '0,248946. Le seuil arrondi de 25 % est ajouté comme hypothèse exploratoire ; '
                  'il ne fait pas partie des premiers seuils fixes. Il faut le figer avant une nouvelle validation.', '',
                  'Pour un achat, mesurer `(min(open, close) - low) / (high - low)` ; '
                  'pour une vente, `(high - max(open, close)) / (high - low)`. '
                  'Ces valeurs utilisent la bougie mid clôturée du signal. Aucune couleur de corps supplémentaire '
                  'n’est imposée. Le motif peut refléter la géométrie stop/cible ; une causalité n’est pas établie.', '',
                  table(report['stopside_wick_sensitivity'], [('threshold','Mèche maximale'), *basic,
                       ('development_mean','R/trade 2020–22'),('validation_mean','R/trade 2023–25')]), '',
                  'Cette table porte sur les réintégrations uniquement. Resserrer le seuil augmente ici la moyenne '
                  'mais réduit fortement le nombre de trades : le seuil de 15 % ne conserve que 28 observations '
                  'sur six ans et ne doit pas être choisi pour son seul PF.', '',
                  'Répartition du groupe à 25 % :', '',
                  table(report['short_wick_failed']['by_symbol'], [('symbol','Paire'), *basic]), '',
                  table(report['short_wick_failed']['by_year'], [('year','Année'), *basic]), '',
                  '### Délai de réintégration', '',
                  'Un délai 0 correspond à une excursion et une réintégration sur la même bougie. '
                  'Un délai 1 implique que la clôture de la bougie d’excursion était encore hors du canal, '
                  'puis que la suivante a clôturé dedans. Ces groupes sont connus au moment du signal.', '',
                  table(report['failed_reentry_timing'], [('reentry_bars','Bougies après excursion'),*basic]), '',
                  'Le groupe à une bougie est positif pour les quatre paires et les six années, '
                  'avec seulement 35 observations. Il recoupe fortement le motif de mèche courte : '
                  '26 de ces 35 trades satisfont aussi le seuil de 25 %. Les deux constats ne constituent '
                  'donc pas deux preuves indépendantes.', '',
                  'La combinaison exploratoire `failed_nextbar_or_short_samebar` conserve la réintégration '
                  'sur la bougie suivante, **ou** la réintégration sur la bougie d’excursion lorsque sa mèche '
                  'côté stop est courte. Elle exclut les retours tardifs à 2–3 bougies. '
                  'Elle est construite après inspection et doit être figée puis rejouée avant toute adoption.', '',
                  '## Sorties alternatives des 22 breakouts', '',
                  'Même entrée, même stop initial, mêmes ticks, sans nouvelles entrées. Un horizon de 14 jours '
                  'calendaires borne le rejeu ; les cas sans clôture sont signalés, jamais comptés comme gains. '
                  'Les variantes de trailing utilisent la distance de risque réelle à l’exécution ; '
                  'l’activation d’origine reste celle enregistrée dans l’ordre.', ''])
    lines.extend([table([dict(name=name, **value['all'], censored=value['censored']) for name, value in report['exit_variants'].items()],
                         [('name', 'Sortie'), *basic, ('censored', 'Non clôturés')]), '',
                  '## Coûts et limites', '',
                  'Le bid/ask est déjà pris en compte. Le rejeu ne rajoute pas de commissions, de swap ou de '
                  'glissement supplémentaire. Le stress suivant enlève un coût **additionnel** par trade, sans doubler le spread :', '',
                  table([{'cost':float(cost), 'r':value} for cost,value in report['cost_stress'].items()], [('cost', 'Coût R/trade'), ('r', 'Somme R restante')]), '',
                  'Les PnL de sous-ensembles ne sont pas des rendements de nouveaux backtests. Les paires '
                  'partagent le dollar, les observations peuvent être dépendantes et les paramètres ont déjà été '
                  'examinés sur cet historique. Les données ne contiennent pas de calendrier macroéconomique : '
                  'aucune perte n’est attribuée automatiquement à une annonce.', '',
                  '## Fichiers et reproduction', '',
                  '- `trades_enriched.csv` : une ligne par trade, caractéristiques et excursions.',
                  '- `contexts.json` : contexte causal des entrées retrouvées.',
                  '- `analysis.json` : résultats, intervalles, groupes, empreinte du log et manifeste des ticks.',
                  '- `explorer.html` : graphique interactif local des entrées.', '',
                  'Depuis le dossier parent de `ultimateBot` :', '', '```powershell',
                  'py -m ultimateBot.audit_hybrid_trades', 'py -m ultimateBot.report_hybrid_audit', '```', ''])
    (directory / 'rapport.md').write_text('\n'.join(lines), encoding='utf-8')
    records = frame.copy()
    for column in ('signal_at', 'submitted_at', 'opened_at', 'closed_at'):
        records[column] = records[column].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    payload = records.to_json(orient='records')
    document = '''<!doctype html><html lang="fr"><meta charset="utf-8"><title>Audit hybride 2020–2025</title>
<style>body{font:16px system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#203040;background:#fafbfc}h1{font-size:26px}select{padding:8px;margin:8px}svg{width:100%;background:white;border:1px solid #dce2e8;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:right}th{position:sticky;top:0;background:#f1f4f7}#rows{max-height:450px;overflow:auto}.note{color:#526477;line-height:1.5}a{color:#005fa3}circle{opacity:.75}circle:hover{stroke:#111;stroke-width:2;opacity:1}</style>
<h1>Entrées hybrides : les 202 trades de 2020–2025</h1>
<p class="note">Analyse du flux exécuté d’origine. Les filtres ci-dessous servent à explorer les observations ; ils ne constituent pas un nouveau backtest. Vert : gain ; rouge : perte ; gris : nul. Survoler un point affiche le trade.</p>
<label>Paire <select id="symbol"><option value="">Toutes</option><option>EURUSD</option><option>GBPUSD</option><option>USDCAD</option><option>USDCHF</option></select></label>
<label>Branche <select id="branch"><option value="">Toutes</option><option>failed-breakout</option><option>breakout-retest</option></select></label>
<label>Variable <select id="feature"></select></label><p id="summary"></p><svg id="plot" viewBox="0 0 1000 430"></svg>
<p class="note">Axe vertical : résultat en R. Les variables MFE / MAE sont postérieures à l’entrée et ne peuvent pas servir de filtres causaux.</p>
<h2>Cumul des résultats en R</h2><svg id="equity" viewBox="0 0 1000 300"></svg>
<h2>Trades sélectionnés</h2><div id="rows"></div><p><a href="rapport.md">Rapport détaillé</a> · <a href="trades_enriched.csv">Données CSV</a></p>
<script>const data=PAYLOAD;
const features=['channel_width_atr','channel_drift','excursion_atr','planned_rr','stop_atr','parallelism','signal_body_direction','rejection_wick','efficiency_24','momentum_24_atr','spread_risk','volatility_6_48','hour_utc','mfe_r','mae_r'];
const el=id=>document.getElementById(id);features.forEach(f=>el('feature').add(new Option(f,f)));
const fmt=x=>x==null?'—':Number(x).toFixed(3);const node=(tag,attrs,text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!=null)n.textContent=text;return n};
function chart(svg,points,width,height,line=false){svg.replaceChildren();if(!points.length)return;let xmin=Math.min(...points.map(p=>p.x)),xmax=Math.max(...points.map(p=>p.x)),ymin=Math.min(0,...points.map(p=>p.y)),ymax=Math.max(0,...points.map(p=>p.y));if(xmin===xmax)xmax++;if(ymin===ymax)ymax++;const X=x=>65+(x-xmin)/(xmax-xmin)*(width-90),Y=y=>height-40-(y-ymin)/(ymax-ymin)*(height-65);for(let i=0;i<=5;i++){let y=ymin+(ymax-ymin)*i/5,x=xmin+(xmax-xmin)*i/5;svg.append(node('line',{x1:65,x2:width-25,y1:Y(y),y2:Y(y),stroke:'#e5e9ed'}),node('text',{x:55,y:Y(y)+4,'text-anchor':'end','font-size':12},fmt(y)),node('text',{x:X(x),y:height-15,'text-anchor':'middle','font-size':12},line?new Date(x).toISOString().slice(0,10):fmt(x)));}svg.append(node('line',{x1:65,x2:width-25,y1:Y(0),y2:Y(0),stroke:'#8092a3'}));if(line)svg.append(node('polyline',{points:points.map(p=>`${X(p.x)},${Y(p.y)}`).join(' '),fill:'none',stroke:'#1768a2','stroke-width':2}));else points.forEach(p=>{let c=node('circle',{cx:X(p.x),cy:Y(p.y),r:5,fill:p.y>0?'#16815e':p.y<0?'#c34343':'#888'});c.append(node('title',{},p.label));svg.append(c)})}
function draw(){let selected=data.filter(r=>(!el('symbol').value||r.symbol===el('symbol').value)&&(!el('branch').value||r.branch===el('branch').value)),f=el('feature').value;let total=selected.reduce((s,r)=>s+r.r,0),g=selected.reduce((s,r)=>s+Math.max(0,r.r),0),l=selected.reduce((s,r)=>s-Math.min(0,r.r),0);el('summary').textContent=`${selected.length} trades · ${fmt(total)} R · moyenne ${fmt(total/selected.length)} R · PF R ${l?fmt(g/l):'—'}`;chart(el('plot'),selected.filter(r=>Number.isFinite(r[f])).map(r=>({x:r[f],y:r.r,label:`#${r.trade_id} ${r.symbol} ${r.opened_at}\n${r.branch} ${r.side}\n${f}=${fmt(r[f])} · ${fmt(r.r)} R`})),1000,430);let sum=0;chart(el('equity'),selected.slice().sort((a,b)=>a.closed_at.localeCompare(b.closed_at)).map(r=>({x:Date.parse(r.closed_at),y:sum+=r.r})),1000,300,true);const cols=['trade_id','opened_at','symbol','branch','side','r',f,'mfe_r','mae_r'];let table=document.createElement('table'),head=document.createElement('tr');cols.forEach(c=>{let th=document.createElement('th');th.textContent=c;head.append(th)});table.append(head);selected.forEach(r=>{let tr=document.createElement('tr');cols.forEach(c=>{let td=document.createElement('td');td.textContent=typeof r[c]==='number'?fmt(r[c]):r[c]??'—';tr.append(td)});table.append(tr)});el('rows').replaceChildren(table)}['symbol','branch','feature'].forEach(id=>el(id).onchange=draw);draw();</script></html>'''.replace('PAYLOAD', payload)
    (directory / 'explorer.html').write_text(document, encoding='utf-8')
    print(f'Reports written to {directory.resolve()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).parent / 'artifacts/hybrid_trade_audit')
    render(parser.parse_args().directory)
