"""Summarize paired opportunity experiments, including displaced primary trades."""
import argparse
import json
from pathlib import Path

import pandas as pd

from ultimateBot.audit_hybrid_trades import grouped_stats,stats,write_json
from ultimateBot.report_hybrid_audit import table


def frame_for(result):
    frame = pd.DataFrame(result['trades'])
    if frame.empty:
        return pd.DataFrame(columns=['symbol','branch','r','pnl','opened_at','closed_at','year','key'])
    for column in ('opened_at','closed_at'):
        frame[column] = pd.to_datetime(frame[column],utc=True,format='ISO8601')
    frame['year'] = frame.closed_at.dt.year
    frame['key'] = [f"{row.symbol}/{row.branch}/{row.side}/{row.context.get('confirmation_bar')}" for row in frame.itertuples()]
    return frame


def render(directory):
    source = json.loads((directory/'results.json').read_text(encoding='utf-8'))
    frames = {name:frame_for(result) for name,result in source['results'].items()}
    summary = {}
    rows = []
    for name,result in source['results'].items():
        frame = frames[name]
        numbers = stats(frame)
        initial = result['balance']-float(frame.pnl.sum())
        summary[name] = dict(all=numbers,by_year=grouped_stats(frame,['year']),
            by_branch=grouped_stats(frame,['branch']),by_symbol=grouped_stats(frame,['symbol']),
            open=result['open'],pending=result['pending'],return_percent=(result['balance']/initial-1)*100,
            additional_cost_0_05r_sum=numbers['sum_r']-.05*len(frame))
        baseline_name = 'quality' if name.startswith('quality-') else 'filtered'
        baseline = frames.get(baseline_name)
        if baseline is not None and name in ('pullback','bounce','quality-pullback','quality-bounce','quality-impulse'):
            primary = frame.loc[frame.branch.isin(['failed-breakout','breakout-retest'])]
            old_keys,new_keys = set(baseline.key),set(primary.key)
            lost = baseline.loc[~baseline.key.isin(new_keys)]
            introduced = primary.loc[~primary.key.isin(old_keys)]
            complement = frame.loc[~frame.branch.isin(['failed-breakout','breakout-retest'])]
            summary[name]['increment'] = dict(primary_retained=len(old_keys&new_keys),
                baseline=baseline_name,baseline_count=len(baseline),
                primary_lost=stats(lost),primary_introduced=stats(introduced),complement=stats(complement),
                net_sum_r_change=numbers['sum_r']-float(baseline.r.sum()),
                closed_trade_multiplier=len(frame)/len(baseline) if len(baseline) else None,
                complement_by_year=grouped_stats(complement,['year']))
        rows.append(dict(name=name,**numbers,return_percent=summary[name]['return_percent'],open=result['open'],
                         stress=summary[name]['additional_cost_0_05r_sum']))
        export = frame.drop(columns=['context'],errors='ignore').copy()
        export.to_csv(directory/f'trades_{name}.csv',index=False)
    write_json(directory/'comparison.json',summary)
    columns = [('n','Trades'),('sum_r','Somme R'),('profit_factor_r','PF R'),('mean_r','R/trade')]
    lines = ['# Comparaison des entrées complémentaires', '',
        f"Période : {source['start']} au {source['end']}. Paires : {', '.join(source['symbols'])}.", '',
        'Toutes les variantes utilisent les mêmes ticks, 60 minutes, régression 25 bougies, risque 1 %, '
        'RR de référence 2,3 et trailing à 1R. Chaque variante possède un portefeuille unique partagé '
        'entre ses branches et ses paires, avec un seul ordre ou une seule position par paire.', '',
        '- `original` : hybride d’origine, confirmations renforcées du breakout.',
        '- `filtered` : sélection des réintégrations à deux cas et spread/stop prévu ≤ 10 %.',
        '- `quality` : sélection des réintégrations seulement ; `spread` : contrôle du coût seulement.',
        '- `pullback` : socle filtré plus repli dans un canal directionnel.',
        '- `bounce` : socle filtré plus rebond intérieur dans un canal peu directionnel.', '',
        'Les variantes `quality-pullback` et `quality-bounce` ajoutent ces candidats au socle de qualité seul.', '',
        '`quality-impulse` ajoute une cassure après compression confirmée par une clôture ultérieure qui reste entièrement hors du canal.', '',
        '## Résultats du rejeu', '',
        table(rows,[('name','Variante'),*columns,('return_percent','Variation du solde %'),('max_drawdown_r','Drawdown R clôturé'),
                    ('stress','Somme R après coût ajouté 0,05R/trade'),('open','Positions ouvertes')]), '',
        'Le PF est normalisé en R. Le drawdown est calculé sur les résultats clôturés ; '
        'il ne mesure pas le drawdown latent. Les résultats des positions encore ouvertes sont exclus '
        'du solde et des sommes R. Le coût ajouté est un stress artificiel en supplément du spread bid/ask déjà simulé.', '']
    for name,values in summary.items():
        lines.extend([f'## {name}', '',table(values['by_year'],[('year','Année de clôture'),*columns]), '',
                      table(values['by_branch'],[('branch','Entrée'),*columns]), '',
                      table(values['by_symbol'],[('symbol','Paire'),*columns]), ''])
        if 'increment' in values:
            increment = values['increment']
            lines.extend([f"Entrées primaires du socle {increment['baseline']} conservées : {increment['primary_retained']}/{increment['baseline_count']}. "
                          f"Entrées primaires retirées : {increment['primary_lost']['n']} "
                          f"({increment['primary_lost']['sum_r']:+.2f}R dans le socle). "
                          f"Autres entrées primaires apparues : {increment['primary_introduced']['n']}. "
                          f"Variation totale : {increment['net_sum_r_change']:+.2f}R. "
                          f"Nombre de trades multiplié par {increment['closed_trade_multiplier']:.2f}.", '',
                          'Branche ajoutée par année :', '',
                          table(increment['complement_by_year'],[('year','Année'),*columns]), ''])
    lines.extend(['## Vérifications et limites', '',
        'Le rejeu calcule les décisions sur toutes les clôtures disponibles. Il ne sélectionne pas '
        'simplement des trades du log précédent. Les événements capables de modifier un ordre ou une '
        'sortie sont exécutés avec les composants habituels du moteur, et les extrema des ticks sautés '
        'sont enregistrés. Les historiques de bougies sont restaurés avant chaque décision.', '',
        'Le contrôle sur janvier 2024 et les quatre paires a retrouvé les mêmes fills et clôtures que '
        'le moteur intégral pour le mode original et le candidat directionnel. Les soldes correspondent '
        'à l’arrondi du log. Les tests synthétiques comparent aussi les excursions, les positions '
        'simultanées, les limites et l’expiration avant exécution.', '',
        'Les dernières confirmations sans tick d’exécution après la fin du flux sont omises. '
        'Le comparateur accéléré exige les paramètres fixes, sans sessions, optimiseur, validateur '
        'ni dataset externe, et assez de places pour les paires sélectionnées.', '',
        'Cette comparaison porte sur un historique déjà examiné. Elle ne constitue pas une validation '
        'hors échantillon indépendante. Les candidats complémentaires conservent leurs paramètres '
        'initiaux ; aucune grille de recherche du meilleur seuil n’a été appliquée.', '',
        'Une branche peut augmenter le nombre de trades tout en dégradant le bilan, ou prendre la '
        'place d’une entrée primaire ultérieure. Les colonnes incrémentales servent à identifier '
        'ces cas. La branche expérimentale reste désactivée par défaut.', '',
        'Configuration et lancement : [regression_quality_and_pullback.md](../../docs/regression_quality_and_pullback.md).', '',
        'Fichiers : `results.json`, `comparison.json`, `trades_<variante>.csv` et `replay.log`.', ''])
    (directory/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print(table(rows,[('name','Variant'),*columns]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path(__file__).parent/'artifacts/regression_opportunities')
    render(parser.parse_args().directory)
