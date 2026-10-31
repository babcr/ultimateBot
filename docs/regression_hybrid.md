# Entrées hybrides : failed-breakout et breakout-retest

Les filtres issus de l’audit et la branche directionnelle optionnelle sont
décrits dans [regression_quality_and_pullback.md](regression_quality_and_pullback.md).

Le mode `--regression-entry-mode hybrid` fait tourner les deux détecteurs dans
un seul backtest, sur le même portefeuille. Il conserve leurs filtres de canal
et leurs confirmations. Il cherche des occasions dans les deux scénarios :
réintégration après une fausse cassure et continuation après un retest.

## Lancement

Depuis le dossier parent de `ultimateBot`, exemple reprenant EURUSD 2024 et
les paramètres du dernier essai :

```powershell
py -m ultimateBot.run_backtest --strategy swing-regression --symbols EURUSD --start-date 2024-01-01 --end-date 2024-12-31 --candle-interval 30m --candle-lookback 150 --reward-risk-ratio 2.3 --level-lookback-bars 150 --regression-lookback-bars 15 --risk-percent 1 --regression-entry-mode hybrid --regression-breakout-confirmation-mode confirmed --trailing-stop --trailing-start-rr-ratio 1.5
```

`confirmed` conserve la confirmation renforcée du breakout-retest. Il faut
l'indiquer explicitement : le réglage général par défaut reste `legacy`.
Les paramètres `--regression-reentry-*` et `--regression-breakout-*` règlent
leurs branches respectives ; les paramètres de qualité du canal sont communs.

Le mode utilise des paramètres fixes. Retirer `--adaptive-strategy` et
`--contextual-bandit` s'ils figurent dans une commande existante : leur
activation est refusée, comme pour le breakout-retest seul. Les options
secondaires d'optimisation n'activent pas l'adaptation à elles seules.

## Arbitrage et exposition

Chaque branche conserve son propre événement et son canal figé. Lorsque les
deux ont besoin d'un nouveau canal à la même clôture, elles partagent un seul
ajustement, calculé sur les bougies précédant l'événement. Une branche en
attente continue avec son canal initial, même si le nouvel ajustement de
l'autre branche échoue.

La première confirmation sélectionne l'entrée et efface les événements des
deux branches pour cette paire. Cette consommation a lieu même si la session,
le gestionnaire de risque ou le validateur refuse ensuite l'ordre : l'autre
branche ne sert pas d'entrée de secours sur le même setup.

Si les deux branches confirment en sens opposés sur la même clôture, les deux
signaux sont rejetés et leurs événements effacés. Si elles confirment dans le
même sens, une seule entrée est sélectionnée, avec priorité à `failed-breakout`.

Une paire ayant déjà un ordre en attente ou une position ouverte ne peut pas
produire une nouvelle entrée. Ses événements sont effacés pendant cette
occupation. Le moteur vérifie aussi cette règle avant l'envoi de l'ordre.
Les autres paires peuvent continuer à signaler des entrées, dans les limites
de risque et d'exposition du portefeuille partagé.

## Sorties

| Branche sélectionnée | Stop initial | Sortie gagnante |
| --- | --- | --- |
| `failed-breakout` | Extrême de l'excursion et marge ATR figée | Centre du canal à la confirmation, figé comme objectif ; trailing désactivé pour cet ordre. |
| `breakout-retest` | Extrême du retest et marge ATR figée | Trailing si `--trailing-stop` est actif ; sinon objectif fixe selon `--reward-risk-ratio`. |

Avec le trailing, le RR de référence et `--trailing-start-rr-ratio` sont
traités par le gestionnaire de sortie habituel. Activer le trailing ne modifie
pas la sortie vers le centre du canal des réintégrations. Sans le flag,
les deux branches gardent des objectifs fixes selon leurs règles propres.

## Diagnostic et validation

Le bilan `Hybrid diagnostics` sépare, pour chaque branche, les confirmations
brutes, les signaux sélectionnés, les trades clôturés, les gains, le PnL et les
positions encore ouvertes. Il compte aussi les conflits, les doublons et les
clôtures bloquées par une exposition sur la paire. Un signal sélectionné peut
être refusé ensuite ou son ordre expirer ; ce compteur n'est donc pas un
compteur de trades exécutés.

Un rejeu sur les ticks EURUSD du 22 au 26 juillet 2024, avec la commande
ci-dessus limitée à ces dates, s'est terminé avec un trade breakout-retest
clôturé au stop, aucune entrée failed-breakout, aucun conflit et aucun ordre
ou position restant. Ce contrôle vérifie le fonctionnement ; il ne mesure
pas un gain de fréquence sur l'année. Son journal est conservé dans
[`artifacts/regression_hybrid_smoke/eurusd_2024-07-22_2024-07-26.log`](../artifacts/regression_hybrid_smoke/eurusd_2024-07-22_2024-07-26.log).

Le contexte d'entrée conserve `entry_mode` égal à la branche sélectionnée et
ajoute `strategy_entry_mode="hybrid"` et `hybrid_selected_branch`. Les exports
d'ordres conservent ainsi l'origine du signal.

Les tests couvrent les confirmations concurrentes, les canaux indépendants,
les refus de risque, l'occupation par symbole et un cycle complet du moteur
avec les deux branches, avec et sans trailing. Le nombre de trades hybrides
n'est pas nécessairement la somme des deux modes isolés : l'arbitrage et
l'exposition peuvent supprimer des occasions. La rentabilité reste à mesurer
sur des périodes distinctes de celles utilisées pour ajuster les règles.

Voir les règles détaillées dans [regression_reentry.md](regression_reentry.md)
et [regression_breakout.md](regression_breakout.md).
