# Fréquence du breakout-retest confirmé

L'étude reprend la commande EURUSD du 1er janvier au 31 décembre 2024 :
régression sur 15 bougies, historique des niveaux et du risque sur 150 bougies,
risque de 1 %, RR de référence 2,3, mode `confirmed`, sans sessions.
L'intervalle effectif est de 30 minutes, faute de `--candle-interval` explicite.

Le trailing n'est pas activé : les sorties utilisent un TP fixe. Les options
`--adaptive-*` ne déclenchent pas l'adaptation sans `--adaptive-strategy`.
Ce mode refuse d'ailleurs l'adaptation et le bandit. Les filtres de réaction
aux niveaux et de tendance globale sont désactivés par ce mode d'entrée.

## Résultats des essais

Les critères de corps, de position de clôture, de pénétration du canal,
de reprise de l'extrême antérieur et de distance d'entrée restent identiques.
Le RR, les règles de risque et les sorties restent ceux de la commande.

| Régression | Délai du retest / confirmation | Trades | Gains / pertes | PnL total en R | Profit factor en R |
| --- | --- | ---: | ---: | ---: | ---: |
| 15 bougies | 3 bougies partagées, réglage initial | 2 | 1 / 1 | +1.490 | 2.461 |
| 15 bougies | 3 bougies puis 3 après le premier contact | 3 | 1 / 2 | +0.334 | 1.153 |
| 15 bougies | 6 bougies puis 3 après le premier contact | 4 | 1 / 3 | -0.671 | 0.789 |
| 30 bougies | 3 bougies partagées | 2 | 0 / 2 | -2.266 | 0 |

Les quatre variantes terminent sans position ni ordre ouverts. Le réglage
initial retrouve les deux trades rapportés par l'utilisateur : achat du
22 juillet, puis achat du 18 octobre. Les délais prolongés ajoutent des
achats le 29 août, puis le 17 juin ; ces deux ajouts sont perdants.

La fréquence augmente avec les délais, mais la qualité mesurée se dégrade
sur cet échantillon. Doubler la fenêtre de régression ne suffit pas non plus.
Les défauts restent donc inchangés. Deux trades ne permettent pas de valider
un filtre, et ces essais sur 2024 ne constituent pas une validation hors
échantillon. Aucun réglage n'est présenté comme optimal.

## Nouveau délai séparé

`RegressionBreakoutParameters.confirmation_max_bars`, exposé par
`--regression-breakout-confirmation-max-bars`, vaut **0 par défaut** : cela
conserve exactement l'échéance commune précédente.

En `confirmed`, une valeur C positive donne deux étapes :

1. Premier contact avec la zone avant t+`max_bars`, borne incluse.
2. Confirmation sur une bougie ultérieure, au plus tard C bougies après
   ce premier contact, borne incluse.

Les retests successifs ne remettent pas le compteur à zéro. Les invalidations
restent actives pendant toute l'attente et le canal doit rester ouvert sur
l'horizon prolongé. `max_bars=1` devient possible en `confirmed` si le délai
séparé est positif. En `legacy`, le nouveau délai est ignoré.

Pour reproduire le test à trois trades, ajouter à la commande initiale :

```powershell
--regression-breakout-max-bars 3 --regression-breakout-confirmation-max-bars 3
```

Pour le test à quatre trades, remplacer le premier 3 par 6. Ces réglages
sont des variantes d'étude ; les résultats ci-dessus ne justifient pas de
les promouvoir par défaut.

Les diagnostics distinguent `breakout_retest_expired` de
`breakout_confirmation_expired`. Le journal et le contexte d'ordre conservent
la bougie du premier contact, le délai de confirmation et son échéance.

## Reproduction de l'étude

Depuis le dossier parent de `ultimateBot` :

```powershell
py -m ultimateBot.audit_breakout_frequency
py -m ultimateBot.replay_breakout_signals
```

Le premier outil agrège les quotes synchrones bid, ask et mid en bougies,
puis appelle la stratégie existante à chaque clôture. Les extrêmes du mid
sont calculés sur les mid de chaque tick, jamais à partir des extrêmes bid
et ask séparés. Le cache comporte les tailles et dates de modification des
fichiers sources ; un changement invalide sa réutilisation. Cet audit compte
les signaux et les acceptations du risque avec portefeuille vide ; ces
dernières ne sont pas à elles seules des fills.

Le second outil rejoue tous les signaux sur les ticks originaux, en utilisant
le gestionnaire de risque, les ordres, la file d'attente, l'exécution, le
portefeuille et le closer du moteur. Il saute uniquement les ticks sans
exposition, ordre en attente ni décision. À chaque décision, il restaure
l'historique de volatilité observé à cet instant, y compris la bougie active.
Les ordres restent exécutables seulement sur un tick ultérieur à la décision.
Ce replay est réservé à la configuration fixe et au symbole de cet audit ;
il ne remplace pas un moteur adaptatif, le bandit ou la capture de datasets.

Le PnL en R utilise le risque initial réel au fill. Le drawdown rapporté
porte sur les R réalisés, pas sur l'equity intratrade. Les positions et ordres
encore ouverts sont comptés séparément. Spread et gaps suivent le modèle
existant ; aucune commission ni aucun financement supplémentaires ne sont
modélisés.

Rapports et journaux :

- `artifacts/regression_breakout_timing/frequency.json` : détection sur 2024.
- `artifacts/regression_breakout_timing/replay.json` : exécution des variantes temporelles.
- `artifacts/regression_breakout_context30/replay.json` : fenêtre de 30 bougies.

La variante de contexte se reproduit avec :

```powershell
py -m ultimateBot.audit_breakout_frequency --regression-lookback-bars 30 --variants 3:0 --candle-cache ultimateBot/artifacts/regression_breakout_timing/eurusd_2024-01-01_2024-12-31_30m.json --output ultimateBot/artifacts/regression_breakout_context30
py -m ultimateBot.replay_breakout_signals --audit ultimateBot/artifacts/regression_breakout_context30/frequency.json
```

133 tests ciblés passent. Ils couvrent les échéances inclusives, l'absence de
prolongation par des contacts successifs, la causalité des confirmations,
le canal sur l'horizon prolongé et la compatibilité des réglages précédents.
Les bougies agrégées sont comparées aux trois `CandleBuilder` réels ; les
replays de signaux sont comparés au moteur complet sur des fixtures couvrant
un fill market avec gap, une entrée limit différée, une expiration et un
signal issu de la dernière clôture sans tick futur.
