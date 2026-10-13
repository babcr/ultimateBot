# Choix adaptatif du lookback, du trailing stop et du filtre global

`--adaptive-strategy` active une recherche aléatoire pendant le backtest.
La configuration initiale reprend la timeframe de `--candle-interval`, le
lookback habituel de la stratégie (`--regression-lookback-bars` en
`swing-regression`, `--level-lookback-bars` en `structure`, ou sa
configuration) et l'activation initiale du trailing (`--trailing-stop`, sinon
désactivé). Le trailing démarre au seuil de `--trailing-start-rr-ratio`,
avant le TP de référence. En mode trailing, les trades peuvent dépasser ce
TP et sortent sur le stop ; en mode sans trailing, le TP reste une sortie
fixe. Le seuil de départ ne varie pas dans la grille adaptative. L'origine
est conservée même lorsqu'elle
est extérieure aux bornes de recherche. Chaque symbole part de cette origine,
puis conserve sa propre configuration, sa fenêtre de précision et sa référence.
Une recherche sur EURUSD ne modifie pas les paramètres d'USDCHF. Les configurations
sont choisies indépendamment ; deux symboles peuvent tirer la même valeur.

Les candidats sont maintenant des n-uplets dépendants du moteur utilisé.
En `swing-regression`, seul le lookback de régression varie. Celui des
supports/résistances conserve `--level-lookback-bars`. Les deux fenêtres
utilisent des indices dans le même historique maximal, dont la capacité
couvre les niveaux, les candidats de régression et la confirmation globale.
Si `--regression-lookback-bars` est omis, l'origine de régression reprend
le lookback des niveaux pour conserver les anciennes commandes.

En structure, l'optimiseur explore le lookback et, par défaut, le mode
`trailing`. En `swing-regression`, il explore aussi, par défaut, l'état de
`use_global_timeframe_validation`. Les options
`--no-adaptive-optimize-trailing-stop` et
`--no-adaptive-optimize-global-timeframe-validation` permettent de retirer
chacune de ces dimensions de la grille adaptative : dans ce cas, leur valeur
reste figée sur l'activation initiale choisie via `--trailing-stop` et
`--regression-global-timeframe-validation`. La timeframe reste stockée par
symbole, initialisée depuis `--candle-interval`, mais l'optimiseur ne la fait
plus varier. Les lookbacks utilisent `range(min, max, step)` :
**min inclus, max exclu**. La borne supérieure exclue suit la taille de vecteur
demandée ; pour inclure 150 avec un pas de 1, utiliser une borne max de 151.
Quand la validation globale fait partie de la grille, elle devient la clé de
tri prioritaire du vecteur, avant le lookback puis le mode de sortie.

Toutes les valeurs par défaut sont définies dans `AdaptiveStrategyConfig`,
puis reprises par le lanceur et son aide. Sans overrides :

| Paramètre | Défaut |
| --- | --- |
| `lookback_min` | 5, inclus |
| `lookback_max` | 20, exclu, donc jusqu'à 19 avec un pas de 1 |
| `lookback_step` | 1 |
| `accuracy_window` | 50 clôtures au maximum |
| `accuracy_drop_percent` | 2 % |
| `accuracy_drop_mode` | `relative` |
| `profit_factor_window` | 50 clôtures au maximum |
| `profit_factor_drop_percent` | 2 % |
| `profit_factor_drop_mode` | `relative` |
| `min_profit_factor` | 0, seuil désactivé |
| `optimization_metric` | `accuracy` |
| `min_accuracy_percent` | 14 % |
| `seed` | 42 |
| `tighten_search_interval` | `True` |
| `search_long_arm_delta` | 1 indice |
| `search_short_arm_delta` | 1 indice |
| `min_risk_percent` | 0,01 % |
| `optimize_risk` | `True` |
| `optimize_trailing_stop` | `True` |
| `optimize_global_timeframe_validation` | `True` |
| `timeframe_min_minutes` | 9, option legacy ignorée |
| `timeframe_max_minutes` | 10, option legacy ignorée |
| `timeframe_step_minutes` | 1 minute, option legacy ignorée |

`--adaptive-strategy` reste désactivé par défaut. Cette seule option suffit
pour activer l'optimisation avec les valeurs ci-dessus, pour structure comme
pour swing-regression. Les bornes peuvent aussi être remplacées séparément.
Le lookback initial reste celui de la stratégie, même hors de la grille.

## Mesure et déclenchement

`--adaptive-optimization-metric` choisit la métrique qui pilote le changement
de configuration. Deux valeurs sont disponibles :

- `accuracy` : proportion de clôtures avec `pnl > 0`.
- `profit_factor` : somme des gains réalisés divisée par la somme absolue des
  pertes réalisées sur la fenêtre glissante.

La précision est la proportion de trades clôturés avec `pnl > 0`. Un
breakeven n'est pas un gain. Elle est calculée sur une fenêtre glissante de
clôtures du même symbole attribuées à sa configuration courante.
Le PnL latent, le MFE et les données futures n'entrent pas dans cette mesure.

Le profit factor adaptatif possède sa propre fenêtre glissante,
`--adaptive-profit-factor-window`. Elle est indépendante de
`--adaptive-accuracy-window`, qui continue à alimenter la précision et le
risque dynamique. Chaque symbole et chaque période de configuration disposent
de leur propre objet `RollingTradeProfitFactor`. Il calcule uniquement le ratio
des PnL réalisés présents dans sa fenêtre :

```text
profit_factor = gains_realises / pertes_realisees_absolues
```

S'il n'existe encore aucune perte dans la fenêtre mais au moins un gain, la
valeur est considérée comme infinie. S'il n'existe aucun gain, elle vaut 0.

En mode `profit_factor`, `--adaptive-profit-factor-drop-percent` et
`--adaptive-profit-factor-drop-mode` remplacent les paramètres de baisse de
précision. Le mode `relative` interprète la valeur comme un pourcentage : une
baisse de 2 % depuis un profit factor de 1,50 déclenche à 1,47. Le mode
`points` l'interprète comme une baisse absolue du ratio : `0.20` déclenche de
1,50 à 1,30. `--adaptive-min-profit-factor` ajoute un seuil absolu ; à égalité
ou en dessous, une nouvelle recherche est lancée. Sa valeur par défaut `0`
désactive ce second déclencheur.

La métrique sélectionnée est définie dès la première clôture du réglage
courant, qui établit la référence. La fenêtre `N` est une capacité maximale.
Avec `k < N` clôtures disponibles, la mesure porte sur ces `k` clôtures ; une
fois la fenêtre remplie, elle porte sur les `N` dernières clôtures.
La référence augmente avec la meilleure valeur observée depuis ce réglage.
Pour une baisse relative de 5 %, une précision de 80 % déclenche le changement
lorsqu'elle atteint 76 % ou moins :

```text
baisse = (précision de référence − précision actuelle) / précision de référence
```

`--adaptive-min-accuracy-percent` reste un seuil absolu de précision. Il agit
comme déclencheur de recherche seulement lorsque `optimization_metric=accuracy`.
Par exemple, `60` signifie 60 %. Dès la première clôture, si le taux du symbole
est inférieur ou égal à ce seuil, une recherche démarre, même si le taux est
stable, en amélioration ou nul. La recherche continue également à égalité :
avec une limite de 60 %, le couple doit dépasser 60 % pour la satisfaire. Le
critère de baisse reste actif. En mode `profit_factor`, ce seuil ne pilote pas
les changements de configuration, mais il continue d'alimenter le risque
dynamique décrit plus bas.

Chaque symbole possède aussi un intervalle inclusif d'indices, initialement
égal au vecteur complet. Lorsqu'un trade de la configuration courante gagne,
sa configuration devient la configuration gagnante et les deux bornes se rapprochent de
son indice, si le resserrement est activé. La borne située sur le bras le plus long avance selon
`search_long_arm_delta` ; la borne du bras court avance selon
`search_short_arm_delta`. La longueur des bras est mesurée entre
le vainqueur et les deux bornes d'origine du vecteur. Une borne arrivée sur le
vainqueur y
reste. Un trade perdant ou à breakeven desserre progressivement les bornes
avec les mêmes pas, sans jamais dépasser les bornes d'origine. Une victoire
sous un autre couple déplace le vainqueur et recentre les mouvements suivants
sur ce nouvel indice. Tous ces états restent indépendants par symbole.
Les options `--adaptive-search-long-arm-delta` et
`--adaptive-search-short-arm-delta` permettent de remplacer ces pas. L'option
`--no-adaptive-tighten-search-interval` désactive complètement ce resserrement:
le vainqueur reste mémorisé, mais l'intervalle de recherche conserve toute la
grille de lookbacks pendant le run.

Lorsqu'au moins un des critères actifs est satisfait, le module tire
uniformément un autre indice uniquement dans l'intervalle courant du symbole,
applique son réglage, puis recommence sa fenêtre de mesure vide. Si
l'intervalle est réduit au seul couple courant, il n'existe aucun candidat à
tirer : ce couple est conservé jusqu'à ce qu'une perte élargisse l'intervalle.
La première clôture de ce
symbole attribuée au nouveau réglage suffit pour reprendre les décisions.
Tant que le taux ne dépasse pas le minimum, les recherches peuvent se répéter
dès la première clôture de chaque nouveau réglage.
La valeur `0` désactive ce seuil absolu ; une référence de 0 %
ne déclenche alors aucune baisse relative.

## Risque dynamique par symbole

Quand l'optimiseur est actif, chaque symbole commence avec
`--adaptive-min-risk-percent`, soit 0,01 % par défaut. `--risk-percent`
devient son plafond. Dès que le symbole possède une précision définie, son
risk suit une interpolation linéaire. Le plafond de précision commence à
`--adaptive-min-accuracy-percent`, c'est-à-dire la limite que le couple doit
dépasser pour arrêter la recherche. Dès qu'elle est dépassée, ce plafond est
remplacé par la meilleure précision atteinte pendant le run par ce symbole :

```text
progression = min(1, précision_courante / plafond_de_précision_du_symbole)
risque = risque_min + progression × (risque_max − risque_min)
```

Avec une limite initiale de 60 %, une précision de 30 % place le risque au
milieu et 60 % atteint le risque maximal, tout en déclenchant encore la
recherche puisqu'elle ne dépasse pas la limite. Si le symbole obtient ensuite
70 %, son plafond devient 70 %. Un retour à 35 % place alors son risque au
milieu. Le meilleur taux et le plafond résultant sont conservés durant les
changements de couple. Les précisions, les records et les risques restent
indépendants entre symboles. Lorsqu'un
symbole change de couple lookback/trailing, sa nouvelle période commence sans
précision et son risque revient au minimum, mais son plafond conserve le
meilleur taux déjà atteint. Les ordres déjà créés conservent
leur quantité ; la nouvelle valeur s'applique aux opportunités suivantes.
Avec `--quantity`, cette valeur sert de plafond de risque à la quantité fixe :
un ordre qui le dépasse au-delà de la tolérance configurée est refusé.
Sans `--adaptive-strategy`, `--risk-percent` reste le risque fixe historique.
`--no-adaptive-optimize-risk` désactive uniquement cette interpolation :
l'optimiseur peut toujours changer le lookback, le trailing stop, le filtre
global et la configuration active, mais le `RiskManager` utilise constamment
la valeur fixe de `--risk-percent`. Dans ce mode,
`--adaptive-min-risk-percent` est ignoré et chaque symbole démarre, continue et
redémarre après un changement avec le même risque fixe.

La seed rend les tirages reproductibles, avec une séquence indépendante pour
chaque symbole. L'activité et l'ordre d'arrivée des autres symboles ne
consomment pas les tirages du symbole observé. Les candidats ne sont pas simulés
en parallèle ni classés par résultat : ce mécanisme explore lors d'une
dégradation, sans garantir de trouver le maximum de précision. Le taux de
réussite et le bénéfice total sont deux mesures différentes.

## Intégration

- `AdaptiveStrategyConfig` définit la grille, la fenêtre, la métrique de
  pilotage, les seuils, les pas de resserrement/desserrement, le risque minimum,
  la présence ou non des dimensions booléennes dans la grille, et la seed.
- `AccuracyRiskScaler` transforme la précision courante en pourcentage de
  risque entre le minimum adaptatif et le plafond `--risk-percent`.
- `RollingTradeAccuracy` conserve les résultats des dernières clôtures.
- `RollingTradeProfitFactor` conserve les PnL de sa fenêtre dédiée et met à
  jour les gains bruts, les pertes brutes et leur ratio en coût constant.
- `SymbolAdaptiveState` conserve le réglage, la précision, la métrique active,
  la période et le générateur aléatoire d'un seul symbole, ainsi que son
  intervalle courant et son indice vainqueur.
- `AdaptiveStrategyOptimizer` surveille la métrique active et applique les
  tirages.
- `ReversalSwingHistory` conserve les candidats de swings par symbole et
  flux de prix. Le premier accès initialise cet historique avec les bougies
  déjà conservées ; ensuite, chaque clôture ne traite que la nouvelle paire
  de bougies, en coût constant amorti. La bougie active n'y participe pas.
- L'interface commune de `Strategy` expose `history_lookback_bars`,
  `get_history_lookback_bars(symbol)`, `history_capacity_for_lookback`,
  `set_history_lookback_bars(bars, symbol=...)` et, quand c'est pertinent,
  `get/set_global_timeframe_validation_enabled(...)`. Le module accepte toute
  stratégie qui implémente les capacités dont elle a besoin ; il ne dépend pas
  d'un moteur de décision particulier.
- Les moteurs structure et régression implémentent cette interface. Le
  moteur structure demande au moins 60 bougies ; la régression au moins 2.
  `dummy`, qui ne consulte pas d'historique de bougies, ne participe pas à
  l'optimisation du lookback.
- `MarketState` ne construit plus la grille complète de timeframes candidates.
  Il conserve seulement la timeframe d'origine du run, ce qui évite le coût de
  maintenance parallèle de plusieurs jeux de bougies. La variable de timeframe
  par symbole reste présente dans la stratégie et dans les métadonnées des
  ordres, mais elle reste initialisée à la timeframe d'origine tant qu'aucun
  code extérieur ne la modifie explicitement.
- `MarketState` utilise des buffers circulaires (`deque`) de même capacité
  fixe pour tous les symboles et les flux bid/ask/mid. Cette capacité est
  calculée une fois au lancement pour couvrir le plus grand lookback candidat,
  le lookback initial, les besoins de volatilité locale et l'historique fixe
  requis par la confirmation sur l'unité de temps supérieure.
  Chaque stratégie reçoit une projection des dernières bougies nécessaires
  au symbole évalué : `get_candles(..., limit=...)` parcourt directement la
  fin du buffer et crée une liste temporaire de références aux bougies.
  Le coût de lecture dépend de la fenêtre demandée, pas de la capacité totale.
  Un changement de lookback conserve le buffer et son historique ; augmenter
  la fenêtre devient donc possible dès que les bougies sont disponibles.
  Pendant le démarrage, le buffer se remplit progressivement, sans bougies
  artificielles ; la stratégie attend sa fenêtre complète de bougies clôturées.
  Seul le lookback local varie de façon adaptative pour toutes les stratégies.
  En `swing-regression`, la timeframe locale reste celle d'origine du symbole,
  mais le filtre global peut désormais être activé ou désactivé par symbole.
  La confirmation sur l'unité de temps supérieure conserve ses règles, son
  lookback configuré au lancement et son besoin d'historique fixe.
  Ses résultats en cache sont conservés lors d'un changement local et ne sont
  actualisés qu'à la clôture d'une nouvelle bougie de cette unité.
  Les caches d'analyse locale sont invalidés uniquement pour le symbole
  dont la timeframe ou le lookback change.
  `--candle-lookback`, qui règle la volatilité
  du gestionnaire de risque, conserve sa valeur.
- Les vecteurs de swings locaux sont projetés sur la fenêtre de bougies
  demandée. Un swing n'est conservé que si les deux bougies qui le confirment
  sont incluses ; son indice est recalé sur cette fenêtre et sa force est
  recalculée avec son ATR actuel. La normalisation s'applique après cette
  projection pour éviter qu'un ancien extrême sorti de la fenêtre masque
  un swing encore valide. Les candidats bruts restent disponibles jusqu'à
  leur sortie du buffer maximal, ce qui permet de réagrandir le lookback.
  La projection parcourt les candidats utiles, sans rescanner toutes les
  paires de bougies ; les calculs d'ATR et de régression restent liés à la
  taille de leur fenêtre. La confirmation sur l'unité supérieure conserve
  son détecteur et sa fenêtre fixes.
- Le mode TP/trailing est figé dans chaque ordre avant le fill, puis transmis
  à la position et au rapport. Les ordres en attente et les positions déjà
  ouvertes conservent ce mode lorsque le réglage change.
- La durée d’un nouvel ordre reste exprimée par
  `--order-expiration-bars`, avec la durée de la timeframe d'origine du
  symbole. Un ordre existant conserve son heure d'expiration lorsque
  l'optimiseur change ensuite de configuration.
- Chaque période de réglage reçoit un `strategy_configuration_id` propre
  au symbole : l'attribution utilise `(symbol, strategy_configuration_id)`. Les
  clôtures des périodes précédentes restent comptabilisées dans le bilan,
  mais ne notent pas le nouveau réglage, même s'il reprend le même couple.
- `ReportCollector` avertit l'optimiseur après chaque clôture enregistrée.
  `BacktestEngine` et le moteur d'audit de volatilité transmettent les mêmes
  métadonnées aux ordres.
- `RiskManager` demande le risque courant du symbole avant chaque sizing. La
  ligne `[trade-created]` affiche le `risk_percent` réellement utilisé et le
  résumé final affiche le risque courant de chaque symbole.

Un changement produit une ligne `[adaptive-strategy]` avec le symbole, le motif
(`accuracy_drop`, `profit_factor_drop`, `accuracy_below_minimum` et/ou
`profit_factor_below_minimum`), la
précision courante, la métrique de pilotage courante, leurs références,
la baisse, l'échantillon utilisé (`sample=k/N`), l'indice choisi, les bornes
de recherche, l'indice vainqueur et l'ancien/nouveau réglage. Les logs de
création, fill et clôture affichent aussi la période et l'état du booléen
`global_validation` du trade. Le résumé final indique le nombre de changements,
la précision courante et la métrique active du réglage courant de chaque
symbole.

## Lancement

Depuis `C:\git`, pour démarrer à 150 bougies avec le trailing actif et explorer
les lookbacks 50, 75, 100, 125 et 150, les deux modes de sortie et, en
`swing-regression`, l'activation ou non du filtre global, indépendamment pour
EURUSD et USDCHF, avec optimisation sur le profit factor :

```powershell
c:\git\.venv\Scripts\python.exe -m ultimateBot.run_backtest --strategy swing-regression --symbols EURUSD USDCHF --start-date 2020-01-01 --end-date 2020-12-31 --candle-lookback 150 --level-lookback-bars 150 --reward-risk-ratio 3. --trailing-stop --adaptive-strategy --adaptive-lookback-min 50 --adaptive-lookback-max 151 --adaptive-lookback-step 25 --adaptive-optimization-metric profit_factor --adaptive-profit-factor-window 50 --adaptive-profit-factor-drop-percent 2 --adaptive-profit-factor-drop-mode relative --adaptive-min-profit-factor 1.0 --adaptive-accuracy-window 50 --adaptive-min-accuracy-percent 60 --adaptive-seed 42
```

Le pas vaut 1 par défaut. La fenêtre conserve au maximum 50 clôtures
(minimum configurable : 1), le seuil vaut 2 % et le mode
de baisse `relative`. `--adaptive-accuracy-drop-mode points` permet aussi une
baisse en points de pourcentage (80 % → 75 % pour 5 points).
