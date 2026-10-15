# Moteur de régression des swings

Le moteur `swing-regression` coexiste avec `structure`, qui reste le choix par
défaut. Les deux stratégies partagent la construction des bougies fermées et
l'extraction des swings : lows sur l'ask, highs sur le bid. La branche de
régression ajuste ses droites sur ces vecteurs et réutilise le regroupement
en niveaux structurels pour filtrer les swings d'entrée.

## Configuration centralisée

`swing_regression_config.py` est le point d'entrée des paramètres de la stratégie.
L'objet `DEFAULT_SWING_REGRESSION_CONFIG` rassemble les valeurs par défaut ;
`SwingRegressionConfig` permet de préparer une configuration propre à un run.
Les paramètres partagés ont une interface propre à la régression : il n'est plus
nécessaire de modifier `DEFAULT_STRUCTURE_RULES` pour régler cette stratégie.

| Rubrique | Paramètres |
| --- | --- |
| `regression` | Fenêtre, points, score, pentes, distances, densités, volatilité de marché, filtres d'entrée et journal d'analyse. |
| `global_timeframe` | Activation, multiplicateur et seuils de la confirmation globale. |
| `levels` | Fenêtre des niveaux, clustering, touches, qualité et distance de réaction. |
| `candles` | Ratios de corps pour la confirmation et les dojis. |
| `protection` | Source du stop et marge supplémentaire en ATR. |
| `structure_compatibility` | Autres règles héritées des helpers de structure, y compris celles sans effet en régression. |
| `risk` | Risque, capital disponible, exposition, quantité, RR, seuil de trailing et spread. |
| `volatility` | Intervalle commun des bougies et paramètres de volatilité du gestionnaire de risque. |
| `orders` | Délai d'exécution et expiration en bougies. |
| `execution` | Capital initial, taille des lots de ticks, trailing et minimum de bougies propre à la stratégie. |
| `schedule` | Sessions, fraction d'entrée et clôture du vendredi. |
| `optimization` | Activation et paramètres de l'adaptation, configuration et mémoire du bandit. |

Les rubriques sont des dataclasses immuables. Exemple de réglage centralisé :

```python
from dataclasses import replace
from ultimateBot.backtester.strategy.swing_regression_config import SwingRegressionConfig
from ultimateBot.run_backtest import build_engine, parse_args

config = SwingRegressionConfig()
config = replace(
    config,
    regression=replace(config.regression, lookback_bars=150, score_threshold=0.65),
    levels=replace(config.levels, level_lookback_bars=500, min_level_touches=4),
    candles=replace(config.candles, confirmation_body_ratio=0.68),
    protection=replace(config.protection, protective_stop_buffer_atr=1.2),
    risk=replace(config.risk, risk_per_trade_percent=0.5),
    orders=replace(config.orders, expiration_bars=3),
)
args = parse_args(regression_config=config)
engine = build_engine(feed, args)  # feed : flux de ticks du backtest
```

Fournir une configuration à `parse_args` sélectionne `swing-regression` par
défaut. Les options CLI explicites remplacent ses valeurs ; une autre stratégie
explicitement sélectionnée conserve ses propres réglages. Le lancement habituel
avec `--strategy swing-regression` utilise automatiquement
`DEFAULT_SWING_REGRESSION_CONFIG`. Les noms historiques des options, notamment
`--structure-stop-source` et `--structure-stop-buffer-atr`, restent acceptés.

Pour construire seulement le décideur, utiliser
`SwingRegressionStrategy(config=config)`. Pour appliquer également le risque,
les ordres, les horaires et les optimiseurs, utiliser le lanceur comme ci-dessus.
`volatility.candle_interval` est l'intervalle commun du marché, de la stratégie,
du risque et des ordres. `execution.minimum_candles` concerne la stratégie,
tandis que `volatility.minimum_candles` concerne le risque.

`strategy.config` contient les entrées retenues au lancement, options CLI
comprises. Les adaptations en cours de run par symbole restent consultables
avec `rules_for_symbol` et les accesseurs existants. `to_rules()` produit un
nouveau dictionnaire pour les moteurs existants ; ceux-ci conservent leurs
normalisations, notamment les planchers de points et de ratios. La valeur de
confirmation par défaut est `0.15`, qui était déjà la valeur effective après
normalisation du réglage local à `0.08`.

La source est contrôlée avant l'ajustement des droites : les highs marqués ask
et les lows marqués bid sont exclus, y compris dans les vecteurs déjà normalisés
et les candidats d'origine historiques. Les points du bord de fenêtre utilisent
également le high bid ou le low ask de la bougie concernée. Cette convention
s'applique à l'analyse locale et à l'unité supérieure.

Depuis `C:\git`, lancement sur un historique court :

```powershell
.\.venv\Scripts\python.exe -m ultimateBot.run_backtest --strategy swing-regression --symbols EURUSD --start-date 2020-01-02 --end-date 2020-01-07 --candle-interval 30m --regression-live-analysis
```

## Origine et droites

La régression utilise les `regression_lookback_bars` dernières bougies fermées. Parmi
leurs swings confirmés, on retient le high le plus haut et le low le plus bas.
L'extrême le plus éloigné du cours de clôture mid de la dernière bougie devient
l'origine. En cas d'égalité, le plus récent est retenu, y compris pour plusieurs
swings au même prix extrême.

- Origine low : régression des lows contrainte à passer par cette origine,
  et régression libre des highs.
- Origine high : régression des highs contrainte à passer par cette origine,
  et régression libre des lows.

Tous les swings de chaque famille dans la fenêtre participent au calcul, même
ceux précédant l'origine. L'axe x est l'indice de bougie de confirmation du
swing, tel qu'il est fourni par le détecteur partagé. La régression libre optimise
à la fois la pente et l'ordonnée à l'origine. La régression ancrée optimise la
pente sous la contrainte `y = prix_origine + m * (x - indice_origine)`.
Deux indices distincts et au moins trois swings par famille sont nécessaires.
Le minimum effectif est réglé par `regression_min_swings_per_side`.

## Score et déclenchement

Pour les pentes `m1` et `m2`, avec une référence de pente `S` exprimée en
ATR par bougie (`regression_slope_scale_atr`, défaut configuré actuellement 0.20) :

```text
parallélisme = 1 - |m1 - m2| / (|m1| + |m2|)
pente_partagée = (m1 + m2) / 2
force_pente = tanh((pente_partagée / ATR) / S)
score = parallélisme * force_pente
```

Le parallélisme est borné entre 0 et 1 ; il vaut 1 lorsque les deux pentes sont
nulles. Des pentes de signes opposés donnent un parallélisme nul. Le score est
borné entre -1 et 1 : positif pour acheter, négatif pour vendre. Un score nul ou
dont la valeur absolue est sous `regression_score_threshold` produit une décision
neutre. L'ATR reprend l'échelle de volatilité du moteur structurel : la moyenne
des amplitudes high-low sur la fenêtre, avec ses planchers existants.

La pente mesure une variation de prix par bougie. Avec `S = 0.10`, une pente
partagée de 0.10 ATR par bougie donne une force de pente de `tanh(1)`, soit
environ 0.762 avant pondération par le parallélisme. Un seuil de 0.95 exige,
pour deux droites parfaitement parallèles, environ 0.183 ATR par bougie.
Un parallélisme plus faible exige une force de pente plus élevée.

La fenêtre choisit les données utilisées pour ajuster les droites ; sa longueur
ne multiplie plus la pente dans le score. L'ancienne multiplication par `N - 1`
faisait saturer `tanh` sur des tendances ordinaires et rendait un seuil élevé
trop permissif. La référence fixe donne aux seuils une échelle explicite.
Augmenter `regression_slope_scale_atr` diminue la valeur absolue du score pour
les mêmes droites et la même volatilité. Le score est un indice de force
directionnelle du canal.

Une évaluation a lieu à chaque nouveau swing confirmé sur la dernière bougie
fermée. Chaque clôture est traitée une fois par symbole ; une bougie active,
un ancien swing qui change d'indice dans la fenêtre glissante, ou une clôture
sans nouveau swing ne déclenchent pas de signal. Si deux familles sont confirmées
sur la même clôture, une seule décision est produite.

Pendant un run, `ReversalSwingHistory` détecte les candidats locaux une seule
fois, à chaque nouvelle bougie clôturée. La stratégie projette ces candidats
sur son lookback courant : les deux bougies du retournement doivent être
incluses, les indices sont relatifs à la fenêtre et les forces utilisent son
ATR actuel. La normalisation est refaite sur cette projection ; les candidats
bruts restent conservés sur la capacité maximale pour permettre de réduire
puis d'augmenter la fenêtre sans perdre de swings valides.

Avec l'optimiseur adaptatif, seul le lookback local de régression change par symbole.
Le lookback des supports/résistances reste celui configuré au lancement.
L'analyse de confirmation sur l'unité supérieure conserve le lookback et les
règles configurés au lancement, ainsi que ses résultats jusqu'à la clôture
suivante de cette unité, même si la nouvelle fenêtre locale est encore en
cours de remplissage.

### Validation par support ou résistance

Le filtre est actif par défaut. Un achat exige un rebond sur support et une
vente un rejet sur résistance. Les cassures seules ne valident pas une entrée.
Un achat est déclenché par un swing low ask (bougie baissière suivie d'une
haussière), une vente par un swing high bid (haussière suivie d'une baissière).
Le filtre examine le pivot de ce swing d'entrée et sa clôture de confirmation.

- Pour acheter, le low ask du swing d'entrée doit toucher la zone d'un support
  préexistant, après une approche depuis le dessus. La clôture mid du swing
  d'entrée doit ressortir au-dessus de cette zone.
- Pour vendre, le high bid du swing d'entrée doit toucher la zone d'une résistance
  préexistante, après une approche depuis le dessous. La clôture mid du swing
  d'entrée doit ressortir en dessous de cette zone.

La proximité du pivot et la sortie de zone utilisent `entry_distance_atr`
(0.30 ATR par défaut). Le côté de l'approche est vérifié sur la clôture mid
qui précède les deux bougies du pivot de réaction. Un simple passage depuis
l'autre côté du niveau est refusé.

Les niveaux sont calculés avec `build_structure_levels`, comme dans la
stratégie `structure`, sur les swings confirmés strictement antérieurs au
pivot de réaction. Le pivot qui touche le niveau et le swing d'entrée ne
peuvent donc ni créer ce niveau ni compléter ses touches. Les paramètres
`level_merge_atr_ratio`, `level_merge_min_points`, `min_level_touches`
et `min_level_quality` restent ceux du moteur structurel. La fenêtre des
niveaux est celle de `level_lookback_bars`, indépendante de la régression.
Son ATR est calculé sur cette fenêtre pour le clustering et la zone de réaction.

Le même swing doit passer le filtre de distance à sa droite de régression et
ce filtre de réaction. Sans niveau suffisamment établi, ou si la clôture
reste dans la zone, aucune entrée n'est produite. La raison de refus est
`trigger_swing_without_level_reaction`. `trigger_level_reactions` détaille le
pivot de départ, les niveaux et la cause du refus ; `entry_level_reaction`
contient le niveau et le style retenus en cas de validation. Le journal
`--regression-live-analysis` affiche aussi la validation, le style et le prix
du niveau. Le bilan compte les refus liés aux supports/résistances.

Ce filtre concerne les entrées locales. L'analyse de tendance de l'unité
supérieure reste indépendante d'un rebond/rejet local. Pour comparer avec
la régression seule, utiliser `--no-regression-level-reaction`.

### Confirmation par la forme de la bougie

Le paramètre structurel `confirmation_body_ratio` était hérité par la
régression, mais ne filtrait pas ses entrées. Le contrôle est désormais
disponible au lancement avec :

```powershell
--regression-candle-confirmation --confirmation-body-ratio 0.68
```

Sur la dernière bougie fermée qui confirme le swing, le moteur réutilise
exactement la classification de la stratégie `structure` :

- Achat sur low ask : bougie ask haussière, corps/amplitude au moins égal au
  seuil configuré et clôture à au moins 55 % de l'amplitude depuis le low.
- Vente sur high bid : bougie bid baissière, corps/amplitude au moins égal au
  seuil et clôture à au plus 45 % de l'amplitude depuis le low.

Le rapport est `abs(close - open) / (high - low)` : `0.68` signifie un corps
d'au moins 68 % de l'amplitude totale. Aucun motif englobant ou marteau n'est
imposé. La bougie précédente et celle de confirmation ont les couleurs opposées
exigées par le détecteur de swing. Les bougies actives sont exclues ; on ne
diffère pas le signal à une clôture ultérieure.

Le contrôle est désactivé par défaut. `--no-regression-candle-confirmation`
le désactive explicitement. Le seuil accepte une valeur finie entre `0.15`
et `1`, comme la règle structurelle existante, et reprend sa valeur configurée
par défaut (`0.68`). `--confirmation-body-ratio` configure aussi la stratégie
`structure`. Avec l'adaptation, ce filtre et son seuil restent ceux du
lancement, même lorsque le lookback change par symbole.

Ce contrôle s'ajoute à la distance du swing à sa droite et, lorsqu'il est
activé, au rebond/rejet sur support/résistance. Il ne s'applique pas à l'analyse
de tendance de l'unité supérieure. Les analyses stockent la bougie, sa source,
le sens de confirmation et le résultat. Le journal `--regression-live-analysis`
affiche le rapport corps/amplitude et le résultat ; le bilan compte les refus
`trigger_candle_not_confirmed` et `missing_confirmation_candle`.

Les signaux passent par le gestionnaire de risque et l'exécution habituels :
exposition, spread, taille, SL et TP. Une décision dans le sens de la tendance
n'implique donc pas forcément la création d'un ordre, notamment si une position
est déjà ouverte sur le symbole. Le regroupement des niveaux intervient dans
le filtre d'entrée ; la direction et la protection restent calculées par la
régression, puis le gestionnaire de risque dimensionne le trade.

## Protection et réglages

En mode `structure_level`, le niveau protecteur est la projection de la droite
des lows pour un achat, ou des highs pour une vente, sur la dernière bougie.
Si cette projection n'est pas du côté protecteur du prix, le moteur utilise le
dernier swing de cette famille situé du bon côté. Sans protection valide, il
ne produit pas de signal. En mode `recent_swing`, il choisit directement ce
dernier swing. Le gestionnaire de risque ajoute ses marges habituelles, puis
`protective_stop_buffer_atr * ATR`, et recalcule la taille et le TP.

Options du lanceur :

Trois fenêtres indépendantes interviennent dans le backtest :

- `--regression-lookback-bars` pilote les swings utilisés par les droites,
  l'origine de régression, les densités de swings et l'ATR de la régression.
  Si cette option est omise, la fenêtre reprend celle des niveaux pour
  conserver le comportement des commandes existantes.
- `--level-lookback-bars` remplace `level_lookback_bars` dans les règles de la
  stratégie. Il pilote les supports/résistances et leur ATR. Sans cette
  option, la stratégie reprend la valeur définie dans le fichier.
- `--candle-lookback` pilote l'historique de volatilité du gestionnaire de
  risque : marges de stop, TP, taille de position, filtres de spread et décalage
  des ordres. Il ne remplace pas la fenêtre de la stratégie.

Le stockage des bougies est commun aux deux stratégies : un seul historique
maximal par symbole et source de prix (bid, ask, mid). Les candidats de swings
sont eux aussi lus une seule fois dans un vecteur commun. Pour un vecteur de
longueur `N`, les indices de départ sont `N - regression_lookback_bars` et
`N - level_lookback_bars`, avec la même borne de fin `N`. Les pivots à deux
bougies sont sélectionnés avant normalisation, afin qu'un ancien extrême
extérieur à une fenêtre ne masque pas un pivot situé à l'intérieur.
Les indices des swings restent ceux du vecteur commun.

La capacité est au moins le maximum des deux fenêtres actives. Elle inclut
aussi la fenêtre de volatilité requise et l'historique nécessaire à la
confirmation globale. Avec l'optimiseur, elle couvre dès le départ la plus
grande fenêtre de régression autorisée : changer de fenêtre déplace les
indices, sans recréer l'historique ni perdre les données plus anciennes.
La fenêtre globale de régression reste fixée au lancement ; elle n'utilise
pas le lookback des niveaux locaux.

Exemple avec 150 bougies pour la régression, 500 pour les niveaux et 100 pour
le risque, depuis `C:\git` :

```powershell
.\.venv\Scripts\python.exe -m ultimateBot.run_backtest --strategy swing-regression --regression-lookback-bars 150 --level-lookback-bars 500 --candle-lookback 100
```

Le run affiche séparément les fenêtres au démarrage. Les indices utilisés
sont visibles dans `last_analysis` (`regression_window_start`,
`level_window_start`, `window_end`) et dans le journal d'analyse.

| Option | Effet |
| --- | --- |
| `--strategy swing-regression` | Sélectionne ce moteur. |
| `--level-lookback-bars N` | Fenêtre des supports/résistances, minimum 2 pour le moteur combiné ; minimum 60 en `structure` seul. |
| `--regression-lookback-bars N` | Fenêtre de régression indépendante, minimum 2. Sans option, reprend celle des niveaux. |
| `--regression-min-swings N` | Minimum par famille, plancher 3 ; défaut issu de la configuration du moteur. |
| `--regression-score-threshold X` | Seuil absolu du score, défaut configuré actuellement 0.3. |
| `--regression-slope-scale-atr X` | Référence de pente positive en ATR par bougie, défaut 0.20. Plus elle est élevée, plus le score est sélectif. |
| `--regression-live-analysis` | Affiche origine, parallélisme, pentes, score, R² des deux droites et motif de décision. |
| `--regression-level-reaction` / `--no-regression-level-reaction` | Active (par défaut) ou désactive la validation par rebond sur support / rejet sur résistance. |
| `--structure-stop-source structure_level\|recent_swing` | Source de protection décrite ci-dessus. |
| `--structure-stop-buffer-atr X` | Marge supplémentaire en ATR, reprend la configuration structurelle. |

La dernière analyse par symbole est également disponible dans
`strategy.last_analysis`, avec les deux droites, leurs nombres de points,
R² et erreurs résiduelles, l'origine horodatée et les swings déclencheurs.
Le R² décrit l'ajustement et peut être négatif pour la droite ancrée ; il ne
constitue pas un filtre supplémentaire. Les valeurs par défaut sont dans
`DEFAULT_SWING_REGRESSION_CONFIG`, dans `swing_regression_config.py`.
`DEFAULT_SWING_REGRESSION_RULES` reste une vue de compatibilité en dictionnaire.
