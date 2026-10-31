# Embeddings TS2Vec et dataset de trades

Les caractéristiques et la décision XGBoost sont prises **à la création de
l'ordre, avant soumission à la plateforme**. Le fill rattache cette capture à
la position, sans calcul d'embedding ni validation supplémentaire. Voir
[l'entraînement et l'utilisation du valideur](xgboost_trade_validator.md).

Le lanceur `run_embedding_sweep.py` reprend la configuration demandée, sur
USDCHF, GBPUSD, USDCAD et EURUSD, du 01/01/2020 au 31/12/2023.
Il lance un processus de backtest indépendant par seed, séquentiellement.

Depuis le dossier `ultimateBot` :

```powershell
py -m pip install -r requirements-embeddings.txt
py run_embedding_sweep.py --seed-start 1 --seed-end 10
```

Les deux bornes de seed sont incluses. Pour répéter la plage plusieurs fois :

```powershell
py run_embedding_sweep.py --seed-start 1 --seed-end 10 --passes 3
```

Chaque passage ajoute des runs avec de nouveaux identifiants, même si une seed a
déjà été exécutée. Le même modèle TS2Vec est réutilisé. Pour vérifier les commandes
sans lancer de calcul ni créer de fichiers, ajouter `--dry-run`.

## Fichiers sur C:

- Base : `C:\TradingBotData\trade_embeddings.sqlite3`.
- Encodeur et normalisation : `C:\TradingBotData\models\ts2vec_order_creation.pt`.
- Logs par campagne, passage et seed : `C:\TradingBotData\logs\<campaign_id>\`.
- `manifest.json` dans ce dossier : commandes exactes, PID des backtests,
  identifiants des runs, fichiers de log et état de chaque exécution.

Les chemins se changent avec `--trade-database`, `--ts2vec-model` et `--log-dir`.
Le lanceur s'arrête au premier échec, sauf avec `--continue-on-error`.
Un Ctrl+C arrête la campagne ; les clôtures déjà insérées restent en base.
Une campagne interrompue se relance en créant de nouveaux runs ; la reprise au
milieu du flux de ticks n'est pas implémentée pour ce dataset.

Un test court sur les données réelles peut utiliser une base et un modèle séparés :

```powershell
py run_embedding_sweep.py --seed-start 1 --seed-end 2 --end-date 2020-01-03 --trade-database C:\TradingBotData\validation_orders\trades.sqlite3 --ts2vec-model C:\TradingBotData\validation_orders\ts2vec.pt --log-dir C:\TradingBotData\validation_orders\logs
```

Un lancement direct conserve tous les arguments habituels de `run_backtest.py` ;
ajouter `--trade-database C:\TradingBotData\trade_embeddings.sqlite3` active la collecte.
Sans cet argument, le backtest conserve son fonctionnement habituel et ne charge
pas PyTorch. La collecte est destinée à `--strategy swing-regression`.

## Ce qui est calculé

Le papier local [TS2Vec](../universal_rep_of_time_series.pdf), sections 2 et 3 et
annexe C, décrit un encodeur entraîné sans labels avec contrastes temporels et
entre instances, à plusieurs échelles. Une agrégation max des représentations
temporelles donne un vecteur de taille fixe pour une fenêtre.

L'intégration utilise directement `fit` et `encode(encoding_window='full_series')`
du sous-module, sans modifier ses sources. Les imports historiques `models` et
`utils` sont isolés dans un espace de noms privé.

Par défaut, le contexte comprend les **150 dernières bougies mid clôturées**, à
l'intervalle de base du bot (10 minutes dans la configuration demandée). Cette
fenêtre reste fixe lorsque le lookback adaptatif change. Si moins de 150 bougies
sont disponibles, toutes les bougies clôturées disponibles sont utilisées, avec
un minimum de deux ; leur nombre réel est conservé. Aucun tick futur ni bougie
active n'entre dans ce contexte.

Chaque bougie est décrite par quatre ratios logarithmiques OHLC au close de la
bougie précédente, multipliés par 10 000. Pour la première bougie de chaque fenêtre,
la référence est son open. Ces valeurs décrivent les variations et les amplitudes
à une échelle comparable entre paires. La moyenne et l'écart-type par canal sont
estimés sur les seules fenêtres d'entraînement, puis figés.

L'encodeur produit **320 valeurs** par défaut, avec les dimensions cachées 64 et
la profondeur 10 du code officiel. L'inférence ne fait aucun apprentissage.
L'agrégation max est appliquée uniquement à une fenêtre connue à la création de l'ordre ;
elle peut combiner toutes les bougies de cette fenêtre sans utiliser le futur du
trade. L'option upstream `causal=True` seule ne rend pas son mode non glissant causal.

### Premier entraînement et versions

Si le checkpoint n'existe pas, la première création d'ordre déclenche un apprentissage de
200 itérations, uniquement avec les bougies déjà connues des symboles présents.
Des fenêtres d'au plus 64 bougies, chevauchantes, sont extraites des historiques
bornés à 150 bougies par symbole. Ce bootstrap n'utilise aucun PnL.
Le modèle et son scaler sont ensuite enregistrés et figés pour toutes les seeds.
L'apprentissage restaure les états aléatoires du processus pour préserver ceux du bot.

Ce bootstrap offre un encodeur réellement entraîné et une collecte utilisable
immédiatement. Il repose sur un petit échantillon initial ; la qualité des
représentations devra être évaluée sur des périodes ultérieures. Pour entraîner
une autre version, choisir un nouveau chemin de modèle et conserver la séparation
chronologique entre entraînement de l'encodeur et observations à prédire.

Le checkpoint contient les poids, le scaler, les dates d'entraînement, les
hyperparamètres, les versions des dépendances et une empreinte des sources upstream.
Son SHA-256 identifie le modèle en base. Un checkpoint entraîné après l'instant
d'une décision est refusé. Les dimensions, la fenêtre et l'intervalle doivent aussi
correspondre. Les runs sans aucun ordre peuvent terminer sans modèle associé.

Les options directes sont `--ts2vec-context-bars`, `--ts2vec-output-dims`,
`--ts2vec-hidden-dims`, `--ts2vec-depth`, `--ts2vec-train-iters`, `--ts2vec-seed`,
`--ts2vec-device` et `--ts2vec-threads`. La seed de l'encodeur est distincte de
`--adaptive-seed` et reste à 1 dans le lanceur de campagne.

## Organisation de SQLite

| Table / vue | Contenu |
| --- | --- |
| `embedding_models` | Identité du checkpoint et métadonnées d'entraînement/scaler. |
| `runs` | Seed, campagne, modèle, dates d'exécution, état, résumé et éventuelle erreur. |
| `run_parameters` | Snapshots JSON séparés : arguments et défauts, règles effectives, configuration de stratégie, risque, volatilité, horaires, ordres, adaptation, encodeur, provenance du code. |
| `run_symbols` | Symboles de chaque run. |
| `order_feature_snapshots` | Capture durable à la création, et rattachement au trade si exécuté. |
| `pending_trade_features` | Capture de création rattachée au fill, sans PnL. |
| `closed_trades` | Une observation complète par position clôturée, identifiée par `(run_id, trade_id)`. |
| `training_samples` | Vue des trades clôturés enrichie avec seed, campagne et état du run. |

À la **création de l'ordre**, l'embedding et l'action sont figés. Le lookback
correspond à celui qui a généré l'ordre ; il est conservé si l'adaptation change
avant le fill. Le trailing est le mode prévu pour cet ordre. `side=1`
signifie BUY, `side=-1` signifie SELL ; `trailing` vaut 0 ou 1.

Au **close**, une transaction insère la ligne complète dans `closed_trades` et
retire la capture en attente. Le vecteur explicatif est :

```text
[embedding_000, ..., embedding_319, regression_lookback, trailing, side]
```

La ligne conserve aussi le symbole, les dates de création, fill/close et du contexte, le nombre
de bougies, la configuration adaptative, l'intervalle, les prix initiaux, le stop
initial, le seuil d'activation du trailing, la quantité et le risque initial.
`pnl` est le résultat réalisé renvoyé par le bot, dans sa convention comptable
(`account_currency` et `profit_multiplier` sont sauvegardés). Il n'introduit pas
de nouveaux frais, slippage ou conversions. `pnl_r = pnl / initial_risk` permet
de comparer les résultats malgré les variations de taille et de capital.
`won` vaut 1 pour un PnL strictement positif.

Les clôtures partielles sont agrégées par le collecteur existant ; l'observation
est écrite à la clôture finale. Une notification finale répétée ne crée pas de
doublon. Les positions ouvertes à la fin du backtest restent en attente et ne
reçoivent aucun label artificiel. Les écritures de trades sont validées à chaque
close ; les captures à la création sont également validées immédiatement (WAL,
`synchronous=FULL`).

## Export pour un arbre

```powershell
py export_trade_dataset.py --output C:\TradingBotData\exports\trades.csv
```

L'export écrit une colonne par dimension d'embedding et les trois paramètres
d'action, puis les résultats et les métadonnées. Il utilise les runs terminés et
exige un seul modèle d'embedding ; sélectionner `--model-id` et/ou `--campaign-id`
si plusieurs versions ou campagnes sont présentes.

Pour un premier modèle supervisé, sélectionner explicitement les colonnes
`embedding_*`, `regression_lookback`, `trailing` et `side` comme entrées. `pnl`,
`pnl_r` ou un seuil de rentabilité servent de cibles. Le symbole peut devenir une
variable catégorielle supplémentaire. Les dates de close, le résultat et les
identifiants techniques ne sont pas des caractéristiques disponibles à la création.

Séparer apprentissage et évaluation **chronologiquement pour toutes les seeds
ensemble**. Le même épisode de marché peut apparaître dans plusieurs seeds et
passages : ce ne sont pas des observations indépendantes. Les positions dont la
clôture dépasse une frontière de découpage doivent être écartées de la partie
d'entraînement concernée. Le futur filtre devra être évalué sur des épisodes de
marché hors apprentissage. Le [valideur XGBoost](xgboost_trade_validator.md)
exploite ces captures et peut filtrer les ordres avant soumission.

`feature_point='order_creation'` identifie les nouvelles observations. Les anciennes
captures portent `feature_point='fill'` et sont exclues de l'export et de
l'entraînement du valideur. Elles restent conservées. Les ordres non exécutés
ou expirés n'ont pas de PnL et ne deviennent pas des exemples perdants.

`feature_observed_at` et `order_created_at` sont l'instant effectif de décision
du moteur. `signal_at` conserve l'horodatage du signal, qui peut correspondre au
dernier tick de la bougie précédente. Le cutoff de l'embedding est l'horloge du
moteur à la création, pas l'horodatage décalé de ce signal.

## Réutiliser l'encodeur depuis le bot

```python
from ultimateBot.backtester.embeddings.ts2vec_service import TS2VecService

# Charge le checkpoint partagé après son premier entraînement.
encoder = TS2VecService(candle_interval=engine.market_state.candle_interval)
snapshot = encoder.embed_market(engine.market_state, "EURUSD", as_of=tick.timestamp)
vector = snapshot.values
```

Le service de collecte est aussi accessible pendant un run via
`engine.trade_dataset.encoder`. Pour d'autres consommateurs, `embed_candles`
encode une liste de bougies clôturées ; `fit_histories` peut initialiser une
nouvelle version à partir d'historiques passés et d'un cutoff explicite.

## Vérification

Depuis le dossier parent `TradingBot` :

```powershell
py -m unittest ultimateBot.tests.test_trade_embeddings
```

Les tests exécutent le vrai apprentissage et l'inférence TS2Vec avec un petit
réseau, puis vérifient le rechargement, le cutoff temporel, la capture à la création,
les clôtures partielles, les doublons, les erreurs de flux, les seeds répétées et
l'export CSV. Les tests historiques du bot restent applicables.
