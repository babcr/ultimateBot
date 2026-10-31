# Valideur XGBoost à la création d'un ordre

Le modèle apprend à partir des trades clôturés dont les caractéristiques ont été
figées **à la création de l'ordre**. Il peut ensuite accepter ou refuser un nouvel
ordre avant sa soumission. Le fill est géré par la plateforme ; il ne déclenche
ni prédiction, ni recalcul d'embedding, ni refus a posteriori.

## Entraîner

Depuis le dossier `ultimateBot` :

```powershell
py -m pip install -r requirements-validator.txt
py train_trade_validator.py
```

La base par défaut est `C:\TradingBotData\trade_embeddings.sqlite3`. L'entraînement
lit un snapshot transactionnel des clôtures déjà validées, y compris celles des
runs de collecte en cours. Il ne prend aucun trade ouvert ni capture historique
au fill. `--completed-runs-only` restreint la source aux runs terminés.

Après le changement du point de capture, attendre les nouvelles données :

```powershell
py train_trade_validator.py --wait-for-data
```

Le processus vérifie la base toutes les 30 secondes et entraîne une seule version
dès que les minima sont atteints. Il conserve les mêmes exigences de séparation
et de classes ; il ne réduit pas automatiquement les minima pour produire un modèle.

Chaque entraînement crée un répertoire neuf sous
`C:\TradingBotData\models\trade_validators\<date_identifiant>`. Il contient :

- `model.ubj` : modèle natif XGBoost avec sa meilleure itération.
- `metadata.json` : contrat de caractéristiques, encodeur compatible, seuil,
  cible, période, paramètres, empreintes et provenance des runs sources.
- `report.json` : résultats apprentissage, validation et test, avant/après sélection.
- `predictions.csv` : scores individuels et partition de chaque observation.

La table `trade_validator_models` enregistre les versions et leurs rapports dans
la même base SQLite. Un répertoire existant n'est jamais écrasé.

## Cible et sélection

Les entrées sont exclusivement les 320 coordonnées TS2Vec, le lookback de
régression, le trailing (0/1) et le side (-1/1). Ni les prix futurs, ni le PnL,
ni les dates de close ne font partie des entrées.

La cible par défaut vaut `pnl_r > 0`, avec `pnl_r = pnl / risque_initial_au_fill`.
`--target-r 1` permet d'apprendre la probabilité d'un résultat supérieur à 1R.
La classe négative signifie « ne dépasse pas la cible », pas nécessairement une perte.
Le score de classification n'est pas une garantie ni une probabilité calibrée.

Les jours UTC de création sont répartis chronologiquement : 60 % apprentissage,
20 % validation, 20 % test, pour toutes les seeds ensemble. Les labels et fenêtres
d'entrée qui chevauchent la partition suivante sont purgés, avec les dates réelles
de début de contexte, y compris les week-ends. Les répétitions d'un même contexte,
symbole et action sont pondérées pour ne pas multiplier leur influence par seed.

L'arrêt anticipé utilise uniquement la validation. Le seuil automatique maximise
le total pondéré en R sur cette même validation, avec au moins 10 observations
effectives acceptées. La grille inclut 0 : si le filtrage n'améliore pas cette
mesure, le modèle peut recommander de laisser passer tous les ordres. Ce résultat
est indiqué explicitement dans les rapports. `--threshold` fixe un seuil manuel
d'entraînement ; il ne doit pas être choisi en regardant les résultats de test.

Le test est réservé au rapport final. Les minima par défaut après purge sont
100 observations effectives d'apprentissage, 30 de validation, 30 de test, et au moins cinq
observations effectives de chaque classe en apprentissage et validation.

Options utiles :

```powershell
py train_trade_validator.py --cutoff 2021-12-31T23:59:59+00:00 --target-r 0 --output-dir C:\TradingBotData\models\validator_2021
```

`--embedding-model-id` et `--campaign-id` sélectionnent une source. Un mélange de
plusieurs encodeurs est refusé. La collecte continue pendant la lecture et le
calcul ; l'empreinte des lignes retenues identifie le snapshot utilisé.

## Utiliser pour les futurs ordres

Ajouter à la commande habituelle du backtest :

```powershell
--trade-validator C:\TradingBotData\models\trade_validators\<version>
```

Le lanceur `run_embedding_sweep.py` accepte aussi `--trade-validator` et
`--trade-validator-threshold`, avec une plage de seeds et des dates postérieures
à `available_after`.

L'encodeur par défaut est
`C:\TradingBotData\models\ts2vec_order_creation.pt`. Le modèle et sa fenêtre doivent
correspondre exactement à ceux de l'entraînement. Un ancien valideur entraîné
au fill est incompatible et refusé au chargement.

L'ordre est construit avec ses paramètres effectifs, puis le bot capture le
contexte et calcule le score **avant** de le placer dans la file d'envoi. Un refus
ne crée ni ordre soumis, ni position, ni label artificiel. Il n'est pas retenté
à chaque tick. Un ordre accepté peut être exécuté plus tard ; sa capture initiale
reste inchangée malgré l'évolution des bougies, du lookback ou du trailing.

Les décisions sont consignées dans
`C:\TradingBotData\validator_decisions\<identifiant>.jsonl`, avec un résumé voisin.
`--trade-validator-log` change le chemin. `--trade-validator-threshold` permet une
surcharge explicite du seuil pour un backtest séparé.

Le modèle ne peut être utilisé qu'après `available_after`, qui inclut les dernières
clôtures consultées pour apprentissage, choix du seuil et évaluation. Un backtest
antérieur échoue au lieu d'utiliser le futur. La stratégie adaptative classique
est compatible ; l'apprentissage parallèle du bandit contextuel n'est pas pris en
charge avec ce valideur.

Pour une future passerelle de trading, appeler `TradeValidator.evaluate` avec la
capture TS2Vec et les paramètres de l'ordre, puis soumettre l'ordre seulement si
`decision.accepted` est vrai. Les notifications de fill doivent rattacher la
capture existante avec l'identifiant de l'ordre de la plateforme.

## Interpréter les rapports

Les mesures comparent une sélection historique de trades déjà observés. Elles ne
constituent pas un rejeu de portefeuille : refuser un ordre change le capital,
les positions disponibles et la trajectoire de la stratégie adaptative. La suite
consiste à tester le modèle sur une période ultérieure avec le filtre réellement
activé. Les données récoltées après filtrage ont aussi une sélection différente
de celles d'une collecte exploratoire sans filtre.

## Vérifier

Depuis le dossier parent `TradingBot` :

```powershell
py -m unittest ultimateBot.tests.test_trade_validator ultimateBot.tests.test_trade_embeddings
```

Les tests entraînent et rechargent réellement XGBoost, vérifient la séparation
chronologique, l'absence d'influence des labels de test sur le modèle/seuil, le
refus avant soumission, la capture identique au close et l'absence de prédiction
au fill. Référence de persistance : [format natif JSON/UBJSON de XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).
