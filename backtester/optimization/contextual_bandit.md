# Optimiseur contextuel persistant

`--contextual-bandit` active un LinUCB distinct par action et symbole pour la
stratégie `swing-regression`. Une action contient le lookback de régression,
l'activation du trailing existant et le supplément de stop en ATR. Le lookback
des supports/résistances, le timeframe, le risque et les paramètres du trailing
restent fixes. Le TP conserve le calcul du moteur ; en mode trailing, il reste
une référence et la position termine sur son stop. L'ancien optimiseur
`--adaptive-strategy` reste disponible et les deux activations sont exclusives.

## Lancement

Depuis `C:\git`, avec l'environnement Python du dépôt :

```powershell
.\.venv\Scripts\python.exe -m ultimateBot.run_backtest --strategy swing-regression --symbols EURUSD --history D:\jforex_history --start-date 2020-01-02 --end-date 2020-01-02 --contextual-bandit --bandit-memory C:\git\ultimateBot\artifacts\bandit_memory\learning.sqlite3
```

Le même lancement recharge et enrichit le modèle existant. Les dates, le chemin
des données, le mode et le solde ne séparent pas les mémoires. Des données ou des
coûts exprimés dans des unités incompatibles nécessitent un autre contrat, donc
une autre identité. Un run sans `--contextual-bandit` n'accède pas au stockage du
bandit. Un run activé peut explorer des actions après les clôtures, même si le
PF est inférieur à 1,01 : ce nombre n'est pas un filtre du nouvel optimiseur.

`--session-filter` limite les entrées aux premiers 75 % des séances Londres et
New York. `--close-friday` ajoute la sortie hebdomadaire à 23h59 UTC. Ces règles
font partie de l'identité quand elles sont activées et s'appliquent aussi aux
comptes shadow. Voir [les horaires et les limites d'exécution au tick](../monitoring/trading_schedule.md).

| Option | Valeur initiale / comportement |
| --- | --- |
| `--bandit-memory` | `ultimateBot/artifacts/bandit_memory/learning.sqlite3`, commun aux modes compatibles |
| `--bandit-read-only` | Désactivé ; sélection après clôture avec le modèle chargé, sans apprentissage ni écriture |
| `--bandit-alpha`, `--bandit-lambda`, `--bandit-seed` | 1, 1 et 42 ; lambda strictement positif |
| `--bandit-warmup` | 20 contextes de clôture valides et distincts ; observations conservées puis rejouées |
| `--bandit-context-bars` | Fenêtre existante `--regression-market-volatility-window-bars`, actuellement 30 |
| `--adaptive-lookback-min/max/step` | Domaine existant `[5,20)` avec pas 1 ; ces options n'activent pas l'ancien optimiseur |
| `--bandit-lookback-grid` | Liste CSV facultative dans le domaine entier autorisé |
| `--bandit-lookback-grid-size` | Jusqu'à 9 valeurs, extrêmes et baseline si celle-ci appartient au domaine |
| `--bandit-stop-grid`, `--bandit-stop-grid-size` | CSV facultatif, sinon 5 coefficients entre 0 et `--structure-stop-buffer-atr` inclus |
| `--bandit-shadow-limit` | 6 moteurs virtuels au total, positions en cours de terminaison comprises ; 0 désactive le shadow |
| `--bandit-shadow-weight` | Budget total 0,25 par groupe, pendant toute sa durée |
| `--bandit-reward-clip` | R borné à ±5, puis divisé par 5 |
| `--bandit-feature-clip` | Features continues standardisées bornées à ±10, masques inchangés |
| `--adaptive-profit-factor-window` | 15 trades, PF glissant existant |
| `--bandit-group-trades` | 0 signifie la fenêtre du PF, donc 15 clôtures principales |
| `--bandit-group-max-bars` | 0 signifie contexte × fenêtre PF, donc 450 chandelles ; garde temporelle séparée |
| `--bandit-report` | JSON ; sinon `artifacts/bandit_reports/<run_id>.json` |
| `--bandit-run-id`, `--bandit-resume` | Identifiant explicite et reprise du checkpoint du même run |
| `--bandit-repeat-weight` | Absent : déduplication normale. Option explicite `(0,1]` pour réentraîner une répétition historique |

Le `lookback_max` est **exclu**. Une grille peut par exemple être `5,10,19`,
mais pas `5,10,20` avec les bornes initiales. Un maximum de supplément nul réduit
la dimension stop à une seule valeur. Le maximum effectif par défaut est 90
actions ; une grille explicite peut en contenir davantage. La baseline hors
domaine sert seulement à l'amorçage et possède une statistique d'audit distincte.

## Intégration et unités

| Composant | Point d'intégration |
| --- | --- |
| `bandit_core.py` | Identités, grilles, contextes immuables, scaler, matrices et récompense |
| `contextual_bandit.py` | Coordinateur, attribution aux ordres/positions, clôtures et décisions par lot |
| `bandit_store.py` | SQLite partagé, déduplication, journal de clôtures, statistiques et politique atomiques |
| `bandit_shadow.py` | Stratégies, comptes et exécutions virtuels isolés ; pool limité avec terminaison normale |
| `bandit_checkpoint.py` | Checkpoints JSON à types autorisés, restauration du marché et des moteurs |
| `bandit_broker.py` | Deals broker autoritatifs, partiels, reprise et réconciliation sans transport d'ordres |
| `run_backtest.py:build_engine` | Activation, résolution des valeurs effectives, historique maximal partagé |
| `BacktestEngine.on_tick` | Marché, stops principaux, shadow, clôtures en lot, puis prochaines entrées |
| `CandleBuilder.update` | Sommes et nombres de volumes bid/ask valides, en conservant les prix existants |
| `SwingRegressionStrategy.rules_for_symbol` | Lookback local et supplément ATR courants ; niveaux et confirmation globale fixes |
| `RiskManager.evaluate` | Placement et sizing existants, puis contraintes instrument optionnelles |
| `TradeCloser.close`, `ReportCollector.add_trade` | Partiels, résultat net natif et événement de clôture complète |

Le contexte est une fenêtre fixe de chandelles **closes**. Ses six valeurs sont
la pente de la droite de référence existante divisée par l'ATR (ATR/chandelle),
la volatilité ATR/prix, et quatre volumes transformés par `log1p`. La moyenne
tick est somme des samples valides / nombre de samples valides. La moyenne par
chandelle est la moyenne des sommes de ces samples, sur les chandelles disposant
de volumes pour ce côté. Une chandelle sans sample n'invente pas un zéro. Les
six masques et le terme constant donnent 13 dimensions. Une absence de volume
masque ses features ; une tendance/histoire insuffisante entraîne un repli
journalisé. Le scaler utilise seulement des contextes passés de clôture, se
fige après le warmup, et ne dérive plus pendant l'apprentissage.

Le bridge Java écrit `ITick.getBidVolume()/getAskVolume()` sans conversion ;
`JForexTickSource` conserve les colonnes TSV correspondantes. Il s'agit des
volumes **disponibles à la quote**, pas des transactions exécutées, conformément
à [ITick, documentation Dukascopy](https://www.dukascopy.com/client/javadoc3/com/dukascopy/api/ITick.html).
Le contrat conserve les unités natives ITick, sans attribuer une devise ou une
échelle non documentée. Une autre source fournit explicitement
`volume_contract` au coordinateur ; une échelle différente ne partage pas
silencieusement les mêmes features.

`protective_stop_buffer_atr` était déjà un coefficient ATR dans les deux
stratégies. La distance supplémentaire est `k × ATR` de l'analyse close au
placement initial. L'exécution différée conserve ce plan ; elle ne recalcule
pas le stop avec un ATR futur. Le risque appris est figé au **fill réel du
moteur**, avec sa quantité et son stop initial, avant breakeven ou trailing.
Le PnL natif reste `(sortie − entrée) × quantité` avec le sens du trade. Les
quotes bid/ask appliquent déjà le spread ; le moteur historique ne déduit pas
de commissions/funding supplémentaires. Aucune commission fictive n'a été
ajoutée. Le nom `engine_price_quantity` décrit ces unités sans prétendre qu'il
s'agit automatiquement de dollars pour tous les symboles.

`InstrumentRules`, injecté dans `RiskManager.instrument_rules_provider`, permet
un tick size, une distance minimale, un pas de quantité, ses bornes et un
multiplicateur de conversion en devise du compte. La protection structurelle
est arrondie vers l'extérieur ; le supplément est arrondi **vers le bas**
séparément pour rester sous sa borne ATR. Le sizing utilise la distance
effective et la conversion, arrondit la quantité vers le bas et refuse une
quantité imposée incompatible. Les coefficients donnant le même supplément
arrondi sont représentés par un seul bras admissible lors de la sélection.
La conversion doit être fournie par l'intégration : aucune cotation FX ou règle
broker n'est devinée par le backtester.

## Attribution, simulations et persistance

Les positions retiennent le contexte de **sélection** pour apprendre ; les
contextes d'entrée et de sortie sont des diagnostics distincts. Une clôture
partielle ne déclenche aucune sélection. Les clôtures d'un même tick sont
triées puis apprises avant une seule sélection par symbole. Une position
ouverte conserve ses paramètres, même lorsque la politique change.

LinUCB utilise des résolutions par Cholesky sans inversion explicite. La
[référence de Li et al.](https://arxiv.org/abs/1003.0146) décrit l'algorithme ;
la récompense en R, le clipping, le budget shadow et leur usage en trading sont
des choix d'ingénierie. Le bonus d'incertitude n'est pas une probabilité de gain.
Le risque nul/inconnu/non fini et les sorties forcées `END_OF_BACKTEST` restent
en audit, avec poids d'apprentissage nul.

Chaque variante produit ses **propres signaux** avec son lookback. Elle reçoit
le même marché et un historique déjà observé ; elle instancie `ExecutionModel`
virtuel et ne reçoit jamais le client principal. Chaque portefeuille conserve
ses limites de risque/positions. Les groupes expirent après le nombre de trades
ou de chandelles configuré. Ils cessent leurs nouvelles entrées, annulent les
ordres non remplis et laissent les positions finir naturellement ; celles-ci
occupent encore un slot. Les rotations ont lieu après clôture. Si aucun slot
n'est libre, une nouvelle variante attend une clôture.

Pour K variantes apprenantes et budget B, le j-ième trade de chacune reçoit
`(B/K) × 2^(-j)`, j ≥ 1. Le budget inutilisé n'est pas réalloué. La copie de
l'action principale sert au diagnostic et n'apprend pas deux fois. Si un
scénario shadow a été appris avant son équivalent exécuté, sa contribution est
retirée et les statistiques sont reconstruites. Le PnL, les PF, le drawdown,
l'exposition et le latent shadow figurent séparément dans le JSON ; ils ne
s'ajoutent jamais au résultat principal.

SQLite utilise WAL, `synchronous=FULL`, un timeout de verrou et
`BEGIN IMMEDIATE`. Chaque écrivain relit le dernier modèle partagé. Observation,
matrices, RNG et politique par run sont enregistrés dans la même transaction.
Le dernier deal et son observation en attente sont journalisés atomiquement
avant cette transaction : une interruption permet de reprendre une clôture
sans la perdre ou l'apprendre deux fois. Les identifiants incorporent la
trajectoire effective et l'empreinte des ticks, pas seulement un nouveau run ID.
Les checkpoints comprennent comptes, positions, ordres, marché, stratégies et
shadow. Ils sont écrits sur événements de trade/groupe, pas à chaque tick.
Les features d'une même chandelle close sont mises en cache.

La reprise `--bandit-run-id <id> --bandit-resume` exige le même flux complet et
les mêmes bornes de dates. Elle vérifie son préfixe par hash et compte les ticks,
ce qui conserve aussi les ticks simultanés de plusieurs symboles. Une mémoire
corrompue, verrouillée ou de version incompatible provoque un échec explicite,
avec journal `.errors.jsonl` si possible ; elle n'est jamais supprimée.
Le schéma initial est version 1. Un changement de contrat exige une identité
nouvelle ou une migration explicite.

## Administration

```powershell
.\.venv\Scripts\python.exe -m ultimateBot.backtester.optimization.bandit_memory --memory C:\git\ultimateBot\artifacts\bandit_memory\learning.sqlite3 inspect
.\.venv\Scripts\python.exe -m ultimateBot.backtester.optimization.bandit_memory --memory C:\git\ultimateBot\artifacts\bandit_memory\learning.sqlite3 export --output C:\git\bandit_export.json
.\.venv\Scripts\python.exe -m ultimateBot.backtester.optimization.bandit_memory --memory C:\git\ultimateBot\artifacts\bandit_memory\imported.sqlite3 import --input C:\git\bandit_export.json
.\.venv\Scripts\python.exe -m ultimateBot.backtester.optimization.bandit_memory --memory C:\git\ultimateBot\artifacts\bandit_memory\learning.sqlite3 reset --model-id <SHA256_COMPLET>
```

L'import refuse de remplacer un modèle existant : importer dans une autre base
ou cibler explicitement le reset du modèle concerné. Il ne mélange pas les
statistiques de deux scalers figés différents. `inspect/export` sont en lecture
seule. La base SQLite, ses fichiers WAL/SHM et les rapports de runs ordinaires
sont exclus de Git.

## Démo et production : raccordement explicite

Le dépôt ne contient ni transport d'ordres live, ni client broker de positions,
ni service de conversion FX. `BrokerBanditBridge` fournit le composant manquant
pour **intégrer les événements d'apprentissage**, sans prétendre fournir ces
services externes. Le coordinateur accepte `demo_executed`, `live_real` ou
`development_executed`, avec le même store et la même identité quand les
contrats effectifs sont compatibles ; la CLI de replay ne peut pas étiqueter
ses fills comme réels.

L'hôte construit/binde le coordinateur avec son moteur et ses contrats, prépare
ses ordres via `prepare_order`, puis fournit à `register_position` le snapshot
d'origine, l'identifiant broker, le fill, la quantité, la devise et le risque
monétaire initial confirmé. `BrokerBanditBridge.on_market_tick` met à jour le
marché et le shadow **sans** appeler les fills ou clôtures virtuels principaux.
`process_deals` reçoit un lot de `BrokerDeal` : net autoritatif, gross facultatif,
devise identique au risque, IDs stables et flag final. Il agrège les partiels
sans redéduire les frais. Les snapshots broker terminés sont conservés pour
les replays de deals après redémarrage. L'hôte reste responsable de son compte,
de l'application effective des ordres/stops et de ses bilans de portefeuille
réels ; les métriques de cash du moteur de replay ne remplacent pas son ledger.

Au redémarrage, appeler `reconcile_positions` avec les IDs réellement ouverts,
puis rejouer les deals manquants depuis le broker. Une position absente sans
deal final devient `awaiting_authoritative_close_deals`, pas un trade fermé
inventé. Le shadow n'a aucune interface d'envoi d'ordres. Aucun compte ou trade
réel n'est nécessaire à la validation locale de ces contrats.

## Vérification reproductible

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ultimateBot/tests -q
.\.venv\Scripts\python.exe -m ultimateBot.backtester.optimization.validate_contextual_bandit --history D:\jforex_history --symbol EURUSD --date 2020-01-02 --sample-minutes 120
```

Le validateur lance quatre processus séparés : bandit désactivé, apprentissage
initial, rechargement de la même mémoire, et baseline au lookback maximal
autorisé (19). Chaque run utilise six slots shadow sauf le témoin désactivé.
Il publie logs, JSON, temps CPU/mural, pic mémoire du processus et empreinte
identique des données dans `artifacts/contextual_bandit_validation`. Si les
fichiers historiques manquent, `--synthetic` fournit un replay reproductible,
explicitement marqué synthétique. Ces petits runs démontrent le fonctionnement,
pas une performance financière hors échantillon.

Rejouer une mémoire ayant appris sur/après le début de la période est autorisé,
mais le rapport signale `knowledge_at_or_after_run_start=true` et
`independent_validation=false`. Les runs normaux apprennent ; une validation
indépendante nécessite une mémoire antérieure, figée en lecture seule, et une
période nouvelle. Le bilan de validation local est dans
`artifacts/contextual_bandit_validation/validation.md`.
