# Prédire les gains des swings avec Informer

Les [collectes à variantes de stop](swing_stop_variants.md) ajoutent un canal de
marge de stop connu au signal, avec un résultat propre à chaque variante.
La validation et le choix du seuil retiennent la première distance acceptée
par swing, avec un seul trade au maximum, comme en backtest.

Le sous-module `Informer2020` référence le dépôt officiel
[zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020), au commit
`29f2a739226a509202a092b464163da81fa74960`. Le code du modèle amont est utilisé
sans modification ; les anciens scripts de chargement des benchmarks ETT ne
sont pas utilisés pour les données de trading.

```powershell
git submodule update --init Informer2020
py -m pip install -r requirements-informer.txt
```

## Ce que le modèle apprend

La cible est le **PnL réalisé en R** de l'opportunité du dernier swing confirmé,
avec les sorties de la collecte : un trade gagnant de 3R a pour cible `3`, une
perte de 1R a pour cible `-1`. Il s'agit d'une régression directe à erreur
quadratique, sans encodage TS2Vec, XGBoost ou conversion en labels 0/1.

Un exemple contient par défaut **64 swings alternés**, soit 32 couples. Cela
désigne un nombre d'événements, pas 64 bougies ni une durée fixe. Le modèle
respecte l'intervalle des bougies de la collecte, qui n'est pas rééchantillonné.
Les confirmations du même type remplacent la précédente ; les confirmations
opposées simultanées réinitialisent le contexte, comme dans le collecteur.

Chaque swing apporte les six variables causales existantes : prix relatif,
rendement de la jambe, type haut/bas, délais entre pivots et confirmations,
délai de confirmation du pivot. Si la collecte contient les volumes, un septième
canal contient `log1p(mean_quote_volume)` : moyenne bid pour un haut, ask pour
un bas. La paire est ajoutée sous forme de canaux indicateurs.
Pour les cinq paires de la collecte actuelle, cela donne **12 canaux d'entrée**.

Les dates UTC de confirmation sont également fournies au modèle : minute,
heure, jour de semaine, jour du mois et jour de l'année. Les écarts de temps
des swings restent explicites, puisque les événements sont irréguliers.

Le modèle est l'encodeur/décodeur Informer d'origine, avec un seul nombre en
sortie. Le décodeur reçoit les dernières variables **observées** et un jeton
vide portant la date du signal actuel. Il ne reçoit **aucun PnL**, ni le label
courant, ni les résultats de trades précédents. Le résultat prédit concerne le
trade du swing actuel, pas le prix du swing suivant.

Les variables et les cibles sont normalisées à partir du train uniquement,
puis les prédictions sont reconverties en R. Un score `0.2` signifie une
prédiction de gain de `0.2R`, pas une probabilité de 20 %.

## Entraîner et valider

Exemple d'un premier modèle compact sur 2022 avec validation sur 2023 :

```powershell
$informerModel = 'C:\TradingBotData\swing_informer\experiments\informer_88a154e4d58a_2022'
py run_swing_informer.py train `
  --run-id 88a154e4d58a4cd7bb0c6c8a10cfa588 `
  --train-start 2022-01-01 --train-end 2022-12-31 `
  --validation-start 2023-01-01 --validation-end 2023-12-31 `
  --d-model 32 --d-ff 64 --e-layers 1 `
  --batch-size 128 --epochs 3 --threads 2 `
  --checkpoint-batches 50 `
  --output-dir $informerModel
```

Ce modèle a 24 289 paramètres : il utilise un contexte identique au modèle
standard, avec une largeur et une profondeur plus petites pour le premier
essai sur CPU. Ces dimensions constituent un point de départ mesuré en coût
de calcul, pas des hyperparamètres déclarés optimaux.

Sans dates explicites, les périodes sont **2015–2021 pour le train** et
**2022–2023 pour la validation**. Le modèle standard utilise `d_model=64`,
deux couches d'encodeur, une de décodeur, quatre têtes, 64 swings, un batch
de 64 et au maximum 10 époques avec patience de 3. Aucun échantillonnage réduit
des données n'est appliqué en silence. `--help` documente les options.

`--symbols EURUSD GBPUSD ...` restreint les paires. `--no-swing-volume` retire
les volumes. `--seq-len` doit être pair, entre 8 et 2 048 ; `--label-len` vaut
par défaut sa moitié. `--device cuda:0` permet de demander CUDA si disponible.
L'intégration a été vérifiée sur CPU.

La validation choisit l'époque avec la plus petite MSE en R, puis un seuil
d'acceptation parmi les quantiles des prédictions, en maximisant la somme des
R avec un effectif minimum pour les collectes à stop unique. Pour les collectes
à stops variables, tous les seuils distincts sont évalués avec la première
distance acceptée par swing, sans minimum de trades acceptés.
**Le seuil ne descend pas sous `--min-threshold`
(0R par défaut).** Si aucune prédiction ne dépasse ce plancher, le modèle
peut accepter zéro trade ; le programme ne baisse pas le seuil pour forcer
des entrées. La ligne finale `[swing-informer-selected]` et `rapport.md`
mettent en évidence le **seuil initial retenu en R**.

Le rapport compare aussi l'erreur à une constante apprise sur le train.
`validation_qualified=True` exige une amélioration de MSE, un résultat positif,
un profit factor d'au moins 1,05 et, pour un stop unique, l'effectif minimum. C'est un diagnostic de
validation, **pas une preuve d'edge** : époque et seuil utilisent cette période.

## Reprise et réutilisation des données

L'extraction lit les colonnes scalaires de `swing_events` en lecture seule,
sans les gros blobs de `swing_contexts`. Elle remonte juste assez loin avant le
début demandé pour construire le contexte initial. Une séquence est stockée
comme des indices vers les swings, pas comme une copie complète de chaque
fenêtre. `dataset.npz` est conservé et n'est pas reconstruit à chaque époque.

Les checkpoints `latest.pt` contiennent les poids, l'optimiseur, l'ordre de
mélange, la position dans l'époque et l'état aléatoire. Ils sont écrits dans
un fichier temporaire, synchronisés et remplacés atomiquement. Une reprise
continue après le dernier batch sauvegardé :

```powershell
py run_swing_informer.py train --resume $informerModel
```

La reprise recharge les dates, paires et paramètres enregistrés. Sur un run
interrompu, on peut modifier le budget total `--epochs`, `--threads`,
`--device` ou `--checkpoint-batches`. Changer l'architecture, les données ou
l'optimiseur demande un autre run. Un modèle déjà figé n'est pas réentraîné
dans son dossier, afin de préserver les backtests qui l'utilisent.

Pour essayer une autre architecture sur exactement les mêmes observations,
sans relire la base ni dupliquer le cache :

```powershell
py run_swing_informer.py train `
  --dataset "$informerModel\dataset.npz" `
  --output-dir 'C:\TradingBotData\swing_informer\experiments\informer_comparison' `
  --d-model 64 --e-layers 2 --epochs 10
```

Le cache fixe les périodes, les symboles, le nombre de swings et les volumes.
Son empreinte est vérifiée. Les checkpoints vérifient aussi l'empreinte du
code amont utilisé. `best.pt` est le meilleur modèle selon la validation ;
`metadata.json` est publié après la finalisation des résultats.

## Test réservé et backtest

Aucun test n'est consulté automatiquement pendant l'entraînement. Les labels
doivent être connus avant la fin de leur période : les trades encore ouverts
à cette borne, les ordres expirés et les résultats manquants sont exclus.
Les historiques de contexte peuvent traverser les frontières puisqu'ils
ne contiennent que du passé.

Pour évaluer une fois le modèle figé sur les opportunités de 2024–2025 :

```powershell
py run_swing_informer.py test --model $informerModel `
  --start-date 2024-01-01 --end-date 2025-12-31 `
  --output-dir "$informerModel\test_2024_2025"
```

Ce test conserve le seuil et les sorties, fournit les résultats par année
et par paire, et n'altère pas le modèle. Il mesure des opportunités
chevauchantes ; les sommes de R ne représentent pas le rendement d'un compte.

Le modèle fonctionne également dans le moteur ordinaire de ticks/portefeuille :

```powershell
py run_backtest.py --strategy swing-informer `
  --swing-informer-model $informerModel `
  --start-date 2024-01-01 --end-date 2024-12-31 `
  --balance 10000 --risk-percent 1 --max-positions 5
```

Le seuil enregistré, les paires, l'intervalle des bougies, l'expiration et les
paramètres de sortie sont repris du modèle. Les bougies antérieures à
`available_after` servent uniquement à initialiser les swings.
Les prédictions ProbSparse utilisent un tirage local fixe à l'inférence pour
éviter que le même contexte obtienne un score différent selon les appels.

Les options `--swing-informer-threshold`, `--swing-informer-min-threshold`,
`--swing-informer-stop-buffer-pips`, `--trailing-start-rr-ratio` et
`--swing-informer-take-profit-r` permettent les mêmes réglages d'exécution que
la stratégie swing existante. Un TP fixe désactive le trailing.
`--informer-threads` contrôle les threads CPU du backtest. Le contrôleur de
cadence est disponible via `--swing-informer-target-trades-per-day` et
`--swing-informer-threshold-step`, mais il modifie le seuil validé ; il n'est
pas activé par défaut. Des sorties différentes changent également le problème
prédit par le modèle.

## Premier entraînement effectué

Le dossier `C:\TradingBotData\swing_informer\experiments\informer_88a154e4d58a_2022`
contient le premier modèle terminé : 40 657 exemples de train en 2022,
40 423 exemples de validation en 2023, trois époques effectuées et meilleure
époque égale à 1.

| Diagnostic en validation | Résultat |
|---|---:|
| Plage des prédictions | −0,633181R à +0,491787R |
| Écart-type des prédictions | 0,155839R |
| MSE du modèle | 2,641607 |
| MSE de la constante apprise sur train | 2,634068 |
| Seuil exploratoire retenu | 0,21719642R |
| Opportunités au-dessus du seuil | 405 |
| Somme des R sélectionnés | −27,886R |
| Profit factor sélectionné | 0,8910 |

**La validation n'est pas qualifiée.** Le modèle différencie les contextes,
mais ces différences ne montrent pas d'amélioration prédictive sur cet essai.
Le seuil reste une sélection exploratoire, pas une recommandation de trading.
Aucun test 2024–2025 ou 2026 n'a été effectué avec ce modèle.

Un essai avec l'architecture standard sur 2015–2021 a été interrompu après
mesure du coût CPU, avant sa première époque complète. Son dataset et son
checkpoint sont conservés dans
`C:\TradingBotData\swing_informer\experiments\informer_88a154e4d58a_2015_2021`.
Il peut être poursuivi sans refaire l'extraction :

```powershell
py run_swing_informer.py train --resume 'C:\TradingBotData\swing_informer\experiments\informer_88a154e4d58a_2015_2021'
```

Vérification depuis le dossier parent de `ultimateBot` :

```powershell
py -m unittest ultimateBot.tests.test_swing_informer ultimateBot.tests.test_swing_ts2vec_robot -q
```
