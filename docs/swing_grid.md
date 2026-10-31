# XGBoost directement sur une grille de swings

Le mode `--representation swing-grid` remplace l'encodage TS2Vec par le codage
défini pour cette expérience. Le modèle principal reçoit **uniquement le vecteur
des prix normalisés des swings**, sans embedding, volume, indicateur de présence
ou variables tabulaires supplémentaires.
Les collectes à plusieurs variantes de stop ajoutent désormais une entrée
`stop_buffer_ratio` : voir [les paramètres et le codage](swing_stop_variants.md).
Pour ces collectes, validation, choix du seuil et comparaison des fenêtres ne
retiennent qu'un trade par swing : la première distance acceptée, dans l'ordre
croissant des marges, comme en backtest.
La collecte diffère désormais le windowing par défaut : choisir les durées
lors de `train` ou `search`. Les grilles sont préparées dans leur cache à cette
étape, sans blobs de contexte génériques en base.

## Formule et axe temporel

Pour une fenêtre terminant au signal `t`, de durée `D`, et un intervalle de
bougie `w`, le vecteur comporte `D / w` cases, de la plus ancienne à la plus
récente. `D` doit être un multiple exact de l'intervalle collecté.

Pour les swings confirmés dans `(t − D, t]` :

```text
moyenne = somme(prix des swings présents) / nombre de swings présents
case avec swing = prix du swing / moyenne − 1
case sans swing = 0
```

La moyenne est donc **la moyenne arithmétique des prix des swings de la fenêtre**.
Les trous n'y participent pas. Elle est recalculée pour chaque exemple ; il
ne s'agit ni d'une moyenne sur les bougies ni d'un normaliseur global appris
sur toutes les dates. Aucune donnée postérieure au signal n'entre dans ce calcul.

L'axe temporel utilise la **confirmation du swing**, comme les fenêtres du
collecteur existant. La dernière case contient le swing du signal courant.
Pour des confirmations sur les cases 0, 2 et 5, avec les prix 100, 130 et 160 :

```text
moyenne = 130
vecteur = [100/130 − 1, 0, 0, 0, 0, 160/130 − 1]
```

La case 2 est ici aussi zéro : un swing exactement égal à la moyenne et une
case vide ont la même valeur, conformément au codage demandé. Aucun masque
supplémentaire n'est ajouté.

Sur des bougies de **20 minutes** : 72h donnent **216 cases**, 96h donnent
**288 cases**. Le nombre réel de swings peut varier sans modifier ces dimensions.
Les fenêtres en nombre de couples sont refusées, car elles n'ont pas une
durée fixe. Les swings alternent ; une confirmation du même type remplace
la précédente et les confirmations opposées simultanées réinitialisent le
contexte, selon les règles du collecteur.

Les matrices fournies à XGBoost restent **denses** : les zéros sont des valeurs
numériques, pas des valeurs manquantes. Une conversion en matrice sparse
changerait ce comportement, comme expliqué dans la
[FAQ officielle XGBoost](https://github.com/dmlc/xgboost/blob/master/doc/faq.rst).

## Normaliser sur toutes les clôtures de bougies

L'option `--grid-normalization candle-close-mean` utilise, pour chaque symbole
et chaque fenêtre, la moyenne arithmétique des clôtures médianes de **toutes**
les bougies closes, y compris celles qui ne contiennent aucun swing :

```text
clôture médiane d'une bougie = (clôture bid + clôture ask) / 2
moyenne = somme des clôtures médianes / nombre de bougies présentes
case avec swing = prix du swing / moyenne − 1
case sans swing = 0
```

Chaque bougie pèse autant, quel que soit son nombre de ticks. Une clôture
« médiane » désigne ici le milieu bid/ask au dernier tick de la bougie, pas
la médiane statistique de ses ticks. Les bougies dont la clôture appartient
à `(t − D, t]` participent à la moyenne. La bougie active et les prix ultérieurs
à la confirmation sont exclus. Les périodes sans cotation, dont les fermetures
de marché, ne produisent aucune bougie artificielle et ne comptent pas comme
des prix nuls. Les symboles sont normalisés séparément.

Pour comparer 6h, 12h et 24h sur la collecte existante :

```powershell
py run_swing_ts2vec.py search `
  --representation swing-grid --cohort-policy per-window `
  --grid-normalization candle-close-mean `
  --run-id 88a154e4d58a4cd7bb0c6c8a10cfa588 `
  --train-start 2018-01-01 --train-end 2022-12-31 `
  --validation-start 2023-01-01 --validation-end 2023-12-31 `
  --target pnl-r --durations 6h 12h 24h --repeats 1
```

La base des swings ne contient pas toutes les bougies. Le premier passage lit
les fichiers de ticks originaux référencés par la collecte pour les seules dates
nécessaires, afin d'en extraire les clôtures. Les petits agrégats journaliers
sont conservés dans `swing_candle_close_cache`, à côté de la base de recherche,
et réutilisés entre durées et campagnes. Les fichiers sources doivent correspondre
à l'instantané de la collecte. Aucune nouvelle collecte de swings n'est nécessaire.

Cette option s'applique à `train`, `search` et `run` en représentation `swing-grid`.
Elle impose un **nouvel entraînement**. Le choix est inscrit dans le modèle,
les rapports, le cache et l'identité de reprise. Le backtest et le test figé
reprennent automatiquement la normalisation enregistrée ; on ne peut pas changer
celle d'un ancien modèle. Les modèles et reprises existants continuent d'utiliser
`--grid-normalization swing-mean`, qui reste la valeur par défaut.

Ce changement est une hypothèse à mesurer sur validation, puis sur un test
indépendant. Les durées continuent à utiliser des moyennes différentes et,
en mode `per-window`, des cohortes propres à chaque fenêtre.

## Entraîner ou rechercher une durée

Les dépendances dédiées n'exigent ni Torch ni TS2Vec :

```powershell
py -m pip install -r requirements-swing-grid.txt
```

Exemple sur la collecte actuelle :

```powershell
py run_swing_ts2vec.py search `
  --representation swing-grid `
  --run-id 88a154e4d58a4cd7bb0c6c8a10cfa588 `
  --train-start 2022-01-01 --train-end 2022-12-31 `
  --validation-start 2023-01-01 --validation-end 2023-12-31 `
  --target pnl-r `
  --window-mode duration --durations 72h 96h `
  --repeats 1 `
  --output-dir C:\TradingBotData\swing_grid\experiments
```

`train` utilise les mêmes paramètres pour un entraînement simple. Les durées
peuvent être nouvelles : aucun contexte en blob ne doit être créé au préalable.
`search --dry-run` affiche les dimensions sans entraîner ni modifier la base.
`--duration-range` fonctionne aussi pour les grilles.

### Comparer les profondeurs des arbres

`search --max-depths 3 5 7 9` compare chaque combinaison **fenêtre × profondeur**.
Chaque combinaison utilise les mêmes seeds, de `--seed` à
`--seed + --repeats - 1`. Les répétitions restent regroupées par profondeur :
le classement compare leur score cible médian, puis leur pire score et leur couverture
médiane. La répétition exportée est la plus proche de la médiane de la
combinaison retenue. Le seuil et le nombre d'arbres restent choisis sur la
validation propre à chaque modèle.

```powershell
py run_swing_ts2vec.py search `
  --run-id 8b762a3e110d414ba26d3d3a6eec6332 `
  --train-start 2016-01-01 --train-end 2022-12-31 `
  --validation-start 2023-01-01 --validation-end 2023-12-31 `
  --representation swing-grid --grid-normalization candle-close-mean `
  --durations 60h 72h --max-depths 3 5 7 9 --repeats 1 `
  --output-dir C:\TradingBotData\swing_grid\search_depths
```

Cela entraîne **8 modèles principaux**, plus leurs témoins tabulaires.
Ajouter `--dry-run` affiche le plan sans entraîner. Les profondeurs réutilisent
les mêmes grilles en cache ; elles ne reconstruisent pas les fenêtres.
Le résumé final et `search.json` indiquent `selected_max_depth`,
`selected.json` indique `max_depth`, et les rapports et le CSV distinguent les
combinaisons fenêtre/profondeur. Le backtest charge automatiquement les arbres
du modèle retenu : aucun paramètre de profondeur n'est nécessaire au backtest.

Sans `--max-depths`, le comportement reste celui d'une profondeur fixe,
`--max-depth 3` par défaut. `--max-depths 5` équivaut à `--max-depth 5`.
Les deux options ne doivent pas être fournies ensemble. Une reprise conserve
exactement la grille de profondeurs et le nombre de répétitions d'origine ;
les essais terminés sont réutilisés. Pour explorer de nouvelles profondeurs,
lancer une nouvelle campagne. Les anciennes campagnes à profondeur fixe
gardent leur identité de reprise.

Toutes les cibles XGBoost existantes restent disponibles, notamment
`--target trailing-step-raw`, `trailing-step-square` et `trailing-step-exp`.
La grille d'entrée ne change pas quand on change la cible.

### Objectif de validation selon la cible

Pour `pnl-r`, le seuil et le classement continuent à maximiser la somme des R.
Pour les autres cibles, ils maximisent **le label moyen des trades acceptés** :

| Cible | Valeur dont on maximise la moyenne |
| --- | --- |
| `win` | `int(pnl_r > 0)` : taux de trades gagnants |
| `trailing-step-1` | `int(trailing_steps >= 1)` : taux d'atteinte de la step 1 |
| `trailing-step-raw` | `trailing_steps` |
| `trailing-step-square` | `trailing_steps ** 2` |
| `trailing-step-exp` | `exp(trailing_steps)` |

Le label est transformé **avant** sa moyenne. Le choix du seuil examine tous
les ensembles réalisables, respecte la première distance acceptée par swing
et n'impose **aucun minimum de trades** pour ces cibles. À qualité égale
au-dessus du label neutre, il privilégie la couverture ; à qualité neutre,
l'abstention est préférable. Le label neutre vaut 0, sauf pour l'exponentielle
où `exp(0)=1`. Les seuils binaires restent dans [0,1] : si une prédiction vaut
exactement 1, un seuil supérieur à 1 ne sera pas exporté pour forcer l'abstention.

Les répétitions sont classées sur la médiane de ce score, puis le pire score,
puis la couverture médiane. Le représentant est choisi près de la médiane
**du score cible**, pas du PnL. `total_r` et `mean_r` restent les vrais résultats
financiers descriptifs ; `total_label`, `mean_label`, `selection_score` et
`selected_validation_score` présentent les unités de la cible. Un résultat
gagnant sur le label peut donc rester perdant en R.

Le nombre d'arbres reste choisi par RMSE/logloss sur le label correspondant.
La qualification conserve le diagnostic de gain prédictif et exige un score
cible supérieur au neutre ; elle ne filtre plus les cibles trailing sur le PnL.
Les anciennes recherches `win` ou trailing calibrées sur le PnL ne peuvent pas être
reprises avec cette nouvelle règle : créer une nouvelle campagne. Les anciens
modèles restent utilisables avec leurs poids et seuils enregistrés. Les reprises
des campagnes `pnl-r` conservent leur identité.

Les options `--ts2vec-*`, `--min-encoder-samples`, `--swing-volume` et les
fenêtres en couples sont refusées pour éviter des réglages sans effet.
Sans `--representation`, le fonctionnement TS2Vec existant est conservé.

Un XGBoost témoin sur le sens, le nombre de swings et la durée est également
entraîné et rapporté séparément. Ces trois variables ne sont pas ajoutées au
vecteur du modèle principal.

## Cache, reprise et sélection

Les événements scalaires sont lus avec une connexion SQLite en lecture seule.
Seule la période demandée, précédée de la durée nécessaire à son initialisation,
est rejouée. Les blobs de `swing_contexts` et les embeddings existants ne sont
ni chargés ni recréés.

Le cache `_swing_grid_cache` contient une matrice par durée, avec des fichiers
binaires denses lus par memory mapping, et la cohorte des événements labellisés.
Les répétitions réutilisent les mêmes matrices ; les empreintes sont vérifiées.
Les caches et modèles sont dans le dossier de sortie, pas dans les tables
d'embeddings de la base. Un search écrit seulement ses petits enregistrements
de suivi dans la base de recherche.

Par défaut, toutes les durées d'un search partagent la même cohorte éligible. Les trades
doivent être clôturés et leur résultat connu avant la fin de leur période.
Les contextes peuvent contenir des données antérieures au début du train :
elles étaient déjà connues au moment du signal.

### Garder un modèle stable quand la grille de durées change

Le protocole historique `--cohort-policy common` impose l'intersection des
événements éligibles à **toutes** les durées, au train comme en validation.
Ajouter une durée change donc aussi les données des modèles des autres durées.
Par exemple une fenêtre de 1h impose au moins deux swings dans la dernière
heure, même au modèle 12h. C'est un filtre supplémentaire sur sa population.

Pour que le modèle 12h conserve les mêmes données et le même entraînement
lorsqu'on ajoute ou enlève d'autres durées :

```powershell
py run_swing_ts2vec.py search --representation swing-grid `
  --cohort-policy per-window `
  --run-id 88a154e4d58a4cd7bb0c6c8a10cfa588 `
  --train-start 2018-01-01 --train-end 2022-12-31 `
  --validation-start 2023-01-01 --validation-end 2023-12-31 `
  --target pnl-r --durations 12h 14h 16h 18h 20h 22h --repeats 1
```

Dans ce mode, chaque durée utilise ses propres exemples pour le train,
l'early stopping et le choix de son seuil. Les modèles sont ensuite comparés
sur les mêmes événements de validation communs, **sans réentraîner ni
rechoisir leur seuil** sur cette intersection.

Le rapport distingue le **résultat de validation propre** et le **score de
classement sur validation commune**. Le premier, les arbres et le seuil
restent stables à données/paramètres identiques ; le score de classement peut
changer si les événements communs changent. Ces deux évaluations appartiennent
à la même période de validation, pas à deux tests indépendants.

Le mode historique reste le défaut pour préserver les campagnes existantes.
La politique choisie entre dans l'identité du search : une reprise doit la
conserver. `per-window` s'applique actuellement uniquement à `swing-grid`.

La sélection de l'époque de boosting, du seuil, de la durée et du représentant
des répétitions reste effectuée sur la validation. Le rapport et la dernière
ligne du programme mettent en évidence le seuil initial retenu. Le test reste
une étape séparée, sans réentraînement ni modification de seuil.

`search --resume CHEMIN` reprend avec exactement les mêmes options, dont
`--representation swing-grid`. Les répétitions terminées sont réutilisées.
Un arrêt pendant un entraînement XGBoost peut demander de refaire cette
répétition, mais le cache des grilles est conservé.

Avec le XGBoost actuel (`subsample=1`, `colsample_bytree=1`, méthode `hist`),
changer la graine peut produire des résultats identiques. **Une répétition
suffit pour un premier essai** ; plusieurs graines ne constituent pas des
validations temporelles indépendantes.

## Backtest et test figé

Le dossier de search ou le sous-dossier d'une durée peut être utilisé :

```powershell
$modele = 'C:\TradingBotData\swing_grid\experiments\CHEMIN_DU_SEARCH'
py run_backtest.py --strategy swing-grid --swing-grid-model $modele `
  --start-date 2024-01-01 --end-date 2024-12-31 `
  --balance 10000 --risk-percent 1 --max-positions 5
```

Le backtest utilise le même calcul de grille que l'entraînement. Il reprend
les bougies, les sorties et le seuil du modèle. Les options
`--swing-grid-threshold`, `--swing-grid-min-threshold`,
`--swing-grid-stop-buffer-pips`, `--swing-grid-take-profit-r`,
`--swing-grid-decision-log` et `--swing-grid-threads` sont disponibles, ainsi
que le réglage de trailing existant `--trailing-start-rr-ratio`.

Pour le test des opportunités avec sorties et seuil figés :

```powershell
py run_swing_ts2vec.py test --resolver $modele `
  --run-id 88a154e4d58a4cd7bb0c6c8a10cfa588 `
  --test-start 2024-01-01 --test-end 2025-12-31 `
  --output-dir C:\TradingBotData\swing_grid\tests
```

Les sommes de R portent sur des opportunités chevauchantes. Elles ne donnent
pas le rendement d'un portefeuille. Une bonne validation ne démontre pas un
edge futur, et modifier les sorties ou adapter le seuil change les conditions
de l'évaluation.

## Premier essai effectué

Sur la collecte `88a154e4d58a4cd7bb0c6c8a10cfa588`, un search a été effectué
avec la cible `pnl-r`, le train en 2022, la validation en 2023, les cinq paires
et trois répétitions. La cohorte commune contient **37 112 exemples de train**
et **36 693 exemples de validation**.

| Durée | Entrées | R de validation retenu | Arbres retenus |
|---|---:|---:|---:|
| 72h | 216 | −404,470R | 1 |
| 96h | 288 | −467,505R | 1 |

Les trois graines donnent les mêmes résultats. La fenêtre sélectionnée est
72h, avec un seuil exploratoire de **−0,06472490727901459R**. Elle retient
2 625 opportunités de validation, avec une moyenne de −0,15408R et un PF de
0,76684. Le gain de MSE face à la constante est de seulement 0,00002836.
**Cette validation n'est pas qualifiée** et ce seuil négatif ne constitue pas
une indication de rentabilité. Aucun test 2024–2025 ou 2026 n'a été consulté
pour cet essai.

Résultats conservés dans
`C:\TradingBotData\swing_grid\experiments\search_88a154e4d58a_9a967110d80b`.
Le résultat porte sur cette cible et cette configuration d'arbres ; les autres
cibles sont disponibles mais n'ont pas été comparées dans cet essai.

Vérification depuis le dossier parent de `ultimateBot` :

```powershell
py -m unittest ultimateBot.tests.test_swing_grid -q
```
