# Réseau neuronal pour prédire le pnl-r des swings

Le modèle `swing-nn` reçoit **la même grille dense que XGBoost swing-grid** :
une valeur par largeur de bougie, prix du swing normalisé, zéro dans les cases
sans swing. Il ne reçoit ni embedding TS2Vec, ni volume, ni symbole encodé,
ni label dans ses entrées. La cible est exclusivement le **pnl-r réalisé**.
Les nouvelles collectes à stops variables ajoutent une marge de stop à ce vecteur :
voir [variantes de stop structurel](swing_stop_variants.md).
La sortie est une valeur en R, sans sigmoid : ce n'est pas une probabilité.

La normalisation par défaut est la nouvelle `candle-close-mean` : moyenne des
clôtures `(bid + ask) / 2` de toutes les bougies closes de la fenêtre, séparément
pour chaque symbole. `--grid-normalization swing-mean` permet une comparaison
avec l'ancienne représentation. La largeur reste celle de la collecte.

## Choix de configuration

`--architecture auto` choisit un MLP pour au plus 128 entrées et un réseau
convolutif 1D au-delà. Ce sont des points de départ explicites, pas une garantie
de configuration optimale ou de rentabilité. `search` compare réellement les
configurations demandées sur validation.

Sur les bougies actuelles de 20 minutes, le profil `balanced` donne :

| Fenêtre | Entrées | Architecture | Paramètres |
| --- | ---: | --- | ---: |
| 6 h | 18 | MLP 64 → 32 → 1 | 3 329 |
| 12 h | 36 | MLP 64 → 32 → 1 | 4 481 |
| 24 h | 72 | MLP 128 → 64 → 1 | 17 665 |
| 72 h | 216 | Conv1D 16 → 32 → 32, tête 32 → 1 | 16 097 |

Le MLP est adapté à la petite taille de ces vecteurs et conserve toutes les
positions. Le Conv1D partage ses filtres dans le temps pour les vecteurs longs :
noyaux de largeur 5, dilatations 1/2/4, puis pooling en huit segments temporels
et une tête dense. L'ordre grossier des segments est conservé.

`--profile compact` réduit la largeur des couches par rapport à `balanced` ;
`wide` l'augmente. On peut imposer `--architecture mlp` ou `conv1d`, y compris
dans la recherche. Les couches utilisent GELU et un dropout de 15 % par défaut.

Autres réglages par défaut : AdamW, learning rate 0,001, weight decay 0,001,
batch 512, maximum 40 époques, patience 8, réduction du learning rate sur plateau,
et norme du gradient plafonnée à 1. Ces paramètres sont modifiables en CLI.
La perte MSE porte sur la cible pnl-r standardisée avec les statistiques du train ;
l'époque est sélectionnée sur la MSE **en R** de validation.

Une unique échelle RMS calculée sur les entrées du train évite des valeurs trop
petites pour l'optimisation. Elle ne soustrait pas de moyenne et conserve donc
exactement les zéros des cases vides. Elle est figée avec le modèle. Le zéro
peut toujours représenter aussi un swing exactement égal à la référence,
conformément au codage d'origine.

```powershell
py -m pip install -r requirements-swing-nn.txt

# Affiche le réseau et son nombre de paramètres sans charger les fenêtres.
py run_swing_nn.py plan `
  --run-id 88a154e4d58a4cd7bb0c6c8a10cfa588 --duration 12h
```

## Entraîner une durée

```powershell
py run_swing_nn.py train `
  --run-id 88a154e4d58a4cd7bb0c6c8a10cfa588 `
  --train-start 2018-01-01 --train-end 2022-12-31 `
  --validation-start 2023-01-01 --validation-end 2023-12-31 `
  --duration 12h --architecture auto --profile balanced `
  --grid-normalization candle-close-mean `
  --output-dir C:\TradingBotData\swing_nn\modele_12h
```

Les bornes sont des dates UTC inclusives. Chaque trade labellisé doit avoir son
résultat connu avant la fin de sa période. Les données précédant le début du
train sont utilisées uniquement pour initialiser le contexte causal. Aucune
période de test n'est utilisée ni évaluée automatiquement.

`--symbols EURUSD USDJPY` sélectionne un groupe explicite. Sans cette option,
tous les symboles de la collecte sont utilisés, avec une moyenne de prix propre
à chacun. `--device cuda` permet l'entraînement GPU lorsqu'il est disponible ;
`--threads 2` règle les threads CPU. Le backtest charge le réseau sur CPU.

Les grilles sont préparées par le même code que XGBoost, à partir des événements
scalaires, sans lire les blobs de contextes ni modifier la base. Le cache partagé
par défaut est `swing_grid_cache` à côté de la base. Pour réutiliser exactement un
cache XGBoost existant, ajouter `--grid-cache-root CHEMIN\trials\_swing_grid_cache` ;
la durée, les dates, les symboles, la cible et la normalisation doivent correspondre.
Le petit cache journalier des clôtures est réutilisé entre toutes les campagnes.
Les nouvelles collectes différeront les contextes par défaut : aucune fenêtre
générique n'est calculée ou stockée pendant `collect`. Les durées du réseau sont
choisies ici, dans `train` ou `search`. `--store-contexts` reste disponible lors
de la collecte pour les usages historiques qui le demandent.

## Rechercher les configurations

```powershell
py run_swing_nn.py search `
  --run-id 88a154e4d58a4cd7bb0c6c8a10cfa588 `
  --train-start 2018-01-01 --train-end 2022-12-31 `
  --validation-start 2023-01-01 --validation-end 2023-12-31 `
  --durations 6h 12h 24h `
  --architectures auto --profiles compact balanced --seeds 1 2 3 `
  --grid-normalization candle-close-mean `
  --output-dir C:\TradingBotData\swing_nn\search_6h_12h_24h
```

Cet exemple entraîne 18 modèles : trois durées, deux profils, trois graines.
`--architectures mlp conv1d` permet une comparaison directe des deux familles.
Les configurations équivalentes auto/explicites ne sont pas entraînées deux fois.

Avant de charger les données, `search` affiche `[swing-nn-search-plan]` avec le
nombre total de modèles, leur répartition MLP/Conv1D et le matériel demandé.
`plan.json` conserve la liste complète des candidats ; les lignes
`[swing-nn-search-window]` indiquent les dimensions et architectures par fenêtre.
Chaque modèle affiche son numéro, le total et le mode `new`, `resume` ou `reuse`.
Les temps de préparation, de modèle et de validation sont affichés séparément.
Lors d'une reprise au milieu d'une époque, `session_train_s` mesure seulement
la partie exécutée depuis cette reprise, sauvegardes comprises.

**Coût CPU :** `auto` passe en Conv1D au-delà de 128 entrées, soit dès 48 h sur
des bougies de 20 minutes. Un nombre de paramètres réduit ne garantit pas un
calcul rapide : les convolutions traitent plusieurs canaux à chaque position.
Pour une première comparaison rapide sur CPU, imposer `--architectures mlp`,
y compris pour les fenêtres de 48 à 96 h, et choisir explicitement les profils.
Par exemple, quatre durées × deux profils × trois graines entraînent 24 modèles,
contre 12 avec `--profiles balanced` seul. La qualité respective des architectures
reste à comparer ; leur coût de calcul ne permet pas de la déduire.
L'architecture d'un entraînement interrompu reste inchangée lors de sa reprise.

Chaque durée garde sa propre cohorte d'apprentissage et de validation, comme
le mode `per-window` de XGBoost. Ajouter une autre durée ne change pas ses
entrées, son entraînement, son early stopping ou son seuil. Les comparaisons
utilisent ensuite les mêmes événements de validation communs à toutes les
durées, **sans réentraîner ni recalibrer les seuils** sur cette intersection.

Le classement utilise la MSE médiane des graines sur cette validation commune,
puis la pire MSE et enfin le R médian en cas d'égalité. Le modèle représentatif
est le plus proche de la MSE médiane de son groupe, pas la meilleure graine.
Le score de classement peut changer si la population commune change.
Le rapport distingue la validation propre au modèle de cette comparaison.
Il affiche aussi les R minimum et maximum, le nombre de graines actives et
les résultats de chaque graine : une médiane nulle ne signifie pas que tous
les modèles ont donné zéro trade.

## Seuil, qualification et abstention

Le seuil est choisi sur le PnL de validation, avec un plancher de **0 R** par
défaut (`--min-threshold`) et **aucun minimum d'événements acceptés**. Un groupe
d'un seul événement est admissible. L'effectif est rapporté comme information,
sans exclure un seuil ni rendre le diagnostic défavorable à lui seul.
Le cas zéro trade reste admissible si aucun groupe au-dessus du plancher
ne produit un gain strictement positif.

L'ancien paramètre `--min-accepted-validation` reste lisible pour reprendre les
anciens runs et charger leurs checkpoints. Sa valeur, même 100 dans une ancienne
configuration, est **ignorée** par le calcul actuel. Les métadonnées indiquent
`acceptance_count_filter=false` et un minimum effectif nul.
Tous les seuils distincts réalisables sont évalués exactement, en regroupant
les scores égaux. Le calcul utilise un tri et des sommes cumulées ; il ne
dépend plus d'une grille de quantiles susceptible de manquer une petite zone
profitable. Les rapports conservent un échantillon borné des candidats, avec
l'optimum exact et les points de diagnostic.

Pour les collectes à stops variables, le seuil et les métriques de trading
retiennent uniquement la première distance acceptée par swing. Le calcul
remplace le trade retenu si une distance plus courte devient admissible.
La comparaison entre fenêtres et seeds applique la même règle. Les familles
de variantes incomplètes sont signalées et écartées de ces métriques ; la MSE
reste calculée sur les labels disponibles. Voir [la politique complète](swing_stop_variants.md#validation-et-choix-du-seuil).

La décision indique explicitement l'une de ces raisons :

- `no_prediction_at_or_above_floor` : aucun score n'atteint le plancher ;
- `no_profitable_admissible_threshold` : aucun seuil au-dessus du plancher ne gagne ;
- `positive_validation_pnl` : un seuil admissible positif a été retenu.
- `no_complete_swing_outcomes` : aucune famille complète de variantes n'est évaluable.

Le rapport présente aussi le résultat au plancher fixé, avant optimisation du
seuil. Le résultat du meilleur seuil est nécessairement non négatif puisque
l'abstention figure parmi les choix. Un résultat positif après ce choix ne
démontre donc pas, à lui seul, un edge hors de la validation.

La qualification exige une meilleure MSE que la constante apprise sur le train,
et un R sélectionné positif. C'est un diagnostic descriptif, pas un test de
significativité statistique. Le prédicteur constant initial
est lui-même candidat à l'époque zéro : si aucune époque ne le bat, le rapport
l'indique et le modèle n'est pas qualifié. La qualification ne bloque pas
techniquement un backtest d'exploration.

`best_epoch=0` signifie donc que les poids retenus prédisent **la moyenne du
train pour chaque entrée**, même si plusieurs époques ont effectivement été
entraînées. Si cette moyenne est négative et le plancher vaut zéro, aucun
trade n'est accepté. Cela ne signifie pas que la validation est vide ou que
les réseaux entraînés produisaient tous une constante. Le nombre de données
d'apprentissage ne garantit pas une meilleure MSE sur une autre période.

Le résumé `[swing-nn-selected]` distingue `predictor=training_mean_constant`
et `predictor=trained_network`, indique `validation_rows`, les MSE du modèle
retenu et de la constante, ainsi que l'époque et la MSE du meilleur réseau
réellement entraîné. Les lignes par époque montrent aussi l'étendue des
prédictions du réseau en cours. Ces diagnostics ne modifient ni le choix des
poids ni le seuil. Les modèles déjà enregistrés restent inchangés.

Les fichiers `metadata.json` et `rapport.md`, ainsi que le résumé final,
mettent en évidence le **seuil initial en R** et la qualification. La validation
sert à choisir époque, architecture, durée et seuil ; elle ne remplace pas
un test indépendant. Les R concernent des opportunités chevauchantes aux sorties
de la collecte, pas la performance d'un portefeuille avec contraintes de risque.

## Sauvegardes et reprise

### Corriger les seuils d'une campagne déjà terminée

Les anciens modèles restent figés. Pour appliquer le calcul exact des seuils
à leurs prédictions de validation sauvegardées, sans relire les fenêtres ni
réentraîner les réseaux :

```powershell
py run_swing_nn.py recalibrate `
  --model C:\TradingBotData\swing_nn\search_6h_12h_24h `
  --output-dir C:\TradingBotData\swing_nn\search_6h_12h_24h_no_minimum
```

Un modèle individuel peut aussi être fourni à `--model`. Le dossier de sortie
doit être nouveau. Les poids sont copiés à l'identique et leurs empreintes
sont vérifiées. Le plancher d'origine est conservé et tout minimum d'événements
acceptés est supprimé, y compris si l'ancienne configuration demandait 100 trades.
La nouvelle calibration, son origine et les anciens résultats sont enregistrés
dans les métadonnées. Pour une campagne, le classement est recalculé sur les
mêmes événements communs avec les nouveaux seuils propres à chaque modèle.
Le dossier dérivé est directement utilisable en backtest.

Ce recalcul réutilise la même validation : son résultat n'est pas un nouveau
test indépendant. Une reprise ordinaire ne modifie pas les seuils des modèles
déjà publiés.

Un processus Python déjà en cours continue avec le code chargé à son démarrage.
Après sa fin, `recalibrate` applique cette politique à ses prédictions sauvegardées,
sans relancer l'entraînement.

### Reprendre un entraînement interrompu

`latest.pt` conserve les poids courants et les meilleurs poids, l'optimiseur,
le scheduler, l'ordre des exemples, le curseur dans l'époque et les générateurs
aléatoires. Il est écrit atomiquement toutes les 100 mises à jour par défaut
(`--checkpoint-batches`) et à chaque fin d'époque. Une interruption reprend
depuis le dernier checkpoint complet, pas depuis un batch partiellement mis à jour.

```powershell
py run_swing_nn.py train --resume C:\TradingBotData\swing_nn\modele_12h
py run_swing_nn.py search --resume C:\TradingBotData\swing_nn\search_6h_12h_24h
```

La reprise restaure les données et les réglages enregistrés. Elle refuse de
changer durée, normalisation, architecture, batch size, taux d'apprentissage
ou autres paramètres statistiques. Un `train` inachevé peut étendre `--epochs` ;
les réglages de matériel et fréquence de sauvegarde peuvent changer. Un modèle
déjà publié est figé : un nouvel entraînement utilise un autre dossier.
Les datasets de recherche et les essais terminés sont réutilisés après interruption.
Une reprise sur CPU identique est testée à poids égaux ; PyTorch ne garantit
pas des résultats identiques entre matériels, versions ou plateformes différents.

## Backtest et test figé

Le dossier d'une recherche ou celui d'un modèle individuel est utilisable :

```powershell
$modele = 'C:\TradingBotData\swing_nn\search_6h_12h_24h'
py run_backtest.py --strategy swing-nn --swing-nn-model $modele `
  --start-date 2024-01-01 --end-date 2024-12-31 `
  --balance 10000 --risk-percent 1 --max-positions 10
```

Le robot reprend automatiquement bougies, grille, normalisation, seuil et plancher figés.
Il utilise les mêmes règles de causalité, d'expiration et de risque que les
autres stratégies de swings. Les options `--swing-nn-threshold`,
`--swing-nn-min-threshold`, `--swing-nn-target-trades-per-day`,
`--swing-nn-threshold-step`, `--swing-nn-stop-buffer-pips`,
`--swing-nn-take-profit-r`, `--swing-nn-decision-log` et `--swing-nn-threads`
sont disponibles. `--trailing-start-rr-ratio` conserve son rôle commun.
Modifier les sorties en backtest change le résultat de la stratégie par rapport
aux labels qui ont servi à entraîner le prédicteur.

Pour évaluer les opportunités labellisées d'un test avec modèle et seuil figés :

```powershell
py run_swing_nn.py test --model $modele `
  --start-date 2024-01-01 --end-date 2024-12-31 `
  --output-dir C:\TradingBotData\swing_nn\test_2024
```

Cette commande n'effectue aucun apprentissage ni choix de seuil. Elle vérifie
que le test suit la validation et conserve les paramètres de collecte d'origine.

Références techniques :
[MSE PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html),
[sauvegardes des modèles et optimiseurs](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html),
[limites de reproductibilité](https://docs.pytorch.org/docs/stable/notes/randomness.html).
