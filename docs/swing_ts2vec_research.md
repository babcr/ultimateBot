# Recherche brute force : swings → TS2Vec → XGBoost

Cet outil collecte les opportunités de retournement sans les filtres d'une stratégie de portefeuille. Les données sont conservées dans une base SQLite. Il est indépendant des moteurs `structure`, `swing-regression` et `swing-patterns`.

Le protocole par défaut est celui demandé :

| Étape | Période | Opération |
|---|---|---|
| Collecte initiale | 2020–2025 | Construire les swings, contextes et résultats de tous les trades indépendants |
| Apprentissage | 2020–2023 | Entraîner TS2Vec, son scaler et XGBoost |
| Validation | 2024–2025 | Choisir la fenêtre, le nombre d'arbres et le seuil d'acceptation |
| Test séparé | 2026 | Transformer les nouveaux contextes et prédire avec les modèles et seuils figés |

**Il n'y a pas de période de préentraînement supplémentaire découpée dans 2020–2023.** TS2Vec apprend sur les contextes de cette période entière, sans utiliser de PnL. XGBoost apprend sur ses trades dont le résultat est connu avant le 1er janvier 2024. Un trade de 2023 fermé en 2024 n'est pas un label disponible pour cet apprentissage. De même, un trade de validation fermé en 2026 n'entre pas dans le choix du seuil.

Les embeddings historiques de l'apprentissage sont des reconstructions avec l'encodeur appris sur 2020–2023 : c'est une transformation du jeu d'apprentissage. Les contextes eux-mêmes n'utilisent que les swings connus à l'instant du signal. La validation et le test utilisent ensuite l'encodeur et le scaler figés. Les contextes de début 2024 ou 2026 peuvent naturellement contenir des swings de la fin de la période précédente.

La commande `train` n'évalue pas le test, même si des lignes de 2026 existent déjà dans la base. La commande `test` ne rappelle ni l'apprentissage TS2Vec, ni `fit` XGBoost, ni la sélection d'un seuil.

## Règles de collecte

Les nouvelles options `--stop-variants`, `--stop-step-pips` et `--max-holding`
sont décrites dans [les variantes de stop structurel](swing_stop_variants.md).
La limite de durée est de deux semaines par défaut pour les nouvelles commandes
de collecte. Les anciens runs conservent leur configuration enregistrée.

Les **nouvelles collectes différeront le windowing par défaut** : elles
enregistrent les swings scalaires et les résultats des trades, sans calculer
ni stocker les blobs de contexte génériques. On peut omettre `--window-mode`,
`--durations` et `--pairs` lors de la collecte, puis choisir les fenêtres dans
`train` ou `search`. Les fenêtres figurant dans les paramètres de collecte
restent des présélections de compatibilité ; elles n'indiquent pas une couverture
déjà calculée lorsque `store_contexts=false`.

`--store-contexts` réactive la construction historique pendant la collecte ;
`--no-store-contexts` rend le choix différé explicite. Le mode choisi est
enregistré dans le run et hérité lors d'un ajout de symboles. Un ancien run sans
ce paramètre garde son comportement historique lors d'une extension.

Les grilles NN/XGBoost sont préparées et mises en cache à leur étape d'apprentissage,
sans créer de contexte générique en base. TS2Vec construit ses contextes à la
demande, uniquement sur les symboles et dates nécessaires, puis les réutilise.

- Un swing high est une bougie haussière bid suivie d'une bougie baissière bid. Son prix est le plus haut bid des deux bougies. Il produit une vente au bid.
- Un swing low est une bougie baissière ask suivie d'une bougie haussière ask. Son prix est le plus bas ask des deux bougies. Il produit un achat à l'ask.
- Le signal devient disponible à la clôture de la seconde bougie. L'ordre au marché s'exécute sur le premier tick strictement ultérieur à son instant d'éligibilité, conformément au modèle d'exécution existant.
- L'ordre non exécuté expire après deux bougies par défaut. **Une position exécutée n'expire pas après deux bougies** : elle reste suivie par son stop et son trailing.
- Le stop initial est le prix du swing, élargi par `--stop-buffer-pips`. Achat : `low ask − marge`. Vente : `high bid + marge`. Le défaut est zéro. Le risque initial et tous les paliers du trailing sont calculés avec ce stop élargi.
- Le trailing s'active à 1 R par défaut, passe au prix d'entrée, puis avance par paliers de 1 R. Il réutilise le `TrailingStopManager` et le moniteur de stops du bot, sans TP fixe.

**Aucun `RiskManager` n'intervient dans la collecte.** Il n'y a ni plafond de positions, ni contrôle de capital ou de marge, ni filtre de spread, de distance de stop, de volatilité ou de RR. Chaque opportunité est simulée indépendamment et peut se chevaucher avec toutes les autres. Un stop déjà franchi au fill ne provoque pas un rejet : le trade est exécuté et sa perte immédiate est enregistrée au prix de sortie exécutable.

La valeur `--risk-cash 100` sert seulement à fixer une unité virtuelle comparable pour les résultats : quantité = 100 / distance initiale du stop. Aucun solde n'est débité pour autoriser un autre trade. Le PnL et le PnL en R sont enregistrés. Les sommes de R de ces opportunités ne sont pas un rendement de portefeuille. Si la distance est exactement nulle, le trade et son PnL restent enregistrés, mais son R reste NULL car cette division n'est pas définie.

Les positions encore ouvertes à la fin des données restent sans PnL réalisé. Les ordres expirés n'ont pas de label de trade. Les ticks manquants ne sont pas remplacés par un parcours OHLC synthétique. Les paires de bougies séparées par une interruption ne deviennent pas artificiellement un retournement de deux bougies.

## La série de swings

À chaque nouvelle confirmation, on fusionne uniquement les highs bid et les lows ask. La séquence alterne strictement high/low. **Deux swings consécutifs de même nature sont remplacés par le plus récent, même s'il est moins extrême.** Une capture enregistrée lors d'un signal précédent n'est jamais réécrite rétrospectivement.

À l'étape de windowing, les deux modes de contexte sont :

- `duration` : les swings confirmés dans les dernières 12, 24 ou 48 heures par défaut ; la fenêtre est mesurée en temps écoulé et peut donc contenir un nombre variable de points ;
- `pairs` : exactement 4, 8 ou 16 couples, donc respectivement 8, 16 ou 32 swings alternés.

Une fenêtre insuffisamment alimentée ne fournit pas encore d'embedding, mais son trade est tout de même collecté. Si le bid confirme un high et l'ask un low à la même clôture, les deux opportunités sont enregistrées et simulées. Le contexte ambigu est réinitialisé : le moteur n'invente pas leur ordre dans une séquence alternée.

Chaque point contient six valeurs : prix logarithmique relatif au premier swing de la fenêtre, variation logarithmique depuis le swing précédent, nature high/low, délai entre les extrêmes, délai entre les confirmations et délai extrême→confirmation. Les temps sont exprimés en périodes de bougie puis transformés par `log1p`. Les dates d'extrême correspondent aux bougies où ils ont été atteints ; leur disponibilité correspond à la clôture de la seconde bougie.

TS2Vec est le code du sous-module existant, avec six canaux d'entrée sans volume, ou sept avec l'option de volume ci-dessous. Le scaler est appris uniquement sur l'apprentissage. Les fenêtres de durée peuvent avoir des longueurs différentes ; à l'inférence, seules des séquences de même longueur sont mises dans un batch, sans padding dans l'agrégation max. Les dimensions de sortie sont 64 par défaut. Tous les contextes d'apprentissage sont admis par défaut (`--ts2vec-max-train-samples 0`). Pour limiter volontairement le volume de calcul, cette option accepte un plafond.

### Moyenne du volume par tick sur les deux bougies du swing

Ajouter `--swing-volume` à une **nouvelle collecte** pour enregistrer une
septième variable, `mean_quote_volume` :

```powershell
py run_swing_ts2vec.py collect --symbols EURUSD GBPUSD USDCAD USDCHF --start-date 2020-01-01 --end-date 2026-09-30 --candle-interval 60m --window-mode duration --durations 24h --stop-buffer-pips 2 --swing-volume
```

Pour un high, elle vaut `somme des bidVolume des deux bougies / nombre total de
ticks des deux bougies`. Pour un low, la même formule utilise les `askVolume`.
On utilise **tous les ticks des deux bougies**, et pas seulement ceux de la
bougie portant l'extrême. Deux bougies contenant respectivement 10 et 100 ticks
ont donc des poids différents ; on ne fait pas la moyenne de leurs deux
moyennes. Les volumes nuls comptent comme observations valides. Un volume
manquant, négatif ou non fini provoque une erreur en mode volume ; il n'est
pas remplacé par zéro. La donnée est le volume de cotation bid/ask fourni par
les fichiers de ticks.

Le volume devient disponible à la clôture de la seconde bougie, comme le swing.
Aucun tick ultérieur ni résultat du trade n'entre dans cette variable. Sa
valeur brute est conservée dans `swing_events.mean_quote_volume` et dans les
fenêtres ; elle est standardisée par le scaler TS2Vec appris uniquement sur le
train, comme les six autres variables. La dimension d'entrée passe de 6 à 7,
mais la taille de l'embedding reste celle de `--ts2vec-output-dims` (64 par
défaut). XGBoost continue de recevoir cet embedding et ses trois variables
tabulaires habituelles.

`train` et `search` reprennent automatiquement le format de la collecte. Sur
une collecte contenant les volumes, on peut entraîner deux recherches avec
les mêmes événements, labels, fenêtres, dates et graines :

```powershell
py run_swing_ts2vec.py search --run-id $swingVolumeRun --train-start 2020-01-01 --train-end 2022-12-31 --validation-start 2023-01-01 --validation-end 2023-12-31 --durations 72h 96h --repeats 3 --swing-volume
py run_swing_ts2vec.py search --run-id $swingVolumeRun --train-start 2020-01-01 --train-end 2022-12-31 --validation-start 2023-01-01 --validation-end 2023-12-31 --durations 72h 96h --repeats 3 --no-swing-volume
```

La seconde ignore le septième canal pour mesurer l'apport du volume, sans
recollecter ni resimuler les trades. Les fenêtres stockées sont réutilisées.
Chaque recherche conserve ses propres modèles et seuils. L'apport éventuel
reste à mesurer sur une période de test séparée avec les mêmes règles de
portefeuille.

Les anciens modèles et collectes à six variables restent compatibles. Ajouter
un symbole à une collecte existante conserve obligatoirement son format de
volume. Les anciennes collectes n'ont pas enregistré cette moyenne : elles ne
peuvent pas être transformées en sept variables sans relire les ticks ; cette
version demande une nouvelle collecte avec `--swing-volume`.
Le backtest déduit le format du modèle et calcule la même moyenne à partir des
ticks reçus. Un modèle à sept variables exige aussi une collecte de test avec
volumes ; un modèle à six variables peut ignorer le canal d'une collecte à sept.

### Configuration liée aux données

Les nouveaux entraînements utilisent les règles suivantes :

| Réglage | Défaut et justification |
|---|---|
| Budget TS2Vec | `--ts2vec-train-epochs 1` : un parcours des lots d'apprentissage, calculé selon le volume réel ; ce minimum de référence n'est pas un nombre d'époques optimal démontré |
| Budget explicite | `--ts2vec-train-iters N` remplace les époques ; un budget inférieur à un parcours des données est signalé dans le plan |
| Longueur | `--ts2vec-max-train-length 0` : fenêtres complètes pendant le préentraînement et l'encodage ; aucune coupe silencieuse des 256 derniers swings |
| Limite explicite | Une limite positive constitue une borne d'entrée : une fenêtre plus longue provoque une erreur, jamais une troncature |
| Profondeur | `--ts2vec-depth 0` : profondeur minimale dont le champ réceptif couvre le plus long contexte d'apprentissage ; une profondeur explicite trop courte est signalée |
| Capacité et taux | 64 dimensions, 32 canaux cachés, taux TS2Vec 0,001 ; XGBoost conserve profondeur 3, taux 0,03 et plafond 400 arbres : ce sont des bases explicites à comparer, pas des optimums déduits automatiquement |

Le champ réceptif du réseau TS2Vec vaut `1 + 4 × (2^(depth+1) − 1)` swings,
car chacun de ses `depth+1` blocs comporte deux convolutions de largeur 3
et une dilation doublée à chaque bloc. Les longueurs de validation ne servent
pas à calculer la profondeur, le scaler ou le budget de préentraînement.
Les recadrages aléatoires contrastifs internes à TS2Vec restent des augmentations
d'apprentissage ; les fenêtres fournies au modèle et au scaler sont complètes.

Le chargeur amont utilise des lots complets (`drop_last=True`). Avec N séquences
et un batch B, une époque compte `floor(N / min(B,N))` mises à jour ; le reliquat
est exclu de cette époque après mélange aléatoire. Le plan affiche ce reliquat,
le nombre de présentations de séquences et leur équivalent en parcours des
données. Ainsi 206 775 séquences, batch 16 et une époque donnent 12 923 mises à
jour et 206 768 présentations, au lieu de l'ancien budget fixe de 200 mises à jour.

`search --dry-run` affiche `encoder_preflight` par fenêtre : effectifs et
mises à jour. Pour les durées, les longueurs, profondeur, champ réceptif et
mémoire sont des **majorants**, calculés avec au plus un swing non ambigu par
bougie ; les champs portent le suffixe `upper_bound`. Pour les couples, la
longueur est exacte. Cela permet un aperçu depuis les index sans relire
tous les blobs de features d'une base de plusieurs dizaines de Go.
Le plan exact est calculé depuis les longueurs réelles avant le fit.
Si des contextes restent à construire, leur coût exact est marqué
`pending_context_build` et sera affiché avant le préentraînement. La mémoire
annoncée ne couvre que le tenseur d'entrée ; les features en mémoire,
activations, copies temporaires et l'optimiseur nécessitent de la mémoire
supplémentaire. Aucune durée d'exécution n'est inventée à partir de ce volume.
Un `training_plan.json` conserve le plan effectif de chaque modèle ; le
checkpoint enregistre la configuration demandée, la configuration résolue,
les mises à jour exécutées, les époques terminées et la dernière perte, même
si une limite explicite d'itérations interrompt une époque.

XGBoost enregistre les courbes apprentissage/validation. Pour la classification,
la logloss de validation contrôle toujours l'arrêt anticipé ; l'AUC est aussi
mesurée. Le rapport compare la prédiction à une constante calculée sur les
labels d'apprentissage. `validation_qualified` exige un gain sur cette constante,
une AUC supérieure à 0,5 pour la classification et un score de validation
supérieur au neutre : somme R pour `pnl-r`, label moyen pour les autres cibles.
Ce diagnostic ne constitue pas une preuve statistique de rentabilité ; les
candidats restent consultables et ne sont pas automatiquement interdits au robot.

Les anciens checkpoints TS2Vec restent lisibles avec leur comportement
d'inférence d'origine. Les nouveaux budgets et la politique de fenêtres
complètes changent le protocole : créer une nouvelle recherche avec le même
run de collecte. `--resume` reste réservé à une configuration identique ;
on ne mélange pas les anciennes répétitions et les nouvelles.

## Stockage SQLite

Défaut : `C:\TradingBotData\swing_ts2vec\research.sqlite3`.

| Table | Contenu |
|---|---|
| `swing_runs` | Configuration de collecte, symboles, périodes, fichiers sources, provenance et statut |
| `swing_run_extensions` | Ajouts de symboles au même run, provenance propre à l'ajout, statut et erreur éventuelle |
| `swing_events` | Chaque retournement, prix du swing, stop élargi, dates d'ordre/fill/close, statut, PnL et R |
| `swing_contexts` | Capture immuable de la séquence alternée, fenêtre et six canaux d'entrée |
| `swing_context_coverage` | Plages de confirmations déjà traitées, par run, fenêtre et symbole, y compris les plages sans contexte exploitable |
| `swing_embedding_models` | Version, scaler, période d'apprentissage et empreinte de TS2Vec |
| `swing_embeddings` | Vecteur float32 rattaché au contexte et au modèle TS2Vec |
| `swing_experiments` | Expérience d'apprentissage/validation et modèle choisi |
| `swing_assignments` | Affectation des contextes à apprentissage, validation ou test et disponibilité des labels |
| `swing_resolvers` | Métadonnées et résultats des modèles XGBoost |
| `swing_predictions` | Prédictions, décisions et comparaison sans embedding, rattachées aux contextes |
| `swing_tests` | Évaluations séparées sur le test avec modèle figé |

Les snapshots et clôtures sont validés en transaction à chaque fichier journalier, avec WAL et `synchronous=FULL`. Les poids TS2Vec et XGBoost sont conservés dans des fichiers de modèle, dont les chemins et empreintes sont enregistrés en base. JSON, CSV et Markdown sont des exports de consultation ; la base reste le stockage principal.

Une nouvelle collecte sans `--run-id` crée un nouveau run dans la même base.
`collect --run-id IDENTIFIANT --symbols USDJPY` ajoute un nouveau symbole à
un run terminé, en conservant son identifiant, ses dates et ses paramètres.
Changer la marge de stop ou le trailing implique de recalculer les résultats
dans un nouveau run. L'étape de test vérifie que ces paramètres correspondent
à ceux du modèle figé ; ce contrôle de protocole n'est pas un filtre de collecte.

Une interruption d'une collecte initiale conserve les transactions journalières
déjà validées et marque le run comme incomplet/échoué. La reprise automatique
au milieu du flux n'est pas implémentée. Pour un ajout de symboles, une erreur
ou une interruption interceptée retire les données partielles de cet ajout
et restaure le run terminé précédent ; l'ajout peut alors être relancé depuis
le début. Seuls les runs terminés sont utilisés par l'apprentissage et le test.

## Commandes

Depuis `C:\Users\Babs\workspace\TradingBot\ultimateBot` :

```powershell
py -m pip install -r requirements-validator.txt
```

Produire les données 2020–2025, avec une marge d'exemple de 2 pips et les deux modes de contexte :

```powershell
py run_swing_ts2vec.py collect --symbols EURUSD GBPUSD USDCAD USDCHF --start-date 2020-01-01 --end-date 2025-12-31 --candle-interval 60m --durations 12h 24h 48h --pairs 4 8 16 --stop-buffer-pips 2
```
Exemple:
```powershell
py run_swing_ts2vec.py collect --symbols EURUSD GBPUSD USDCAD USDCHF --start-date 2020-01-01 --end-date 2022-12-31 --candle-interval 10m --pairs 500 --stop-buffer-pips 10
```

Le résumé affiche le `run_id`. Le copier pour figer la collecte utilisée :

```powershell
$swingTrainRun = "RUN_ID_AFFICHE"
py run_swing_ts2vec.py train --run-id $swingTrainRun --train-start 2020-01-01 --train-end 2023-12-31 --validation-start 2024-01-01 --validation-end 2025-12-31
```

### Ajouter USDJPY à cette collecte

Une fois le premier collecteur terminé, fournir son identifiant et les
symboles à ajouter :

```powershell
py run_swing_ts2vec.py collect --run-id $swingTrainRun --symbols USDJPY --dry-run
py run_swing_ts2vec.py collect --run-id $swingTrainRun --symbols USDJPY
```

Les fichiers de ticks doivent déjà exister sous `C:\jforex_history\USDJPY`.
`--history` et `--database` permettent d'utiliser d'autres dossiers. Le premier
appel prévisualise les paramètres sans écrire en base ; le second collecte.
Le résumé conserve le même `run_id` et indique `added_symbols`.

Les dates, l'intervalle des bougies, les fenêtres, le stop, le trailing,
l'expiration et le risque virtuel sont hérités du run : il n'est pas
nécessaire de les recopier. Toute valeur explicite incompatible est refusée.
Ainsi, si USDJPY a été téléchargé de 2015 à aujourd'hui mais que le run couvre
2020–2022, son ajout ne traitera que 2020–2022. Un run en cours ne peut pas
recevoir un second collecteur simultané ; attendre sa fin avant l'ajout.

Les symboles déjà présents sont ignorés, y compris si la liste mélange
anciens et nouveaux symboles. Les nouvelles opportunités, leurs contextes et
leurs résultats sont ajoutés ; les trades et modèles existants restent
inchangés. La liste des symboles, les statistiques et l'empreinte des sources
du run sont actualisées après succès. Un modèle déjà entraîné conserve son
ancien périmètre ; entraîner un nouveau modèle pour inclure USDJPY.
Une recherche commencée avant l'ajout conserve son ancienne empreinte :
lancer une nouvelle recherche sur le run étendu pour comparer les nouvelles
données.

### Choisir les symboles pour apprendre

`--symbols` est déjà disponible pour `train` et `search`. Il sélectionne
exactement les symboles concernés, à la fois pour TS2Vec et pour XGBoost :

```powershell
# Apprentissage groupé sur EURUSD et USDJPY uniquement.
py run_swing_ts2vec.py train --run-id $swingTrainRun --symbols EURUSD USDJPY --window-mode duration --durations 12h --target trailing-step-1

# Recherche de fenêtres sur ce même groupe.
py run_swing_ts2vec.py search --run-id $swingTrainRun --symbols EURUSD USDJPY --durations 6h 12h 24h --pairs 4 8 16 --target trailing-step-1 --repeats 3
```

Reprendre les bornes d'apprentissage/validation correspondant au run si elles
diffèrent des défauts 2020–2023 / 2024–2025. Un seul symbole, par exemple
`--symbols USDJPY`, est également possible. Sans cette option, tous les
symboles collectés du run sont regroupés. Chaque fenêtre garde son propre
couple TS2Vec/XGBoost ; cette sélection ne crée pas un modèle par symbole.

Ces quatre dates sont déjà les défauts. XGBoost prédit le PnL en R (`--target pnl-r`). Pour prédire la probabilité d'un PnL strictement positif : `--target win`.

Pour apprendre directement le niveau de trailing plutôt qu'une classe binaire,
`train`, `run` et `search` acceptent aussi ces trois régressions :

| Cible | Label pour `s = trailing_steps` | Labels pour s = 0, 1, 2, 3 |
| --- | --- | --- |
| `trailing-step-raw` | s | 0, 1, 2, 3 |
| `trailing-step-square` | s² | 0, 1, 4, 9 |
| `trailing-step-exp` | exp(s), base e | 1, 2,718, 7,389, 20,086 |

```powershell
py run_swing_ts2vec.py search --run-id $swingTrainRun --symbols EURUSD USDJPY GBPUSD USDCAD USDCHF --train-start 2018-01-01 --train-end 2022-12-31 --validation-start 2023-01-01 --validation-end 2023-12-31 --target trailing-step-square --durations 6h 12h 24h 48h --repeats 5
```

Ces modes utilisent `XGBRegressor`, une erreur quadratique et un arrêt anticipé
sur le RMSE de validation. Les labels viennent de la colonne `trailing_steps`
des trades clôturés ; les étapes inconnues, négatives ou non entières sont
exclues, sans les remplacer par zéro. Les clôtures doivent rester connues avant
la fin de leur période. Le trailing réalisé reste un label, jamais une variable
explicative du modèle.

Le score prédit est dans l'échelle du label choisi : nombre de paliers, carré
ou exponentielle. Aucune conversion inverse n'est appliquée. Le seuil fixe et
le pas adaptatif du backtest utilisent cette même échelle et ne sont pas
bornés à [0,1]. Le seuil enregistré maximise **le label moyen des trades
acceptés en validation**, sans minimum de trades. La moyenne porte sur les
labels déjà transformés, pas sur une transformation de la moyenne des steps.
Les diagnostics
comparent le MSE à une prédiction constante apprise sur les labels du train ;
le MSE brut de deux transformations différentes n'est pas directement comparable.

Le carré et l'exponentielle amplifient l'influence des grands trailings sur
l'erreur d'apprentissage, sans garantir une meilleure séparation ni un meilleur
PnL hors échantillon. L'exponentielle croît très vite : exp(0) vaut 1 ; à partir
de s = 45, l'outil refuse explicitement les labels dont l'erreur quadratique
dépasserait la capacité numérique float32 de XGBoost, avant l'entraînement de
TS2Vec. Il ne plafonne pas silencieusement les labels.

La collecte existante et ses contextes en cache sont réutilisables. Changer
`--target` démarre une nouvelle recherche ; cela ne transforme pas les modèles
binaires déjà entraînés et ne reprend pas leur recherche avec une autre cible.

Pour apprendre la probabilité d'atteindre **au moins la step 1 du trailing** :

```powershell
py run_swing_ts2vec.py train --run-id $swingTrainRun --target trailing-step-1
```

Cette option est également disponible avec `run` et avec la recherche répétée :

```powershell
py run_swing_ts2vec.py search --run-id $swingTrainRun --target trailing-step-1 --durations 6h 12h 24h 48h --pairs 4 8 16 32 --repeats 3
```

Les dates restent configurables : reprendre les bornes d'apprentissage et de
validation correspondant à la collecte sélectionnée. Une collecte en cours
peut continuer avec son processus actuel ; attendre son statut `completed`
pour l'entraîner. Aucune nouvelle collecte ni réécriture des trades n'est
nécessaire. La vue SQLite `swing_event_labels` est créée automatiquement à
l'ouverture de la base par le nouvel outil : elle expose
`trailing_step_1_label`, calculé à partir des données existantes et des futures
clôtures du collecteur.

| État du trade | Label XGBoost |
| --- | --- |
| Clôturé, `trailing_steps = 0` | 0 |
| Clôturé, `trailing_steps >= 1` | 1 |
| Étape inconnue, invalide, ou position encore ouverte | NULL, sans label d'apprentissage |

Dans le moteur actuel, l'activation et le passage au break-even correspondent
encore à la step 0. La step 1 nécessite une progression supplémentaire d'un R
après l'activation. Avec un démarrage du trailing à 1R, elle correspond donc
à un prix exécutable ayant atteint 2R. Le label utilise directement la colonne
`trailing_steps`, pas le signe du PnL final ni une approximation par `mfe_r`.
Les trades des deux classes restent présents : on ne filtre pas la base pour
ne garder que les réussites.

XGBoost utilise une classification binaire ; sa sortie est la **probabilité
entre 0 et 1 d'atteindre cette step**, et le seuil d'acceptation est lui aussi
compris entre 0 et 1. Un score élevé favorise l'acceptation (`score >= seuil`).
Le robot lit cette cible dans le modèle sauvegardé. Un seuil manuel se règle
avec `--swing-ts2vec-threshold 0.6`, et l'ajustement dynamique reste borné à
`[0, 1]`. Les anciens seuils négatifs de régression PnL/R ne sont pas valides
pour ce nouveau modèle.

La loss de classification et l'arrêt anticipé utilisent les labels binaires.
**Le choix du seuil et le classement des fenêtres maximisent le taux moyen
de labels positifs parmi les trades acceptés**, sans minimum de trades.
Les bilans financiers de validation et de test conservent les vrais `pnl_r`. La somme R,
le profit factor et le taux de trades gagnants restent des mesures du PnL,
distinctes du taux de labels positifs. Les CSV de prédictions incluent
`trailing_steps` et `target_label` pour vérifier cette distinction. La
définition de la cible et le critère de sélection sont enregistrés avec les
modèles et les rapports.

Seuls les trades clôturés avant la fin de leur période fournissent un label
d'apprentissage ou de validation, même si la step avait déjà été atteinte.
L'étape réalisée et les résultats du trade sont des **labels**, jamais des
variables fournies au modèle au moment de décider d'une entrée. TS2Vec reste
appris sur les seules séquences de swings. Au test figé, un trade avec PnL
connu mais step inconnue peut contribuer au bilan PnL ; il ne contribue pas
à l'erreur de classification. Cette erreur est calculée uniquement sur les
labels connus.

**Un couple TS2Vec/XGBoost correspond à un intervalle de bougie, un mode de fenêtrage et une taille de fenêtre fixés. Les paires peuvent être regroupées dans cet apprentissage.** Par exemple, H1 + durée 24h et H1 + 8 couples produisent deux encodeurs et deux modèles XGBoost distincts. Des encodeurs appris séparément ne partagent pas nécessairement les mêmes coordonnées latentes : leurs embeddings ne sont pas mélangés pour entraîner un XGBoost commun.

Pour fixer une seule fenêtre tout en regroupant toutes les paires du run :

```powershell
py run_swing_ts2vec.py train --run-id $swingTrainRun --candle-interval 60m --window-mode duration --durations 24h --ts2vec-output-dims 64
py run_swing_ts2vec.py train --run-id $swingTrainRun --candle-interval 10m --window-mode pairs --pairs 500 --ts2vec-max-train-length 1000 --ts2vec-output-dims 64
```

`--durations 24h` seul sélectionne uniquement cette durée ; `--pairs 8` seul sélectionne uniquement ce nombre de couples. `--pairs 4 8 16` entraîne trois couples de modèles distincts, puis les compare sur la validation. `--window-mode duration` sans liste compare toutes les durées déjà collectées. Sans sélecteur de fenêtre, toutes les variantes collectées sont comparées séparément.

`--symbols` est facultatif : sans cette option, toutes les paires du run participent au même apprentissage pour une fenêtre donnée. Si besoin, `--symbols EURUSD GBPUSD` limite cet ensemble à deux paires ; cela ne crée pas un modèle par paire. Les mêmes sélecteurs s'appliquent aux contextes TS2Vec et aux résultats de trades XGBoost. Le périmètre est enregistré en base et avec les modèles, puis conservé au test.

`--candle-interval` sélectionne une collecte existante à cet intervalle ; il ne reconstruit pas du H4 à partir des swings H1. Il faut d'abord collecter les bougies H4 avec `collect --candle-interval 4h`. Les tailles demandées doivent également avoir été collectées. Un `--run-id` incompatible provoque une erreur explicite ; sans `--run-id`, le programme choisit le dernier run terminé compatible avec les sélecteurs. Les bornes `--train-start`, `--train-end`, `--validation-start` et `--validation-end` ciblent les périodes calendaires et sont indépendantes de la durée du contexte.

La sélection des fenêtres et des seuils se fait uniquement sur la validation et sur les **mêmes opportunités disposant de toutes les fenêtres sélectionnées pour cette comparaison**. Une fenêtre non sélectionnée ne limite pas les effectifs. Un modèle de comparaison sans embedding utilise seulement le sens du trade, le nombre de swings et la durée du contexte. Les résultats et les effectifs sont enregistrés dans la base et dans `rapport.md` / `comparison.csv` du dossier d'expérience affiché. Un résultat de validation négatif reste signalé comme tel : le programme ne le présente pas comme une stratégie gagnante.

## Recherche répétée des modes et tailles après collecte

`train` effectue un apprentissage par fenêtre déjà collectée. La commande
`search` explore une grille plus large et répète chaque apprentissage avec
plusieurs graines aléatoires. Il n'est pas nécessaire de relire les ticks ni
de recalculer les résultats des trades : les événements de swings stockés
suffisent pour construire les nouveaux contextes. Les tailles demandées à
`search` peuvent donc être absentes de la collecte initiale. L'intervalle des
bougies, le stop et le trailing restent ceux de cette collecte.

Exemple depuis `ultimateBot`, après une collecte couvrant les périodes
d'apprentissage et de validation choisies :

```powershell
# Prévisualisation : aucune modification de la base et aucun entraînement.
py run_swing_ts2vec.py search --run-id $swingTrainRun --duration-range 6h 72h 6h --pairs-range 2 32 2 --repeats 3 --dry-run

# Même grille, avec entraînement effectif.
py run_swing_ts2vec.py search --run-id $swingTrainRun --duration-range 6h 72h 6h --pairs-range 2 32 2 --repeats 3
```

Cette grille comporte 12 durées et 16 nombres de couples : **28 fenêtres × 3
répétitions = 84 couples de modèles TS2Vec/XGBoost**, plus les comparateurs
sans embedding. Les bornes sont incluses quand le pas les atteint. On peut
commencer par une grille plus courte :

```powershell
py run_swing_ts2vec.py search --run-id $swingTrainRun --durations 6h 12h 24h 48h --pairs 2 4 8 16 --repeats 3
```

Sans liste ni plage, `search` reprend les fenêtres déjà enregistrées et les
répète trois fois. `--window-mode duration` ou `pairs` limite la famille à
explorer. Les listes et les plages sont fusionnées sans doublons. Les options
de `train` restent disponibles, notamment les dates, l'intervalle, le groupe
de paires, les dimensions de l'embedding et le nombre d'itérations. Pour un
run limité à 2020–2022, préciser par exemple `--train-end 2021-12-31
--validation-start 2022-01-01 --validation-end 2022-12-31` ; la recherche
n'invente pas les années absentes.

`--max-depths 3 5 7 9` ajoute une grille de profondeurs **XGBoost** au search,
avec ou sans TS2Vec. Chaque combinaison fenêtre/profondeur possède ses propres
répétitions, avec la même séquence de seeds à chaque profondeur. Le classement
utilise le score cible médian de chaque combinaison ; il ne mélange pas les profondeurs
comme s'il s'agissait de répétitions. Le plan multiplie donc le nombre de modèles
par le nombre de profondeurs distinctes. En mode TS2Vec, chaque essai exécute
le pipeline d'encodage et d'arbres ; en mode `swing-grid`, les matrices en cache
sont réutilisées entre profondeurs.
Sans cette option, `--max-depth` reste une valeur fixe (3 par défaut).
Le résumé affiche `selected_max_depth`, et les exports indiquent la profondeur
du modèle retenu. La reprise exige la même grille ; les anciennes recherches
à profondeur fixe restent compatibles. Voir [l'exemple complet](swing_grid.md#comparer-les-profondeurs-des-arbres).

Chaque fenêtre possède un encodeur et un XGBoost distincts, pour chaque
répétition. Les graines TS2Vec et XGBoost partent de `--ts2vec-seed` et
`--seed`, puis augmentent de 1 à chaque répétition. Pour les fenêtres en
couples, la limite de longueur d'entraînement TS2Vec est automatiquement
portée au minimum à deux fois le plus grand nombre de couples demandé. Pour
les durées, cette longueur reste réglable par `--ts2vec-max-train-length`.
La configuration effective figure dans la prévisualisation.

Toutes les fenêtres et répétitions utilisent la même cohorte de trades
éligibles. Une fenêtre trop grande peut réduire cette cohorte ou provoquer
un refus pour effectifs insuffisants. Les labels qui débordent de leur
période sont exclus. La construction des contextes de recherche s'arrête à
la fin de validation et ne lit ni les prix ni les résultats futurs.

Le classement privilégie la **médiane du score cible retenu en validation**
sur les répétitions : somme R pour `pnl-r`, label moyen pour les autres cibles.
Il compare ensuite le pire score et la couverture médiane en cas
d'égalité. Il affiche aussi le minimum, le maximum, l'écart-type et le nombre
de répétitions positives. Le modèle exporté est celui de la répétition la
plus proche de la médiane, avec son propre seuil figé. La sélection ne prend
donc pas le meilleur entraînement isolé. Chaque répétition sélectionne son
seuil et son nombre d'arbres sur la validation, comme `train`.

À la fin de `train`, `run` et `search`, le résumé affiche `selected_threshold`
et `selected_target`, puis met en évidence le **seuil initial du modèle retenu**
avec l'option `--swing-ts2vec-threshold` prête à recopier. Ce seuil exact est
aussi indiqué en début de `rapport.md` et dans `selected.json` (`threshold`,
`target`). Pour une recherche répétée, il s'agit du seuil propre à la répétition
exportée, la plus proche de la médiane. Sans surcharge explicite, le backtest
utilise déjà automatiquement ce seuil enregistré. L'état `validation_qualified`
reste affiché pour distinguer un modèle retenu d'un modèle satisfaisant les
critères de qualification.

Le dossier affiché contient `search.json`, `leaderboard.csv`, `rapport.md`,
les expériences individuelles et un `selected.json` compatible avec le test
figé et le robot :

```powershell
$swingSearch = "C:\TradingBotData\swing_ts2vec\experiments\search_RUN_RECHERCHE"
py run_swing_ts2vec.py test --resolver $swingSearch --run-id $swingTestRun --test-start 2026-01-01 --test-end 2026-12-31
py run_backtest.py --strategy swing-ts2vec --swing-ts2vec-model $swingSearch --start-date 2026-01-01 --end-date 2026-12-31
```

Si la fenêtre figée n'était pas collectée pour le test, `test` la reconstruit
depuis les swings de ce run, sans nouvel apprentissage ni nouveau seuil.
Les nouveaux contextes sont ajoutés à `swing_contexts`, leur couverture précise à
`swing_context_coverage`, leur préfixe commun à `swing_context_builds`, et la recherche à `swing_window_searches` ; les
modèles, embeddings et prédictions de chaque répétition utilisent les tables
d'expérience habituelles. Les labels et les anciens contextes restent inchangés.

Après interruption, reprendre avec **les mêmes options** et ajouter :

```powershell
--resume $swingSearch
```

Les répétitions terminées sont réutilisées ; une répétition interrompue est
recommencée. Aucun `selected.json` n'est publié avant la fin de la recherche.

La construction des fenêtres est un cache persistant par **run, fenêtre,
symbole et plage de dates**. Une recherche demande seulement les symboles
sélectionnés, du début de l'apprentissage à la fin de la validation. `test`
demande seulement sa période de test et les symboles du modèle. Les plages
déjà couvertes sont réutilisées, même si les hyperparamètres ou le dossier de
recherche changent. Une extension comble uniquement les trous de couverture.

Pour une fenêtre de durée, le rejeu commence au plus une durée de fenêtre et
une bougie avant la plage manquante. Pour une fenêtre de couples, le moteur
remonte seulement jusqu'au nombre de swings alternés nécessaire, ou au dernier
retournement ambigu qui réinitialise la série. Les contextes de cette préparation
antérieure ne sont pas matérialisés en dehors de la plage demandée. Les
remplacements par le dernier swing de même nature sont respectés.

Ainsi, pour apprentissage 2022, validation 2023 et fenêtres 48 h/72 h/96 h,
une fenêtre déjà couverte n'est pas reconstruite. Si seule 72 h manque, seuls
les symboles demandés sont traités entre 2022 et fin 2023, avec une préparation
commençant fin décembre 2021. Le rejeu ne repart pas automatiquement de 2018.

Contextes et couverture sont sauvegardés ensemble après chaque paire terminée.
`cache=reused` indique les fenêtres déjà disponibles ; `checkpoint=reused`
indique une paire déjà couverte. Le message `checkpoint=saved` confirme une
sauvegarde validée en base. Après interruption, seules les plages manquantes
de la paire interrompue sont recommencées. La couverture enregistre aussi les
plages calculées sans contexte exploitable, afin de ne pas les recalculer.
Les sauvegardes antérieures dans `swing_context_build_checkpoints` et
`swing_context_builds` restent reconnues sans reconstruire leurs données.
`swing_context_builds.completed_before` représente le préfixe couvert en commun
depuis l'origine sur toutes les paires, et ne revendique pas une couverture
globale après une construction limitée à quelques dates ou symboles.

`search --dry-run` expose `context_build_preflight` avec les plages manquantes
par fenêtre et symbole. Aucun rejeu ni migration n'a lieu pendant cet aperçu.
Un processus Python déjà démarré conserve la version de code qu'il a chargée :
pour appliquer cette optimisation à une recherche en cours, interrompre avec
Ctrl+C puis relancer les mêmes options avec `--resume` vers son dossier.

Pour les anciennes collectes, le premier instant d'observation des bougies
n'est pas enregistré : la reconstruction utilise conservativement le début
de la paire de bougies du premier retournement. Cela peut raccourcir les
contextes disponibles au tout début, sans inventer de couverture antérieure.
Les remplacements par le swing de même nature le plus récent et les remises
à zéro lors des retournements simultanés ambigus sont conservés.

La fenêtre retenue est la meilleure **dans la grille explorée**, pas un
optimum global démontré. Répéter les graines mesure la variabilité de
l'apprentissage ; cela ne crée pas de nouvelles validations temporelles.
Multiplier les essais peut surajuster la validation 2024–2025. Le test 2026
reste séparé et ne sert jamais à classer les fenêtres. Les sommes R portent
sur des opportunités indépendantes, potentiellement simultanées ; elles ne
constituent pas un rendement de portefeuille.

Pour collecter ensuite le test, on peut inclure décembre 2025 pour alimenter les premiers contextes de 2026. Garder la même liste de fenêtres, la même marge et le même trailing :

```powershell
py run_swing_ts2vec.py collect --symbols EURUSD GBPUSD USDCAD USDCHF --start-date 2025-12-01 --end-date 2026-12-31 --candle-interval 60m --durations 12h 24h 48h --pairs 4 8 16 --stop-buffer-pips 2
$swingTestRun = "NOUVEAU_RUN_ID_AFFICHE"
$swingModelDir = "DOSSIER_EXPERIENCE_AFFICHE_PAR_TRAIN"
py run_swing_ts2vec.py test --run-id $swingTestRun --resolver $swingModelDir --test-start 2026-01-01 --test-end 2026-12-31
```

Seuls les fichiers réellement disponibles sont lus. Si 2026 est incomplet, le rapport de test indique la dernière confirmation observée et les effectifs réellement labellisés ; une période future n'est pas simulée. `test` prend par défaut le modèle choisi par `selected.json`, sans refaire de sélection sur 2026. Les positions non clôturées avant la fin du test ne deviennent pas des pertes artificielles.

`run` combine `collect` puis `train`, sans tester 2026. `--window-mode duration` et `--window-mode pairs` permettent de limiter une collecte à un mode. `--dry-run` affiche les paramètres sans créer de données. Les checkpoints sont enregistrés sous `C:\TradingBotData\swing_ts2vec\experiments\` par défaut ; `--database` et `--output-dir` changent les chemins.

Pour les paires FX classiques, un pip vaut 0,0001 ou 0,01 pour une paire cotée en JPY. `--pip-size` permet de donner explicitement l'unité de prix par pip pour un autre symbole.

Exporter les opportunités ou les embeddings d'un modèle précis :

```powershell
py run_swing_ts2vec.py export --run-id $swingTrainRun --output C:\TradingBotData\swing_ts2vec\opportunities.csv
py run_swing_ts2vec.py export --run-id $swingTrainRun --model-id MODELE_TS2VEC_ID --output C:\TradingBotData\swing_ts2vec\embeddings.csv
```

Prédire sur un contexte déjà en base, avec un modèle disponible avant sa date :

```powershell
py run_swing_ts2vec.py score --resolver $swingModelDir --context-id CONTEXT_ID
```

## Utiliser les modèles entraînés

La valeur `directory` affichée à la fin de `train` désigne un dossier d'expérience. Ce dossier contient `selected.json`, qui pointe vers le modèle choisi sur la validation. On peut aussi donner directement le sous-dossier d'une fenêtre, par exemple `duration_43200`, pour tester ce modèle précis. Il faut conserver ensemble `metadata.json`, `report.json`, `ts2vec.pt`, `xgboost.json` et `xgboost_tabular_only.json`.

Deux chemins permettent d'évaluer les modèles figés :

| Chemin | Source des données | Mesure |
|---|---|---|
| `run_swing_ts2vec.py test --resolver ...` | Contextes et résultats déjà collectés dans SQLite | Résultat de la sélection parmi les opportunités indépendantes |
| `run_backtest.py --strategy swing-ts2vec --swing-ts2vec-model ...` | Ticks historiques relus directement | Résultat du robot avec capital, risque par trade et limites de positions |

Le test statistique demande une collecte couvrant la période après validation, avec les mêmes bougies, stop et trailing. On peut collecter uniquement la fenêtre du modèle testé. Si cette collecte existe déjà, la commande `test` la réutilise avec `--run-id` ; aucun nouvel apprentissage n'est nécessaire. `report.json` et `predictions.csv` sont écrits dans un nouveau dossier de test, et les prédictions/résultats sont aussi enregistrés dans SQLite.

Le mode du robot charge automatiquement l'intervalle des bougies, la fenêtre exacte, la marge de stop, l'expiration d'ordre, le délai d'exécution, le trailing et le seuil depuis les modèles. Par défaut, il reprend également les symboles de leur périmètre. Un intervalle, une expiration ou un délai incompatibles sont refusés. Les optimisations adaptatives de stratégie ne sont pas appliquées aux modèles figés.

Le backtest permet de remplacer le niveau d'activation du trailing avec
`--trailing-start-rr-ratio`, même s'il diffère du réglage d'entraînement :

```powershell
py run_backtest.py --strategy swing-ts2vec --swing-ts2vec-model $modele --trailing-start-rr-ratio 2 --start-date 2024-01-01 --end-date 2024-12-31
```

Ici, le passage au break-even commence à 2R, puis le stop continue de progresser
par paliers d'un R. Le niveau est positif et fini ; sans option, le réglage du
modèle reste utilisé. Il est recalculé sur le risque réel au fill, y compris
après un délai, pour les achats comme pour les ventes. La référence RR est
automatiquement suffisante par défaut ; une valeur explicite
`--reward-risk-ratio` doit rester supérieure au départ du trailing. Cette
référence ne ferme pas la position en mode trailing.

Les décisions et le bilan indiquent le niveau effectif et celui de
l'entraînement. Les poids, les labels et le seuil enregistré ne sont pas
modifiés : leurs résultats de validation décrivent toujours l'ancien réglage
de sortie. Le test statistique figé `run_swing_ts2vec.py test` conserve sa
vérification de conformité du trailing avec la collecte du modèle.

Pour remplacer entièrement le trailing par un take-profit fixe, ajouter
`--swing-ts2vec-take-profit-r` au backtest :

```powershell
py run_backtest.py --strategy swing-ts2vec --swing-ts2vec-model $modele --swing-ts2vec-take-profit-r 2 --start-date 2024-01-01 --end-date 2024-12-31
```

Le TP vaut alors **2 × la distance entre le prix d'entrée réellement exécuté et
le stop structurel initial**. Le trailing et son passage au break-even sont
désactivés ; le stop reste fixe. Le paramètre prend priorité sur
`--trailing-stop`, `--trailing-start-rr-ratio` et `--reward-risk-ratio`. La valeur
doit être positive et finie ; un TP inférieur à l'ancien départ du trailing est
autorisé. Le TP est recalculé au fill, pour les achats comme pour les ventes,
y compris après un délai d'exécution. Les prix de sortie restent les cotations
exécutables bid/ask, et peuvent dépasser le niveau lors d'un saut de prix.

Sans cette option, le trailing actuel est conservé. Les traces et le bilan
précisent le mode de sortie, le TP en R et le réglage de trailing d'entraînement.
Les scores, le seuil et les fichiers du modèle restent inchangés ; leur
validation historique décrit toujours les sorties utilisées à l'entraînement.

Le stop structurel peut aussi utiliser une autre marge que celle du modèle :

```powershell
py run_backtest.py --strategy swing-ts2vec --swing-ts2vec-model $modele --swing-ts2vec-stop-buffer-pips 5 --start-date 2024-01-01 --end-date 2024-12-31
```

Le SL est placé à 5 pips sous le pivot ask pour un achat, ou à 5 pips au-dessus
du pivot bid pour une vente. Il s'agit d'une marge par rapport au pivot, pas
d'une distance fixe depuis l'entrée. `0` supprime la marge ; sans option, le
réglage enregistré du modèle est conservé. Les valeurs négatives ou non finies
sont refusées. La taille du pip reste celle du modèle, ou celle déduite du
symbole (0,0001 en FX standard, 0,01 pour les paires cotées en JPY).

Le dimensionnement de l'ordre utilise ce nouveau SL. Le TP fixe et l'activation
du trailing sont calculés au fill depuis la distance réelle entrée–SL. Cette
option fonctionne avec le trailing ou avec `--swing-ts2vec-take-profit-r`.
Les décisions et le bilan affichent la marge utilisée et celle de l'entraînement.
Les fichiers, scores et labels du modèle restent inchangés : la validation
historique correspond toujours au stop d'origine.

Depuis le dossier `ultimateBot`, par exemple :

```powershell
$swingModelDir = "C:\TradingBotData\swing_ts2vec\experiments\DOSSIER_AFFICHE_PAR_TRAIN"
py run_backtest.py --strategy swing-ts2vec --swing-ts2vec-model $swingModelDir --start-date 2025-12-01 --end-date 2026-12-31 --balance 10000 --risk-percent 1 --max-positions 4 > swing_ts2vec_robot.log
```

Cet exemple suppose une validation terminée fin 2025. Les bougies de décembre préparent le contexte ; les premières entrées sont autorisées à partir du 1er janvier 2026. Adapter la période aux dates réelles inscrites dans `metadata.json`. Pour un contexte en couples très long, prévoir davantage d'historique avant la période d'entrée.

Pour remplacer le seuil d'acceptation uniquement pendant un backtest du robot, ajouter `--swing-ts2vec-threshold`. Par exemple :

```powershell
py run_backtest.py --strategy swing-ts2vec --swing-ts2vec-model $swingModelDir --swing-ts2vec-threshold 0 --start-date 2025-12-01 --end-date 2026-12-31 --balance 10000 --risk-percent 1 --max-positions 4
```

Le robot accepte alors les prédictions supérieures ou égales à zéro. Avec la cible `pnl-r`, les valeurs sont des scores en R et peuvent être négatives ; augmenter le seuil réduit ou maintient le nombre de décisions acceptées pour les mêmes prédictions. Avec la cible `win`, le seuil explicite doit être compris entre 0 et 1. Sans cette option, le seuil choisi sur validation reste utilisé. Les modèles et leur seuil enregistré restent intacts : le seuil effectif, le seuil d'origine et la provenance de la valeur (`model` ou `override`) sont tracés dans les décisions et dans le bilan du robot. La commande de test statistique `run_swing_ts2vec.py test` continue d'utiliser le seuil enregistré dans le modèle. Un seuil modifié constitue un nouveau réglage à évaluer sur validation avant le test final.

Deux tests supplémentaires vérifient la surcharge du seuil, les décisions à la frontière, le zéro explicite, les seuils négatifs en R, l'absence de modification des fichiers du modèle et le refus des valeurs non finies ou des probabilités hors bornes. L'ensemble des vérifications ciblées passe alors à **244 tests réussis**.

Pour adapter le seuil à une cadence pendant le backtest du robot, fournir `--swing-ts2vec-target-trades-per-day`. La valeur accepte les décimales positives. Le pas `--swing-ts2vec-threshold-step` est facultatif et vaut **0,001 par défaut** :

```powershell
py run_backtest.py --strategy swing-ts2vec --swing-ts2vec-model $swingModelDir --swing-ts2vec-target-trades-per-day 0.25 --swing-ts2vec-threshold-step 0.001 --start-date 2025-12-01 --end-date 2026-12-31 --balance 10000 --risk-percent 1 --max-positions 4
```

Le seuil initial provient du modèle, ou de `--swing-ts2vec-threshold` si cette option est donnée. La durée de chaque période est **1 / cadence jours**, pour toutes les valeurs, sans arrondir aux jours entiers : 0,25/jour donne 4 jours, 0,5/jour donne 2 jours, 1/jour donne 24 heures, 2/jour donne 12 heures, 1,5/jour donne 16 heures et 0,3/jour donne 80 heures. Chaque période vise **un signal accepté par le modèle, toutes paires confondues**, avant les contrôles de risque. Les périodes se suivent en temps écoulé UTC, sur une grille initialement ancrée au premier jour observé.

Le deuxième signal accepté dans une période relève immédiatement le seuil d'un pas pour les décisions suivantes ; chaque acceptation supplémentaire ajoute un pas. Un rejet ne relève pas le seuil. Si une période avec des prédictions se termine sans aucune acceptation, le seuil descend d'un pas au premier traitement de la période suivante. À exactement une acceptation, il est conservé. Les compteurs ne sont donc pas remis à zéro chaque nuit lorsque la période dure plusieurs jours, et une période dépassant son objectif ne provoque pas une nouvelle hausse à sa clôture.

Il s'agit d'un **objectif de cadence, sans plafond strict ni garantie d'obtenir exactement ce nombre d'entrées**. Les décisions passées restent intactes et aucun résultat futur de trade n'est utilisé. La préparation antérieure à la disponibilité du modèle et les périodes sans contexte exploitable ne font pas baisser le seuil. Les périodes sans données ne sont pas ajoutées comme des périodes à zéro trade. La dernière période du replay est affichée dans son état courant ; sa clôture n'est appliquée que si un traitement ultérieur atteint la période suivante.

Avec la cible `pnl-r`, le pas s'exprime en score R ; avec `win`, il s'exprime en probabilité et le seuil évolue dans [0,1]. Le mode adaptatif ajuste seulement le seuil en mémoire, sans modifier les poids ni le seuil enregistré. Les traces optionnelles de décision et les contextes d'ordres conservent le seuil effectivement utilisé pour chaque signal, même lorsqu'une acceptation déclenche sa hausse pour le signal suivant. `[swing-ts2vec-threshold-period]` donne les bornes et le bilan de chaque période observée terminée ; le bilan final affiche la cadence, la durée, le pas, les hausses, les baisses et le seuil courant. Sans cadence, le seuil reste fixe ; un pas seul est refusé. Le test statistique `run_swing_ts2vec.py test` conserve son protocole à seuil enregistré fixe.

Pour empêcher l'ajustement adaptatif de descendre sous un score donné, ajouter
`--swing-ts2vec-min-threshold`, par exemple :

```powershell
--swing-ts2vec-threshold 0.30 --swing-ts2vec-min-threshold 0.25 --swing-ts2vec-threshold-step 0.001 --swing-ts2vec-target-trades-per-day 0.1
```

Le seuil peut alors descendre de 0,30 à 0,25, puis reste au minimum à 0,25 même
si la cadence n'est pas atteinte. Il peut toujours remonter. Si le seuil initial
enregistré ou fourni explicitement est inférieur au minimum, il est relevé au
minimum dès le départ, y compris sans cadence adaptative. Sans cette option,
aucun plancher supplémentaire n'est appliqué. Le minimum est fini et dans
[0,1] pour `win` et `trailing-step-1` ; les régressions utilisent l'échelle du
score choisi (R, trailing brut, carré ou exponentielle). Il filtre les scores
prédits et ne garantit pas le résultat des trades. Il apparaît dans la
progression, les décisions, les bilans de période et le résumé final ; les
poids et le seuil enregistré du modèle restent inchangés.

Les tests couvrent les périodes de 4 jours, 12 heures, 16 heures et 80 heures, la conservation des compteurs sur plusieurs jours, le pas par défaut de 0,001, le dépassement immédiat, les périodes sans contexte, les bornes de probabilité et le regroupement des paires avant les rejets de risque. Les anciens fichiers `artifacts/swing_ts2vec_dynamic_threshold.log` et `.json` décrivent la première version à objectif quotidien entier ; ils ne représentent pas la règle actuelle à un signal par période réciproque.

Les vérifications ciblées totalisent **256 tests réussis**. Le contrôle actuel sur les ticks EURUSD des 2–3 janvier 2023 utilise une cadence de 0,25/jour sans préciser le pas : la période reste ouverte du 2 au 6 janvier, le pas effectif vaut 0,001 et les 70 contextes des deux jours alimentent le même compteur. Trois signaux sont acceptés et provoquent deux hausses ; il n'y a aucun reset quotidien. Les preuves sont `artifacts/swing_ts2vec_fractional_threshold.log` et `.json`. Ce contrôle vérifie le mécanisme, sans constituer une évaluation financière de la cadence.

À chaque clôture, le robot met à jour la série alternée bid-high/ask-low, même quand des positions sont déjà ouvertes. Si le contexte est prêt et la date du modèle respectée, il calcule l'embedding puis la prédiction XGBoost. Une décision acceptée produit un ordre au marché avec le stop structurel du modèle. L'ordre utilise la première cotation éligible après la confirmation et son éventuel délai ; le trailing est recalculé à partir du prix réellement exécuté. L'expiration ne ferme pas les positions remplies.

Les traces `[swing-ts2vec-decision]` sont **désactivées par défaut**. Ajouter `--swing-ts2vec-decision-log` (ou `--swing-ts2vec-decision-log accepted`) pour afficher uniquement les signaux acceptés par le modèle. `--swing-ts2vec-decision-log all` affiche aussi les décisions rejetées pour le diagnostic. `--swing-ts2vec-decision-log off` ou `--no-swing-ts2vec-decision-log` désactive ces traces. Quand elles sont activées, elles indiquent le score, le seuil et l'acceptation du modèle ; l'affichage ne change pas les prédictions ni les ordres.

`[trade-created]`, `[trade-filled]`, `[trailing-stop]` et `[trade-closed]` suivent les ordres et positions. Le bilan final indique les décisions acceptées/rejetées et les rejets par le gestionnaire du risque, en plus du bilan habituel du robot. Les ordres acceptés par le modèle peuvent être moins nombreux à être exécutés, notamment à cause du spread ou du plafond de positions. Les ordres retenus restent au marché : le filtre de spread du robot peut les rejeter, mais ne les transforme pas en entrées limit.

`[swing-ts2vec-progress]` affiche la date simulée, les compteurs et la phase dès la première fenêtre de bougie traitée, puis à chaque changement de jour ou de phase, ou après 30 secondes lors d'une nouvelle fenêtre. `warmup_before_model` signifie que les ticks sont antérieurs à la disponibilité du modèle : aucune entrée n'est autorisée, quel que soit le seuil. `trading_period` signifie que cette date est atteinte ; il faut aussi disposer du contexte de swings nécessaire. Un démarrage au 15 décembre avant un modèle disponible au 1er janvier peut donc rester longtemps sans décision pendant la lecture des ticks de décembre. L'affichage de progression ne modifie pas les décisions du modèle.

Le diagnostic avec le modèle utilisateur `217311c9b408_078d8333df26/duration_43200` et un seuil de `-5.048` confirme 70 décisions acceptées sur EURUSD les 2 et 3 janvier 2023 : 53 positions exécutées, dont 43 clôturées et 10 encore ouvertes, avec 17 signaux rejetés par la gestion du risque. Ce contrôle vérifie le fonctionnement des entrées après la validation terminée fin 2022 ; il ne constitue pas une évaluation annuelle de la stratégie. Son log est `artifacts/swing_ts2vec_no_trades_diagnostic.log`. Un test supplémentaire vérifie l'affichage pendant la préparation, son actualisation et le passage à la période de trading sans déclencher de prédiction prématurée. Les vérifications ciblées totalisent **245 tests réussis**.

Le mode `swing-ts2vec` est branché au moteur de **backtest** du robot. Il n'ajoute pas de connexion à un courtier pour exécuter des ordres en réel ou en démo.

## Vérification

Les tests dédiés couvrent les sources bid/ask, le remplacement par le dernier swing, les deux fenêtres, les dates de disponibilité, les stops élargis, les pertes immédiates sans veto, les positions simultanées, l'expiration d'ordre seulement, TS2Vec réel, les deux cibles XGBoost et le test séparé sans appel à `fit`.

Un essai technique sur EURUSD janvier–février 2024 utilise des bornes calendaires raccourcies pour vérifier toute la chaîne : apprentissage en janvier, validation du 1er au 15 février, puis test séparé du 16 au 29 février. Il ne constitue pas l'expérience financière 2020–2026. Ses fichiers sont dans `artifacts/swing_ts2vec_validation/`.

Cet essai lit 3 882 506 ticks et enregistre 520 opportunités : 506 clôturées, 6 encore ouvertes, 7 ordres expirés et 1 en attente à la fin. Quatre variantes de fenêtre sont apprises. Les 111 contextes du test séparé sont prédits après gel du modèle ; 103 disposent d'un résultat connu dans la période du test. Les empreintes des trois checkpoints sont identiques avant et après ce test. Ce petit essai ne démontre aucun avantage financier.

La référence sur tous les ticks et la simulation accélérée produisent exactement les mêmes champs d'événements et les mêmes contextes sur 313 457 ticks et 33 opportunités réelles, avec une marge de 2 pips. Les preuves sont `validation.json` et `tick_equivalence.json` dans ce dossier.

Les 29 tests propres à cet outil et les tests ciblés des stratégies et de l'exécution donnent **234 tests réussis**. Ils vérifient aussi les sélecteurs d'entraînement, le regroupement des paires à fenêtre fixe, l'indépendance vis-à-vis des fenêtres non sélectionnées et le maintien du périmètre au test. L'un d'eux vérifie explicitement les dates par défaut : apprentissage sur les quatre années 2020–2023, validation sur 2024–2025, et aucune lecture de caractéristiques 2026 volontairement rendues inutilisables dans le jeu de test. Une vérification élargie des anciens modules d'embeddings retrouve par ailleurs un échec préexistant du lanceur de sweep : un test attend `adaptive_lookback_max=50` alors que sa configuration actuelle vaut 20. Ces fichiers n'ont pas été modifiés par cet outil.

Les anciennes expériences de mise au point dans `swing_ts2vec_smoke` et `swing_ts2vec_protocol_smoke` ne correspondent pas toutes au protocole final. La version de la politique d'exécution est enregistrée ; les premières collectes qui rejetaient les stops déjà franchis ne sont pas admises pour un nouvel apprentissage avec le collecteur sans veto.

L'intégration au robot ajoute huit tests couvrant le chargement des paramètres, la concordance des contextes avec le collecteur, l'observation continue pendant les positions, le refus des signaux hors période, les décisions du modèle, le premier tick exécutable, les achats à l'ask et ventes au bid, le sizing, le stop structurel, le trailing, les fills différés, les stops déjà franchis au fill et les limites de risque. Avec les autres vérifications ciblées, **242 tests réussissent**. Un essai technique sur les ticks EURUSD du 14 au 18 février 2024 charge un modèle appris en janvier et validé jusqu'au 15 février : 10 décisions après validation, 2 acceptées, 1 position clôturée par trailing et 1 encore ouverte. Les 10 prédictions correspondent exactement à celles calculées depuis les contextes collectés ; les empreintes des poids correspondent toujours à celles enregistrées après apprentissage. Ce contrôle d'intégration n'est pas le test financier du modèle utilisateur. Son log et sa preuve sont `artifacts/swing_ts2vec_robot_validation.log` et `artifacts/swing_ts2vec_robot_validation.json`.
