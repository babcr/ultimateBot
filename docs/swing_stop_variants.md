# Variantes de stop structurel dans les collectes de swings

Une nouvelle collecte peut simuler plusieurs trades indépendants pour chaque
swing confirmé. Leur entrée reste identique ; chaque variante possède son stop,
son risque initial, sa quantité, son trailing et son résultat réalisé.

```powershell
py run_swing_ts2vec.py collect `
  --symbols EURUSD USDJPY GBPUSD USDCAD USDCHF `
  --start-date 2020-01-01 --end-date 2026-09-30 `
  --candle-interval 1h `
  --stop-buffer-pips 2 `
  --stop-variants 4 --stop-step-pips 5 `
  --max-holding 2w
```

Ajouter `--dry-run` pour afficher les paramètres sans lancer la collecte.
La commande terminée fournit le nouveau `run_id` à utiliser pour l'apprentissage.
Le windowing est différé par défaut : la collecte ne construit ni ne stocke
de contexte générique. Les durées sont choisies ensuite dans `train` ou `search`.
`--store-contexts` permet de conserver le mode historique si nécessaire.

## Paramètres

| Option | Valeur par défaut en CLI | Rôle |
| --- | ---: | --- |
| `--stop-buffer-pips` | 0 | Marge de base au-delà du swing structurel |
| `--stop-variants` | 1 | Nombre total de trades par swing, stop de base compris |
| `--stop-step-pips` | 5 | Marge supplémentaire par variante successive |
| `--max-holding` | `2w` | Durée maximale d'une position depuis son ouverture effective |

Avec la commande ci-dessus, les quatre marges sont **2, 7, 12 et 17 pips**.
Pour un achat, le stop est sous le swing low ask ; pour une vente, il est au-dessus
du swing high bid. La taille d'un pip est inférée pour les paires FX standard,
dont 0,01 pour USDJPY ; `--pip-size` permet une valeur explicite.

La distance totale entrée–stop inclut la distance entre le prix d'entrée et le
swing structurel, puis cette marge. La quantité maintient le même risque cash
virtuel par variante. Le niveau d'activation du trailing et ses pas restent
calculés en multiples de **son propre risque initial** : élargir le stop éloigne
donc aussi le trailing. Aucun résultat n'est copié depuis la variante de base.

Les paramètres doivent être strictement positifs, sauf la marge de base qui
peut être nulle. `--max-holding` accepte par exemple `10d`, `2w` ou `3w`.
L'API lit les anciennes configurations sans limite comme auparavant ; les
variantes de stop nécessitent une limite de durée positive.

## Fermeture à durée maximale

La position est clôturée au premier tick de son symbole à l'échéance ou après,
au bid pour un achat et à l'ask pour une vente. L'absence de cotations, par
exemple pendant le week-end, reporte l'exécution à la première cotation disponible.
Le motif est `MAX_HOLDING_TIME`. Un stop ou take profit également touché à ce tick
garde sa priorité habituelle. Le simulateur accéléré prend l'échéance en compte
dans ses sauts et ne la franchit pas en sautant une cotation admissible.

Un ordre non exécuté garde son expiration distincte `--order-expiration-bars`.
Une position encore ouverte à la fin des données, dont l'échéance n'a pas pu
être exécutée, ne reçoit pas de PnL réalisé inventé. Les périodes d'apprentissage
et de validation ne conservent que les résultats connus avant leur propre fin.

## Nouvelle entrée des modèles

Pour une collecte avec plusieurs variantes, la caractéristique supplémentaire
est `stop_buffer_ratio` :

```text
stop_buffer_ratio = marge_en_pips × taille_du_pip / prix_du_swing_courant
```

La marge et le prix du swing sont connus au signal. On n'utilise pas le prix
futur d'exécution pour construire cette entrée. La caractéristique permet
de distinguer les variantes, dont le contexte de marché est identique.

- **Réseau neuronal dense / Conv1D et XGBoost swing-grid :** le scalaire est
  ajouté après la grille. Une fenêtre de 24 h sur bougies de 1 h donne donc
  **25 entrées**, dont 24 cases temporelles et une marge de stop. La normalisation
  des cases temporelles et leurs zéros restent définis comme précédemment.
- **TS2Vec + XGBoost :** le scalaire est ajouté aux entrées tabulaires de XGBoost.
  L'encodeur TS2Vec conserve son contexte de marché ; ses fenêtres communes sont
  dédupliquées pour le préentraînement et l'encodage.
- **Informer :** le scalaire est un canal connu et constant dans la séquence
  de chaque trade. Les points temporels restent partagés entre variantes.

Les métadonnées figées identifient les nouvelles dimensions et caractéristiques.
Les anciens modèles restent chargeables avec leurs anciennes entrées. Une
nouvelle collecte et un nouvel apprentissage sont nécessaires pour exploiter
la nouvelle caractéristique : les anciens labels ne sont pas transformés.

Exemple de campagne neuronale CPU sur le nouveau run :

```powershell
py run_swing_nn.py search `
  --run-id ID_RETOURNE_PAR_LA_COLLECTE `
  --train-start 2020-01-01 --train-end 2022-12-31 `
  --validation-start 2023-01-01 --validation-end 2023-12-31 `
  --durations 6h 12h 24h 48h `
  --architectures mlp --profiles balanced --seeds 1 2 3 `
  --output-dir C:\TradingBotData\swing_nn\search_stops_variables
```

## Base et backtest

Chaque trade possède un `event_id` et un `context_id` distincts. Les champs
`swing_id`, `variant_index` (de 0 à x−1) et `stop_buffer_pips` identifient sa
famille et son réglage. Les gros payloads de fenêtre ne sont stockés qu'une fois,
**lorsqu'un contexte générique est demandé** (TS2Vec à l'apprentissage, ou collecte
avec `--store-contexts`). Les autres contextes les référencent par
`shared_context_id`. Pour lire les
fenêtres complètes en SQL, utiliser **`swing_contexts_resolved`**.

Le cache des grilles calcule une seule fois le vecteur temporel d'un swing,
puis ajoute la marge propre à chaque observation. Les seeds réutilisent ce cache.
Le nombre d'exemples d'apprentissage augmente néanmoins avec x : les durées
d'entraînement et les matrices mises en cache augmentent en conséquence.
Ces variantes sont des réglages corrélés d'un même événement de marché,
pas de nouveaux événements indépendants.

Le backtest d'un nouveau modèle évalue les variantes **de la plus petite marge
à la plus grande** et s'arrête dès la première prédiction atteignant son seuil.
Il émet **au plus un trade par swing**, avec le stop de cette première variante
validée. Si aucune variante ne passe, aucun trade n'est émis. Si le risque refuse
le signal retenu, le swing est abandonné ; les distances suivantes ne sont pas
essayées comme solutions de repli. Le seuil adaptatif compte chaque swing une
seule fois, indépendamment du nombre de distances évaluées.
Les limites ordinaires de risque, d'exposition et de nombre de positions
s'appliquent au trade retenu. La durée
maximale est héritée de la collecte. Les réglages existants de seuil, trailing,
take profit fixe et marge de base restent utilisables. Un changement de marge
de base décale toute la grille de stops du même montant.

Ajouter des symboles avec `collect --run-id ...` hérite des variantes et de la
durée maximale du run, ainsi que du mode de stockage des contextes ; ces paramètres
ne peuvent pas être changés en extension.

## Validation et choix du seuil

Le calcul du PnL-R, du nombre de trades et de la couverture applique la même règle
que le backtest : pour chaque swing, on choisit **la première distance dont la
prédiction atteint le seuil**, en parcourant les marges de la plus petite à la
plus grande. Le PnL des autres variantes n'intervient pas dans cette décision.
Cela s'applique à la validation propre, à la comparaison commune des fenêtres et
des seeds, au témoin tabulaire et aux évaluations sur une période de test figée.
Les champs `accepted` des exports n'admettent eux aussi qu'une variante par swing.

Le seuil optimal est recalculé sous cette règle en évaluant les seuils distincts
réalisables. Abaisser le seuil peut remplacer un stop large par un stop plus court,
et faire baisser le R sans changer le nombre de trades. Le calcul tient compte de
ces remplacements, des scores égaux et de l'abstention ; il n'impose aucun minimum
de trades acceptés pour les collectes à stops variables. Sa complexité reste
O(n log n), avec un échantillon borné des seuils conservé dans les rapports.

Les métriques de trading portent sur les familles **complètes** : toutes les
variantes doivent avoir un résultat connu avant la fin de leur période. Si l'une
d'elles reste ouverte, est censurée à la frontière ou n'a pas de label utilisable,
le swing est écarté de ces métriques. Cela évite de remplacer silencieusement un
stop dont le résultat manque par un autre stop disponible. Les rapports donnent
`complete_swings`, `incomplete_swings`, `raw_variants` et la règle
`first_accepted_stop_per_swing_v1`. La couverture est calculée sur les swings
complets ; le nombre de lignes reste celui des variantes labellisées.

L'apprentissage et sa MSE continuent d'exploiter les labels disponibles de toutes
les variantes. La sélection d'époque sur MSE reste inchangée ; les critères de
gain et les sommes de R utilisent ensuite la politique de trading ci-dessus.
La validation conserve les dates demandées et n'exécute pas les contraintes de
portefeuille, de capital ou de seuil adaptatif propres à une commande de backtest.

Pour un modèle neuronal déjà figé, les prédictions de validation enregistrées
permettent de recalculer le seuil et les métriques dans un **nouveau** dossier,
avec les mêmes poids et sans réentraînement :

```powershell
py run_swing_nn.py recalibrate `
  --model C:\TradingBotData\swing_nn\ANCIEN_MODELE_OU_SEARCH_TERMINE `
  --output-dir C:\TradingBotData\swing_nn\validation_premier_stop
```

Les anciens modèles à stop unique gardent leur comportement habituel.
