# Réintégration après cassure échouée

Le mode `failed-breakout` teste un retour au centre d'un canal. Le mode
historique `trend` reste disponible pour les comparaisons. Les seuils initiaux
sont des hypothèses de départ, sans calibration ni probabilité de gain établie.

Le lanceur à paramètres fixes, depuis le dossier parent de `ultimateBot` :

```powershell
py -m ultimateBot.run_regression_reentry
```

Par défaut : USDCHF, GBPUSD, USDCAD, EURUSD ; 2020–2023 ; bougies de 10 minutes ;
régression sur 15 bougies ; historique de volatilité de 150 bougies ; risque
de 2,2 % ; filtre de session actif ; sans adaptation ni trailing.
Les options explicites permettent de changer les dates, symboles et paramètres :

```powershell
py -m ultimateBot.run_regression_reentry --symbols EURUSD --start-date 2024-01-01 --end-date 2024-01-31 --risk-percent 0.1
```

Le même moteur est accessible avec
`py -m ultimateBot.run_backtest --strategy swing-regression --regression-entry-mode failed-breakout ...`.
Le lanceur habituel conserve son mode `trend` par défaut.

## Événement

1. Calculer les régressions à partir des bougies clôturées **précédant** la
   bougie de dépassement, avec les mêmes swings high bid / low ask et les mêmes
   filtres de qualité du canal que le moteur historique. Les deux droites sont
   obligatoires. La pente minimale de tendance ne s'applique pas.
2. Rejeter les droites qui se croisent dans la fenêtre ou l'horizon de détection.
   Limiter la dérive maximale d'une droite sur la fenêtre à 0,5 fois la largeur
   minimale du canal. La clôture précédente doit être à l'intérieur.
3. Détecter un dépassement d'au moins 0,1 ATR : high bid au-dessus de la borne
   haute pour une vente, low ask sous la borne basse pour un achat.
4. Figer les coefficients et l'ATR. Attendre une clôture bid sous la borne haute
   et ask au-dessus de la borne basse, sur cette bougie ou dans les deux suivantes.
   Les valeurs des droites sont projetées avec les coefficients figés.
5. Créer une vente après excursion haute, un achat après excursion basse.
   Stop au-dessus du plus haut ask / sous le plus bas bid de l'événement,
   avec une marge de 0,2 ATR. Cible au centre du canal à la confirmation,
   ensuite conservée à prix fixe.

Une bougie dépassant les deux bornes est ambiguë et rejetée. Une excursion
touchant ensuite la borne opposée, une interruption de l'historique ou un délai
dépassé invalide l'événement. Une confirmation est consommée une seule fois,
même si la gestion du risque ou le filtre de session refuse ensuite l'ordre.
La stratégie n'utilise pas de bougie active et ne simule aucun fill à la clôture
qui a donné le signal : l'exécution reste celle du moteur et de la plateforme.

La gestion du risque conserve les filtres de spread et les règles d'instrument.
Elle dimensionne la quantité sur le stop explicite, puis vérifie un gain/risque
minimal de 1 au prix d'entrée exécutable (spread inclus). Une entrée devenue
incohérente avec le stop ou la cible est refusée. Le ratio générique
`--reward-risk-ratio` ne remplace pas cette cible.
Le replay ne modélise pas de commission supplémentaire ; le résultat inclut
le spread et les écarts de prix réellement simulés par le moteur.

## Paramètres et capture

| Option | Départ |
|---|---:|
| `--regression-reentry-max-bars` | 2 bougies supplémentaires |
| `--regression-reentry-max-drift-ratio` | 0.5 |
| `--regression-reentry-min-excursion-atr` | 0.1 |
| `--regression-reentry-stop-buffer-atr` | 0.2 |
| `--regression-reentry-min-rr` | 1.0 |

Ils figurent également dans `RegressionParameters` et le snapshot du run.
Les confirmations directionnelles historiques (tendance globale, réaction à
un niveau, couleur de bougie) sont désactivées dans ce mode. Le lanceur refuse
l'adaptation, le bandit et le trailing pour conserver ce premier protocole fixe.

Pour constituer un dataset distinct avec l'embedding capturé à la création :

```powershell
py -m ultimateBot.run_regression_reentry --trade-database C:\TradingBotData\regression_reentry.sqlite3 --ts2vec-model C:\TradingBotData\models\ts2vec_regression_reentry.pt --dataset-campaign-id regression-reentry-v1
```

Les embeddings, paramètres et labels PnL suivent le circuit existant. La table
`order_signal_contexts` ajoute les bornes, coefficients, excursion, ATR, délai,
stop et cible figés. Elle se joint à `closed_trades` sur `(run_id, order_id)`.
Le vecteur du valideur XGBoost existant n'est pas modifié ; les nouvelles
variables restent disponibles pour un entraînement ultérieur. Évaluer cette
nouvelle population séparément avant de réutiliser un valideur historique.
Le checkpoint dédié permet de démarrer l'apprentissage de l'encodeur à partir
du premier ordre observé de cette campagne, sans réutiliser par inadvertance
un modèle entraîné après le début du replay.

## Évaluation

Comparer `trend` et `failed-breakout` sur les mêmes ticks, sessions, spread,
lookback et risque, sans optimiser les seuils à partir du résultat de comparaison.
Leurs sorties diffèrent intentionnellement : 3R pour le témoin, centre pour la
réintégration. Ce test compare donc deux stratégies complètes ; il n'isole pas
à lui seul l'effet du déclencheur. Une étude complémentaire peut utiliser les
mêmes sorties pour les deux entrées.

Mesurer effectif, espérance et profit factor en R, excursions défavorables,
durées et résultats par paire. Comptabiliser aussi les positions encore ouvertes.
Réserver une période réellement inutilisée pour une validation chronologique.
Un replay court contrôle le fonctionnement ; une meilleure fréquence de gains
sur cet échantillon ne démontre pas un avantage durable.
