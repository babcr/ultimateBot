# Cassure confirmée et retest du canal

Le mode `breakout-retest` complète `failed-breakout` : il cherche une continuation
après une cassure confirmée et un retest. Ses paramètres initiaux sont des
hypothèses à évaluer, sans avantage statistique établi.

La confirmation expérimentale `confirmed` ajoute des contrôles de qualité à la cassure et
attend une reprise de l'extrême d'une bougie de retest déjà clôturée. L'ancienne
règle `legacy` reste le défaut : le premier essai renforcé n'a produit aucun
trade sur les deux périodes de contrôle EURUSD. Cela ne démontre pas un
avantage ; le candidat n'est donc pas promu automatiquement.

L'étude de fréquence sur EURUSD 2024, avec le TP fixe de 2,3R de la commande
utilisateur, est décrite dans [regression_breakout_frequency.md](regression_breakout_frequency.md).
Elle ajoute un délai de confirmation séparé, optionnel ; les premiers essais
augmentent le nombre de trades mais dégradent les résultats, sans changer le défaut.

## Lancement

Depuis le dossier parent de `ultimateBot` :

```powershell
py -m ultimateBot.run_regression_breakout
```

Pour tester explicitement la confirmation renforcée :

```powershell
py -m ultimateBot.run_regression_breakout --regression-breakout-confirmation-mode confirmed
```

Ce lanceur reprend les conditions de départ de la campagne reentry : USDCHF,
GBPUSD, USDCAD, EURUSD ; 2020–2023 ; bougies de 10 minutes ; régression sur
15 bougies ; volatilité du risque sur 150 bougies ; risque de 2,2 % ; filtre
de session actif. Il active le trailing à 1,5R avec un TP de référence à 3R.
L'adaptation et le bandit sont désactivés. Les arguments explicites remplacent
ces valeurs, par exemple :

```powershell
py -m ultimateBot.run_regression_breakout --symbols EURUSD --start-date 2024-01-01 --end-date 2024-01-31 --risk-percent 0.1
```

Le lanceur commun permet une comparaison avec TP fixe, sans `--trailing-stop` :

```powershell
py -m ultimateBot.run_backtest --strategy swing-regression --regression-entry-mode breakout-retest --symbols EURUSD --start-date 2024-01-01 --end-date 2024-01-31 --candle-interval 10m --regression-lookback-bars 15 --candle-lookback 150 --risk-percent 0.1 --reward-risk-ratio 3 --session-filter
```

Ajouter `--trailing-stop --trailing-start-rr-ratio 1.5` à cette même commande
pour comparer les sorties en conservant les autres paramètres.

## Règles causales

1. Ajuster le canal sur les bougies clôturées **précédant** la cassure. Les
   filtres communs restent actifs : volatilité minimale du marché, nombre et
   densité des swings, paires de swings trop proches, parallélisme et erreur
   des droites en ATR. Les deux droites doivent être présentes et ne pas se
   croiser dans la fenêtre ni jusqu'à la fin du délai de retest.
2. La clôture précédente doit être dans le canal (bid sous la borne haute,
   ask au-dessus de la borne basse). Une clôture bid au-dessus de la borne
   haute d'au moins `min_close_atr × ATR` arme un achat ; une clôture ask sous
   la borne basse arme une vente. Un dépassement par une mèche seule ne suffit
   pas. Une bougie dépassant les deux bornes est ambiguë et rejetée.
   En confirmation `confirmed`, le prix doit aussi progresser dans le sens
   de la cassure depuis sa clôture précédente. Le corps directionnel doit
   représenter au moins 50 % de l'amplitude, avec une clôture dans les 30 %
   favorables de la bougie. Un déplacement de la droite sans progression
   du prix, une bougie de la mauvaise couleur ou un petit corps sont refusés.
3. Figer les coefficients et l'ATR pré-cassure. Projeter les droites jusqu'aux
   bougies suivantes, sans réajuster le canal pendant l'événement. Le score
   de pente locale reste un diagnostic : une consolidation horizontale peut
   produire une entrée. Les confirmations historiques de tendance globale,
   de réaction aux niveaux et de forme de bougie sont désactivées. La
   confirmation propre au retest, décrite ci-dessous, reste obligatoire.
4. Attendre un retest entre t+1 et t+`max_bars`, inclus. La bougie de cassure
   ne peut pas confirmer son propre retest. Pour acheter, l'amplitude bid
   doit rencontrer la zone `borne haute ± retest_tolerance_atr × ATR`.
   Pour vendre, utiliser l'amplitude ask autour de la borne basse.
5. Dès le premier contact, mémoriser le plus bas bid et le plus haut ask du
   retest jusqu'à sa confirmation. En mode `confirmed`, mémoriser aussi
   l'extrême de la dernière bougie de contact : son high bid pour un achat,
   son low ask pour une vente. Une **bougie ultérieure** doit clôturer au-delà
   de cet extrême et de la borne projetée, avec une marge de 0,05 ATR. Son
   corps directionnel doit représenter au moins 35 % de son amplitude et sa
   clôture se situer dans les 30 % favorables. Les mesures utilisent le bid
   pour un achat, l'ask pour une vente. Le premier contact ne suffit jamais
   à entrer sur cette même bougie. Si une nouvelle bougie retouche la zone,
   son extrême devient la référence pour les bougies suivantes ; on compare
   toujours la clôture à la référence antérieure, sans utiliser son propre
   high/low ni des données futures. Par défaut, le délai initial reste inclusif
   et commun au retest et à sa confirmation.
   Avec `--regression-breakout-confirmation-max-bars C` strictement positif,
   `max_bars` devient le délai du premier contact. La reprise dispose ensuite
   de C bougies supplémentaires à partir de ce **premier** contact, avec une
   borne inclusive. Un nouveau contact ne repousse pas cette échéance. Les
   contrôles de profondeur et de clôture s'appliquent pendant toute l'attente ;
   la validité des droites est contrôlée sur l'horizon prolongé.
   En mode `legacy`, une bougie de la bonne couleur clôturant hors du canal
   suffit, y compris sur la première bougie de contact.
6. Invalider si une clôture rentre dans le canal de plus de
   `invalidation_atr × ATR`, si la borne opposée est touchée, si une bougie
   manque ou si le délai est dépassé. Une petite rentrée tolérée peut donc
   être suivie d'une confirmation dans le délai initial.
   En mode `confirmed`, une mèche dépassant de plus de 0,5 ATR à l'intérieur
   invalide aussi l'événement, même si la clôture récupère ensuite la borne.
7. Consommer la confirmation une seule fois, même si la session, le risque ou
   le valideur refuse ensuite l'ordre. Refuser une clôture exécutable déjà
   trop éloignée de la borne (`max_entry_distance_atr × ATR`). Le signal
   transmet aussi la zone d'entrée au risque, qui la revérifie sur le prix
   du tick fourni et sur le prix d'ordre planifié, spread inclus.
   En mode `confirmed`, la zone autorisée commence au seuil de reprise :
   un retour sous ce seuil pour un achat, ou au-dessus pour une vente,
   refuse la création de l'ordre même si le prix est encore hors du canal.

Le stop est le plus bas bid du retest moins `stop_buffer_atr × ATR` pour
un achat ; le plus haut ask plus cette marge pour une vente. L'extrême de
la bougie de cassure ne participe pas à ce stop. Les règles de l'instrument
peuvent élargir la distance minimale du stop avant le calcul de quantité.

L'exécution conserve les règles du moteur : délai, expiration et éventuel
passage en limit si le spread l'exige. Aucun fill n'est simulé sur la clôture
qui produit le signal. Les bornes d'entrée sont contrôlées à la création de
l'ordre ; elles ne garantissent pas le prix de fill ultérieur en présence
d'un gap. Le replay modélise le spread et les écarts d'exécution, sans ajouter
de commission ni de financement.

## Risque et trailing

Le signal fournit un stop explicite sans cible fixe. Le risque dimensionne la
quantité sur le stop final, puis recalcule le TP de référence avec le RR et les
marges de spread habituelles. Le seuil de trailing est calculé sur cette même
distance de stop. En mode trailing, le closer retire le TP fixe de l'ordre.

Avec le lanceur dédié : activation à 1,5R prévu, stop au prix de fill, puis
progression de 1R par palier avec le risque initial réel au fill. Les positions
peuvent dépasser le TP de référence. Les clôtures de calendrier restent
applicables. Voir [trailing_stop.md](../backtester/monitoring/trailing_stop.md)
pour les conventions exactes de prix et de gaps.

## Paramètres dédiés

La configuration Python est `SwingRegressionConfig.breakout`, définie dans
`backtester/strategy/regression_breakout_config.py`. Les options et snapshots
du run reprennent ces champs :

| Option | Défaut | Unité |
| --- | ---: | --- |
| `--regression-breakout-confirmation-mode` | `legacy` | `confirmed` expérimental ou `legacy` |
| `--regression-breakout-max-bars` | 3 | Délai commun ou délai du premier contact ; minimum 2 en `confirmed` à délai commun, sinon 1 |
| `--regression-breakout-confirmation-max-bars` | 0 | 0 : délai commun ; positif : délai après le premier contact, en `confirmed` |
| `--regression-breakout-min-close-atr` | 0.2 | ATR pré-cassure |
| `--regression-breakout-retest-tolerance-atr` | 0.2 | ATR pré-cassure |
| `--regression-breakout-invalidation-atr` | 0.2 | ATR pré-cassure |
| `--regression-breakout-max-entry-distance-atr` | 0.75 | ATR pré-cassure |
| `--regression-breakout-stop-buffer-atr` | 0.2 | ATR pré-cassure |
| `--regression-breakout-min-breakout-body-ratio` | 0.5 | Corps directionnel/amplitude de la cassure |
| `--regression-breakout-min-confirmation-body-ratio` | 0.35 | Corps directionnel/amplitude de la reprise |
| `--regression-breakout-min-close-location` | 0.7 | Position favorable de la clôture, de 0.5 à 1 |
| `--regression-breakout-max-retest-depth-atr` | 0.5 | Pénétration maximale de la mèche du retest |
| `--regression-breakout-confirmation-buffer-atr` | 0.05 | Marge au-delà de l'extrême antérieur et de la borne |

Les deux ratios de corps doivent appartenir à [0, 1] ; la position de clôture
à [0.5, 1]. Les distances doivent être finies et positives ou nulles. Ces cinq
contrôles supplémentaires sont ignorés en `legacy` ; la validation de leur
configuration reste active. Les paramètres `regression_reentry_*`, le stop générique
de structure et les seuils directionnels du mode `trend` ne règlent pas cette
entrée. Le lanceur refuse le bandit et l'adaptation pour ce premier protocole
fixe ; le trailing est autorisé. Les autres modes gardent leurs politiques.

Les diagnostics comptent les confirmations et les attentes/refus par motif.
Avec `--trade-database`, le circuit existant persiste le contexte sous
`entry_mode="breakout-retest"` : dates, coefficients, bornes projetées, ATR,
distance de cassure, délai, extrêmes du retest, stop et zone d'entrée.
Le contexte ajoute le mode de confirmation, les ratios de corps, positions
de clôture, l'extrême précédent et le niveau exact de reprise. Le journal
inclut aussi `first_retest_bar`, `confirmation_bars` et l'échéance de confirmation.
Le journal
`--regression-live-analysis` expose le seuil et les motifs de refus : corps
ou clôture faibles, absence d'impulsion, retest trop profond, reprise en attente.
Les champs du valideur XGBoost ne changent pas. Évaluer cette population
séparément avant de réutiliser un valideur d'une autre campagne.

## Validation statistique

Comparer les sorties fixes et trailing de ce mode sur les mêmes ticks,
sessions, fenêtres et risque. Comparer ensuite reentry seule, breakout-retest
seul et leur combinaison avec une enveloppe de risque comparable. Cette
première implémentation sélectionne un seul mode par run ; elle ne combine
pas automatiquement leurs ordres.

Mesurer effectif, espérance en R, profit factor, drawdown, excursions et
positions encore ouvertes, par paire et par période. Le taux de réussite
seul ne suffit pas. Réserver une période chronologique inutilisée pour la
validation ; les tests automatisés vérifient le fonctionnement, pas la
rentabilité.

Le comparateur exécute les deux confirmations avec le même historique, les
mêmes filtres de canal, sessions, stop, risque de 0,1 %, RR de référence et
trailing à 1,5R, sans recherche de paramètres :

```powershell
py -m ultimateBot.compare_regression_breakout --symbols EURUSD
```

Par défaut : comparaison du 2 au 12 janvier 2024, puis contrôle séparé du
15 au 26 janvier 2024. Les dates sont modifiables par `--start-date`,
`--end-date`, `--validation-start`, `--validation-end` ; les périodes doivent
être disjointes et chronologiques. `--symbols` accepte plusieurs paires.
Le JSON et les journaux se trouvent dans
`artifacts/regression_breakout_detection_v2/comparison` ou dans `--output`.
Le PnL en R utilise le risque initial réel au fill de chaque position. Le
drawdown rapporté est celui des R réalisés ; il ne mesure pas les fluctuations
des positions ouvertes. Le rapport compte séparément positions et ordres
encore ouverts. Une absence de trades donne une espérance indéfinie, jamais
une preuve d'avantage.

## Vérifications de l'implémentation

La révision passe 120 tests ciblés, dont 40 couvrent le breakout et son rapport
de comparaison. Les 17 nouveaux cas vérifient notamment le corps, la position
de clôture, la pénétration des mèches, l'extrême strictement antérieur,
la symétrie bid/ask, le refus d'une entrée revenue sous le seuil, la causalité
du pipeline sur trois clôtures et le calcul des résultats en R.

Comparaison EURUSD avec les seuils ci-dessus fixés avant le replay, mêmes
sessions, risque 0,1 % et trailing 1,5R pour les deux variantes :

| Période | Legacy : trades / PnL R | Confirmed : trades / PnL R |
| --- | ---: | ---: |
| 2–12 janvier 2024 | 2 / -2.017R | 0 / 0R |
| 15–26 janvier 2024, contrôle séparé | 3 / -3.043R | 0 / 0R |

Les cinq trades legacy sont perdants ; aucun trailing ne s'active. Les quatre
runs terminent sans position ni ordre ouverts. Le candidat filtre tous les
trades, donc son espérance et son profit factor sont indéfinis. **Ces résultats
ne prouvent pas une amélioration de la qualité des entrées.** Le mode renforcé
reste une option d'étude et les seuils n'ont pas été recherchés à partir de
ces résultats. Rapport détaillé :
`artifacts/regression_breakout_detection_v2/comparison/comparison.json`.

Historique des vérifications de la première version :

- La version initiale passait 103 tests ciblés, dont 23 dédiés : symétrie
  achat/vente, causalité de l'ajustement, projections figées, délai inclusif,
  invalidations, consommation unique, bornes d'entrée, contraintes instrument,
  quantité, TP de référence et cycle ordre/fill/trailing.
- La suite globale présente 25 échecs et 10 erreurs. Une reconstruction de
  l'état antérieur à cet ajout dans un dossier temporaire reproduit ces mêmes
  échecs (347 tests antérieurs). Le journal est dans
  `artifacts/regression_breakout_smoke/previous_state_tests.log`.
- Le replay EURUSD du 2 au 5 janvier 2024, avec risque de 0,1 %, termine avec
  549 fenêtres analysées, deux confirmations, deux ordres exécutés et deux
  pertes au stop initial, pour un rendement de -0,20 %. Aucun trailing ne
  s'active dans ce petit échantillon. Ce résultat contrôle le fonctionnement
  du circuit sur les ticks historiques ; il ne valide pas la qualité de
  l'entrée. Journal :
  `artifacts/regression_breakout_smoke/eurusd_2024-01-02_2024-01-05.log`.
