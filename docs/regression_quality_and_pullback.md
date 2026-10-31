# Hybride sélectif et entrées complémentaires expérimentales

Les résultats de l’audit 2020–2025 sont des hypothèses de recherche. Deux
filtres sont disponibles explicitement ; le mode historique reste accessible
avec ses paramètres d’origine. Les compléments optionnels explorent les
replis directionnels, les rebonds intérieurs et les cassures après compression.

## Lancement du socle sélectif

Depuis le dossier parent de `ultimateBot` :

```powershell
py -m ultimateBot.run_regression_hybrid
```

Ce lanceur reprend les quatre paires, 2020–2025, bougies de 60 minutes,
régression sur 25 bougies, volatilité du risque sur 150 bougies, risque de
1 %, RR de référence 2,3, breakout `confirmed` et trailing à 1R. Il active
la sélection des réintégrations. Le filtre de coût reste optionnel : dans
le rejeu 2024–2025, son ajout réduit les trades de 28 à 13 et la somme des
résultats de +14,18R à +9,35R, tout en augmentant le PF en R de 2,06 à 2,82.
Le socle privilégie donc la fréquence parmi ces deux variantes. Il n’active
aucune branche complémentaire.
Les arguments explicites remplacent les valeurs du lanceur.

Pour compléter la commande `run_backtest` existante :

```powershell
--regression-reentry-quality-mode selective --regression-max-spread-stop-ratio 0.10
```

Pour la sélection seule, omettre le second argument ou le régler à zéro.
Avec le nouveau lanceur, ajouter explicitement le contrôle de coût ainsi :

```powershell
py -m ultimateBot.run_regression_hybrid --regression-max-spread-stop-ratio 0.10
```

`--regression-reentry-quality-mode legacy` et
`--regression-max-spread-stop-ratio 0` désactivent respectivement ces filtres.
Ces deux valeurs restent les défauts du lanceur générique, pour conserver
les comparaisons avec les anciens runs.

## Filtre des réintégrations

En mode `selective`, une confirmation failed-breakout est sélectionnée si :

- elle survient sur la première bougie suivant celle de l’excursion ;
- ou elle survient sur la bougie de l’excursion, et sa mèche côté stop
  représente au plus 25 % de son amplitude.

Les confirmations après deux ou trois bougies sont écartées. La proportion
est réglable avec `--regression-reentry-max-same-bar-stopside-wick`.
Pour un achat, mesurer la mèche basse ; pour une vente, la mèche haute.
La mesure utilise le vrai OHLC mid clôturé, sans reconstruire ses extrêmes
en moyennant les extrêmes bid et ask.

Ce filtre s’applique après confirmation et consommation de l’événement.
Il ne modifie ni le délai maximal du détecteur, ni l’horizon de contrôle
des droites. En hybride, une confirmation rejetée par ce filtre consomme
aussi les autres branches : aucune entrée de secours ne remplace ce signal.
Le stop de l’excursion et l’objectif fixe au centre sont conservés.

## Filtre du coût d’entrée

Après l’évaluation du risque et le calcul du prix prévu et du stop final,
le moteur contrôle :

`(ask courant - bid courant) / abs(prix prévu - stop final) <= seuil`.

La cotation est celle déjà connue à la soumission, après la clôture qui a
produit le signal. Une ouverture de nouvelle bougie ou un gap peut changer
son spread par rapport au dernier tick de la bougie du signal. Le filtre
prend cette différence en compte. Il n’utilise pas le prix de remplissage
futur. Il rejette aussi une cotation croisée ou un ratio non fini.

Ce contrôle est distinct de `--max-spread-ratio`, dont le dénominateur
provient du modèle de volatilité habituel. Un refus intervient avant la
création de l’ordre. Les ordres acceptés enregistrent le spread, le ratio,
le seuil et le prix prévu dans leur contexte.

## Branche expérimentale trend-pullback

```powershell
py -m ultimateBot.run_regression_hybrid --regression-pullback-enabled
```

Elle est disponible uniquement dans le mode `hybrid`. Ses paramètres se
trouvent dans `SwingRegressionConfig.pullback` et dans l’aide CLI sous
`--regression-pullback-*`. Les valeurs initiales sont fixées avant les essais,
sans recherche du meilleur paramétrage historique.

1. Ajuster le canal sur les seules bougies précédant le contact. Appliquer
   les contrôles communs de qualité. Exiger deux droites de même sens, un
   parallélisme d’au moins 0,7 et une largeur d’au moins 1,5 ATR.
2. Exiger un déplacement de la droite la moins pentue, sur la fenêtre de
   régression, représentant au moins 0,6 fois la largeur maximale du canal.
   Ce régime directionnel complète les réintégrations, limitées par défaut
   à une dérive de 0,5 pour la droite la plus pentue et la largeur minimale.
3. En hausse, attendre un contact bid avec le bord bas ; en baisse, un
   contact ask avec le bord haut. La clôture précédente doit être éloignée
   de ce bord d’au moins la tolérance de contact. Le contact doit clôturer
   dans le canal, ne pas traverser le bord opposé et ne pas dépasser le bord
   de plus de 0,5 ATR par sa mèche. Figer le canal et son ATR.
4. Une bougie **ultérieure**, au plus trois bougies après le contact, doit
   clôturer au-delà du high bid initial pour un achat, du low ask initial
   pour une vente, avec une marge de 0,05 ATR. Son corps dans le sens du trade
   doit occuper au moins 35 % de l’amplitude ; sa clôture doit se situer dans
   les 35 % favorables. Le premier contact ne suffit pas à entrer.
5. Limiter le prix d’entrée à 60 % de la largeur depuis le bord de départ.
   Exiger au moins 1R d’espace jusqu’au bord opposé au prix de confirmation.
   Poser le stop au-delà de l’extrême de tout le repli, avec 0,2 ATR de marge.
6. Utiliser le trailing existant si activé, sinon le TP selon le RR de
   référence. La branche ne reçoit pas le TP au centre réservé aux réintégrations.

Un trou dans l’historique, une pénétration excessive, une sortie du canal,
le contact du bord opposé ou l’expiration invalident le setup. Un nouvel
ajustement ne remplace pas le canal d’un événement en attente.

### Autre candidat : channel-bounce

```powershell
py -m ultimateBot.run_regression_hybrid --regression-pullback-enabled --regression-pullback-regime range
```

Ce régime remplace la branche expérimentale directionnelle par un rebond
intérieur de canal, nommé `channel-bounce` dans les résultats. Il demande une
dérive maximale de 0,5 pour la droite la plus pentue sur la largeur minimale,
sans exiger deux pentes de même sens. Le bord bas prépare un achat ; le bord
haut prépare une vente. Les mêmes contrôles de reprise sur une bougie
ultérieure, de corps, d’espace disponible et de stop restent actifs.

Une excursion extérieure d’au moins 0,1 ATR invalide ce candidat, afin de
réserver cette situation au failed-breakout. Les paramètres
`--regression-pullback-range-max-drift-ratio` et
`--regression-pullback-range-max-excursion-atr` règlent ces limites. Il utilise
le trailing si activé, sinon le TP de référence ; il n’hérite pas de la cible
au centre des réintégrations. Les deux régimes complémentaires se comparent
séparément : un seul est activé dans chaque variante.

Les deux branches d’origine ont priorité lorsqu’elles confirment au même
instant que le pullback. Le pullback ne déclenche pas un rejet d’une entrée
primaire opposée. Toutes les branches sont consommées lors d’une décision.
Les conflits entre les deux branches d’origine gardent leur règle de rejet.

Le portefeuille et le budget de risque sont communs. Une seule position ou
un seul ordre en attente est autorisé par paire. Une position pullback peut
donc empêcher une entrée primaire plus tard : le résultat combiné n’est pas
la somme de backtests indépendants.

## Troisième candidat : breakout-hold après compression

```powershell
py -m ultimateBot.run_regression_hybrid --regression-impulse-enabled
```

La branche `breakout-hold` recherche une continuation sans attendre le
retest. Elle exige une volatilité ATR courte pré-cassure au plus égale à
0,85 fois l’ATR de régression et une largeur de canal d’au plus 4 ATR.
Une première clôture doit sortir du canal de 0,2 à 0,75 ATR, dans le sens
de la pente commune, avec un corps directionnel d’au moins 70 % et une
clôture dans les 20 % favorables. La clôture précédente doit être dans le
canal.

La bougie suivante doit rester entièrement hors du bord d’au moins 0,1 ATR,
progresser d’au moins 0,05 ATR depuis la clôture de cassure et présenter un
corps directionnel d’au moins 35 %, avec une clôture dans les 35 % favorables.
L’entrée est limitée à 1 ATR du bord ; le stop est placé 0,2 ATR à l’intérieur
du canal figé. Le trailing habituel s’applique si activé. Un retour au bord,
une reprise insuffisante ou un trou de cotation consomment ce setup : aucune
confirmation retardée n’est possible.

Ces paramètres sont regroupés dans `config.impulse`, sous les options
`--regression-impulse-*`. La branche est désactivée par défaut et garde la
priorité des deux entrées primaires. Si deux compléments confirment en sens
opposés sans confirmation primaire, leurs signaux sont rejetés.

Le candidat peut être comparé séparément, sans modifier les premiers essais :

```powershell
py -m ultimateBot.research_regression_complement --lazy-quotes --variants quality quality-impulse --output ultimateBot/artifacts/regression_impulse_research
py -m ultimateBot.report_regression_complement --directory ultimateBot/artifacts/regression_impulse_research
```

## Comparaison et diagnostic

Le bilan hybride sépare `failed-breakout`, `breakout-retest` et, lorsqu’elle
est activée, `trend-pullback` ou `channel-bounce`. Il affiche les confirmations, les signaux
sélectionnés avant les filtres de soumission, les trades exécutés, le PnL,
les refus de qualité et ceux de spread/stop. Les journaux de création et
de remplissage portent explicitement `entry_mode` pour distinguer les
branches à trailing.

L’outil suivant compare le mode d’origine, le socle sélectif et son ajout
directionnel ou son rebond intérieur, avec les mêmes données et un capital
partagé pour chaque variante :

```powershell
py -m ultimateBot.research_regression_complement --start-date 2024-01-01 --end-date 2025-12-31 --lazy-quotes
py -m ultimateBot.report_regression_complement
```

Les décisions utilisent toutes les clôtures horaires. Entre décisions, le
rejeu conserve les ticks pouvant remplir ou expirer un ordre, toucher un
stop/objectif ou modifier le trailing. Les extrema des ticks sautés sont
enregistrés pour les excursions, et l’historique réel des bougies est restauré
avant chaque décision. Les fills, sorties et règles de risque restent ceux
du moteur existant. Les dernières confirmations sans tick d’exécution après
la fin du flux sont omises du comparateur ; aucune position exécutée ne
peut en résulter.

Cet accélérateur est réservé aux paramètres fixes, sans session/optimiseur/
validateur/dataset et avec assez de places pour les paires sélectionnées.
Le moteur standard reste la référence. Les tests comparent les fills,
clôtures et excursions à un rejeu intégral. Les résultats de recherche sont
conservés dans `artifacts/regression_opportunities/`.

L’option `--lazy-quotes` utilise les prix exacts du premier tick déjà stockés
dans les métadonnées lorsque toutes les variantes sont sans exposition sur
une paire. Dès qu’un ordre est soumis, les ticks bruts sont chargés pour son
exécution. Les tailles de cotation au premier tick synthétique sont ignorées :
elles ne participent à aucun calcul des modèles de prix autorisés ici.
Les tests couvrent cette option ainsi que le parcours complet des ticks.

Le comparateur distingue également `quality` (sélection des réintégrations
seule), `spread` (coût seul), `filtered` (les deux) et les ajouts
`quality-pullback` / `quality-bounce` au socle de qualité. Cela mesure l’effet
de chaque filtre avant de décider de les cumuler.
