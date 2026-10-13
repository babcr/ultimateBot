# Trailing stop par paliers

Le mode s'active avec `--trailing-stop`. L'absence de cette option conserve la
sortie au take profit fixe. Dans le code, le booléen d'entrée est
`TradeCloser(trailing_stop_enabled=True)`.

Le gestionnaire de risque calcule trois niveaux : le SL initial, le seuil de
départ du trailing et le TP de référence. En mode trailing, le closer retire
la sortie fixe au TP avant d'envoyer l'ordre : `take_profit` devient `None`
sur l'ordre et la position. Le seuil de départ est transmis séparément dans
`trailing_stop_activation_price`, puis conservé au fill. La position peut
continuer au-delà du TP prévu et sort uniquement sur le stop.

`--trailing-start-rr-ratio` définit le départ du trailing en multiples de la
distance entre l'entrée prévue et le SL initial. Sa valeur doit être finie,
strictement positive et inférieure à `--reward-risk-ratio`. Par exemple,
`--reward-risk-ratio 3 --trailing-start-rr-ratio 1` démarre le trailing à 1R
avec un TP de référence à 3R, sans clôturer à 3R. En l'absence de valeur
explicite, le départ utilise la moitié
du RR du TP : un TP à 3R donne donc un départ à 1,5R.

Pour une distance de risque prévue `R0` et un ratio de départ `S` :

```text
achat : départ = entrée prévue + S × R0
vente : départ = entrée prévue - S × R0
```

Le seuil est inférieur au TP en distance de gain. Pour une vente, son prix
est donc supérieur au prix du TP, car le mouvement favorable est baissier.
Sans `--trailing-stop`, le seuil calculé reste inutilisé et la sortie demeure
au SL ou au TP fixes. Le mode peut aussi être activé pour les nouveaux ordres
par l'optimiseur adaptatif.

Le calcul de la cible reste celui du gestionnaire de risque : sa distance vaut
`RR × (distance du SL + spread)` pour les ordres limit/stop ; les ordres market
ajoutent encore un spread à cette distance. Le RR est borné à `0,1` minimum.
Ces marges restent propres au TP de référence. Le seuil de départ utilise la distance du
SL prévu, sans les ajouter. Changer le ratio de départ ne modifie ni le TP,
ni le SL initial, ni la taille de la position.

`TrailingStopManager`, dans `trailing_stop.py`, mémorise un `TrailingStopState`
pour chaque position. Le breakeven est le prix réel de fill, qui donne un PnL
nul avec la formule actuelle du backtesteur ; le modèle ne facture pas de
commission ni de financement. La distance initiale `R` est la valeur absolue
de la différence entre ce prix de fill et le SL initial, avec toutes les marges
calculées par le gestionnaire de risque. Cette distance reste constante.
Le seuil de départ reste celui prévu à la création de l'ordre, même si le
prix exécuté diffère de l'entrée prévue. Les paliers utilisent le risque réel
au fill. Le TP de référence intervient dans le calcul et la validation du
ratio de départ ; il ne borne pas le suivi.

Le prix utilisé est le **bid pour les achats** et l'**ask pour les ventes**,
comme pour les clôtures du backtesteur.

- Avant le seuil d'activation, la position est protégée par son SL initial.
- Au seuil, le stop passe au breakeven et la position reste ouverte.
- Après chaque déplacement favorable de `R` au-delà du seuil, le stop avance
  aussi de `R`. Le stop ne recule jamais.
- Si un tick franchit plusieurs paliers, ils sont tous pris en compte sur ce
  tick, sans interpoler un trajet de prix.
- Atteindre ou dépasser le TP prévu laisse la position ouverte et le stop
  continue d'avancer par paliers.

Pour une entrée acheteuse à 100, un SL initial à 98, un départ à 1R (102)
et un TP de référence à 5R (110), sans spread ni écart d'exécution :

| Bid atteint | Nouveau stop |
| --- | --- |
| Avant 102 | 98 |
| 102 | 100 |
| 104 | 102 |
| 106 | 104 |
| 108 | 106 |
| 110 | 108, position toujours ouverte |
| 112 | 110 |

Pour une vente, les mouvements sont symétriques sur l'ask. Le SL déjà en place
est contrôlé avant de déplacer le stop. Un gap qui franchit le TP prévu
avance les paliers et laisse la position ouverte. Les sorties sont exécutées
au bid/ask du tick disponible : un gap peut donner un prix différent de celui
du stop. En mode sans trailing, le TP fixe continue de fermer la position.

Le motif de clôture est `stop_loss` avant activation et `trailing_stop` après
activation. Les mises à jour produisent une ligne `[trailing-stop]` contenant
le seuil, le breakeven, la distance initiale, le nombre de paliers et le stop.
Un stop au breakeven ou en bénéfice ne réserve plus de capital au titre du
risque de perte dans le calcul de la marge libre.

Le closer utilise `prepare_order` à la création, `prepare_position` au fill et
`check` à chaque tick avec le moniteur de trades. `close` conserve son rôle de
comptabilisation d'une fermeture. Le moteur d'audit de volatilité utilise les
mêmes points d'entrée.

Un ordre fourni directement au closer doit renseigner le seuil d'activation
séparé. Une position fournie directement, sans évaluation de risque ni seuil,
utilise par défaut le milieu entre son prix d'entrée et son TP. Les états déjà
initialisés et les ordres en attente conservent leur seuil et leur mode lors
des changements de configuration adaptative.

Exemple depuis `C:\git` :

```powershell
c:\git\.venv\Scripts\python.exe -m ultimateBot.run_backtest --strategy swing-regression --symbols USDCHF --start-date 2020-01-01 --end-date 2020-06-30 --candle-lookback 150 --reward-risk-ratio 3 --trailing-stop --trailing-start-rr-ratio 1
```
