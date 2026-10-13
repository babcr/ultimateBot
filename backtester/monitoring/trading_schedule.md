# Filtre de séances et clôture du vendredi

Les deux options sont indépendantes et désactivées par défaut. Ajouter à la
commande habituelle :

```powershell
--session-filter --close-friday
```

Exemple depuis `C:\git` :

```powershell
.\.venv\Scripts\python.exe -m ultimateBot.run_backtest --strategy swing-regression --symbols EURUSD --start-date 2020-01-03 --end-date 2020-01-03 --session-filter --close-friday
```

## Premiers trois quarts des séances

`--session-filter` accepte les nouvelles entrées pendant les premiers 75 %
d'au moins une des deux séances de référence, du lundi au vendredi :

| Séance | Horaires complets locaux | Fenêtre d'entrée locale par défaut |
| --- | --- | --- |
| Londres, `Europe/London` | 08:00–16:30 | 08:00 inclus à 14:22:30 exclu |
| New York, `America/New_York` | 09:30–16:00 | 09:30 inclus à 14:22:30 exclu |

Les horaires de référence proviennent du
[London Stock Exchange](https://www.londonstockexchange.com/personal-investing/faqs?lang=en)
et de la [séance principale NYSE](https://www.nyse.com/trade/trading-information).
Il s'agit d'un filtre d'horaires hebdomadaires : les jours fériés, fermetures
exceptionnelles et séances écourtées des bourses ne sont pas importés dans ce
calendrier. Il ne détermine pas la disponibilité effective des quotes Forex.

Les changements d'heure sont calculés avec `zoneinfo`, séparément pour les deux
zones. En UTC, en hiver : Londres 08:00–14:22:30 et New York 14:30–19:22:30 ;
en été : Londres 07:00–13:22:30 et New York 13:30–18:22:30. Les semaines où les
États-Unis et le Royaume-Uni changent d'heure à des dates différentes ont leurs
propres fenêtres. La fraction exacte conserve la demi-minute finale.

`--session-entry-fraction 0.75` modifie cette fraction, entre 0 exclu et 1
inclus. `--no-session-filter` désactive le filtre. Un horodatage sans fuseau est
interprété en UTC, conformément au moteur de chandelles.

Le filtre s'applique à l'horodatage du signal et à l'heure de traitement. Un
lot retardé ou le dernier lot du backtest ne crée donc pas une entrée tardive.
Les ordres market, limit et stop sont annulés lorsqu'ils dépassent la fin de
leur fenêtre autorisée, y compris si le premier tick suivant arrive dans une
autre séance ou le lendemain. Ils peuvent traverser un chevauchement entre
séances, mais pas une interruption des fenêtres. Les positions déjà remplies
gardent leurs stops/TP/trailing et peuvent sortir en dehors des heures d'entrée.

## Vendredi à 23h59 UTC

`--close-friday` ferme les positions à partir du vendredi 23:59:00 UTC avec le
motif `END_OF_WEEK`. Les achats sortent au bid, les ventes à l'ask ; le moteur
conserve ses coûts et le PnL habituels. La règle s'applique également aux
positions en trailing. Les ordres en attente sont annulés et les nouvelles
entrées sont bloquées jusqu'au lundi 00:00 UTC.

Le backtester avance sur les ticks, pas sur une horloge murale. Chaque symbole
est fermé sur sa première cotation à l'heure prévue ou après. Les autres
symboles attendent leur propre cotation ; aucun prix d'un autre symbole ou
prix futur n'est réutilisé. Si les quotes s'arrêtent avant 23h59 puis reprennent
le dimanche/lundi, la fermeture attend cette reprise. Si le backtest se termine
avant toute cotation après l'échéance, les positions restent au bilan ouvertes.
Cette option ne peut donc pas garantir une sortie avant un week-end sur un
marché qui ne cote plus à 23h59 UTC. L'heure est configurable pour s'adapter
aux données et au marché :

```powershell
--close-friday --friday-close-time 21:55 --friday-close-timezone UTC
```

`--friday-close-time` accepte `HH:MM` ou `HH:MM:SS` ; le défaut est `23:59`.
`--friday-close-timezone` accepte une zone IANA ; le défaut est `UTC`, comme
demandé. `--no-close-friday` désactive la règle. Si le fuseau est modifié, le
vendredi, l'échéance et le blocage jusqu'au lundi sont évalués dans ce fuseau.

Une position passée au travers d'une interruption de quotes conserve sa
fermeture due après restauration du checkpoint. Une nouvelle position ouverte
le lundi ne reprend pas l'échéance du vendredi précédent.

## Optimiseurs, rapports et événements broker

Les règles sont communes aux stratégies structure, swing-regression et dummy,
à l'ancien optimiseur adaptatif et aux variantes shadow du bandit. Les comptes
shadow ont leurs propres compteurs, conservent leurs positions jusqu'aux sorties
normales et suivent la même clôture hebdomadaire.

Quand elles sont activées, les règles effectives entrent dans l'identité du
modèle LinUCB. Le risque, les domaines appris et les autres paramètres restent
ceux déjà configurés. Modifier une règle activée sépare les mémoires ; les
options désactivées conservent les identités existantes. Une clôture
`END_OF_WEEK` est une sortie normale et peut entraîner le bandit, selon les
conditions habituelles de risque et de contexte. Les annulations ne produisent
aucune récompense ni aucun trade perdant fictif.

La CLI et les rapports JSON affichent les règles et les compteurs : signaux
rejetés, ordres en attente annulés, positions clôturées pour la fin de semaine.
Ces compteurs et l'horloge du moteur sont sauvegardés dans les checkpoints.

Pour un hôte utilisant `BrokerBanditBridge`, `entry_allowed(emitted_at,
timestamp=now)` expose le même filtre. `weekly_close_requests(now)` retourne
les identifiants des positions broker encore ouvertes à clôturer, avec le motif
`END_OF_WEEK`, et peut être interrogé par un timer à 23h59. L'hôte doit envoyer
les demandes avec son transport broker et traiter les deals confirmés. Cette
requête ne génère ni fill virtuel principal ni résultat inventé ; les snapshots
clôturés ou absents après réconciliation ne sont pas redemandés.

Les tests de `tests/test_trading_schedule.py` couvrent les frontières à la
microseconde, les calendriers d'heure d'été/hiver différents, les trous entre
séances, les ordres différés, les sorties bid/ask sur plusieurs symboles,
le trailing, la reprise, la compatibilité des identités et les requêtes broker.
