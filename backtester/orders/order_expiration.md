# Expiration des ordres

`OrderBuilder` accepte `expiration_bars: int`, avec une valeur par défaut de **2**,
définie dans `OrderConfig` (`backtester/config/backtest_config.py`).
Ce multiplicateur utilise la timeframe de base du moteur, transmise via
`candle_interval`. Avec une timeframe de 15 minutes et `expiration_bars=2`,
la validité d'un ordre est de 30 minutes.
Pour chaque ordre MARKET, LIMIT ou STOP, il fixe
`expires_at = created_at + expiration_bars * candle_interval`.

```python
from datetime import timedelta
from ultimateBot.backtester.orders.order_builder import OrderBuilder

builder = OrderBuilder(expiration_bars=2, candle_interval=timedelta(minutes=15))
```

Le backtest expose ce multiplicateur entier :

```powershell
c:\git\.venv\Scripts\python.exe -m ultimateBot.run_backtest --strategy swing-regression --symbols EURUSD --start-date 2020-01-01 --end-date 2020-12-31 --candle-lookback 150 --level-lookback-bars 15 --reward-risk-ratio 3. --risk-percent 1.0 --trailing-stop --order-expiration-bars 2
```

Le multiplicateur doit être un entier strictement positif. La durée résultante
doit être supérieure au délai d'exécution (50 ms par défaut). Elle utilise
le temps simulé des ticks, indépendamment de la vitesse réelle du run,
du lookback choisi par l'optimiseur et de l'unité de temps de confirmation globale.

Au premier tick du symbole dont l'horodatage est supérieur ou égal à
`expires_at`, un ordre encore en attente est retiré avant toute tentative
de fill. Le log `[order-expired]` indique l'ordre, sa création, son échéance
et le tick qui constate l'expiration.

Un ordre en attente ne bloque pas les nouvelles opportunités sur son symbole :
plusieurs ordres peuvent coexister. La limite `max_positions` reste appliquée aux
positions réellement ouvertes, à la création des ordres et lors de leurs fills.
Les ordres qui ne disposent pas d'un emplacement restent en attente jusqu'à
leur exécution ou leur expiration.

Une fois l'ordre exécuté, la position conserve sa gestion par SL/TP ou trailing
stop : cette durée de validité ne ferme pas les positions ouvertes.
Le bilan final affiche aussi le nombre d'ordres encore en attente.
