# Validation locale — 13 octobre 2026

233 tests passent avec `C:\git\.venv\Scripts\python.exe -m unittest discover
-s ultimateBot/tests -q` (14,83 s). Les 206 tests existants sont conservés ;
27 tests supplémentaires couvrent l'optimiseur, la persistance et l'intégration.
La compilation Python passe également.

Les tests vérifient l'identité effective et son partage entre quatre modes,
les grilles et les bras équivalents après arrondi, le risque et sa conversion,
le clipping, les volumes absents et les chandelles actives exclues, la
normalisation figée et le replay du warmup, l'apprentissage contextuel
déterministe, l'attribution au contexte d'origine, les partiels et les lots
simultanés, le budget shadow et les copies exécutées, la reprise après
interruption, le préfixe de données modifié, les écritures concurrentes depuis
deux processus, la lecture seule, l'import/export/reset ciblé, la corruption,
les verrous et les versions incompatibles. Les moteurs shadow possèdent leurs
stratégies et comptes ; le client d'exécution espion principal ne reçoit aucun
appel. Un marché inactif ne produit pas de récompense fictive. Le bandit
désactivé conserve le replay natif sans accès au store.

## Historique local et rechargement

Les quatre runs utilisent 3 480 ticks EURUSD provenant de
`D:\jforex_history\EURUSD\2020\01\2020-01-02.tsv`, entre
00:00:01.022 et 02:00:00.938 UTC. Empreinte commune :
`5198466c4c6913c3d45b5252ff9aec3237cef5624b628152639f88afd5b17723`.
Le validateur conserve les ticks complets, sans sous-échantillonnage.

Ces runs de vérification emploient des paramètres explicites : chandelles 1 m,
R initial 10, supports/résistances 30, confirmation globale désactivée,
filtre de réaction désactivé, seuils de régression relâchés, TP 0,5R,
activation trailing 0,1R, contexte 10, warmup 2, six slots shadow, groupes
bornés à 30 chandelles. Ils ne modifient pas les valeurs de stratégie du robot.
Les commandes et tous les paramètres sont dans les JSON.

| Run | Trades principaux | PnL net natif | PF principal glissant | Drawdown | Trades shadow | Changements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Témoin désactivé | 2 | −0,938630 | — | — | 0 | — |
| Mémoire neuve | 2 | −0,938630 | 0,646305 | 4,851073 | 5 | 5 |
| Mémoire rechargée | 2 | +1,633064 | 1,618861 | 4,900407 | 7 | 8 |
| Lookback initial maximal 19 | 4 | +0,653328 | 1,250440 | 3,745785 | 12 | 8 |

Ces montants conservent l'unité native du backtester, pas une conversion
automatique en EUR/USD. Le témoin ne calcule pas les nouveaux diagnostics de
drawdown ; son bilan natif est préservé. Les résultats shadow sont séparés.

Le premier run utilise le modèle
`a6b7ef95ea8c8a5000d17ae9ac4cda665bbd9302d7bda2adb3c432c0591f30ef`.
Il démarre avec `(10, trailing=false, k=1,5)` et finit avec
`(10, trailing=true, k=0,375)`. Sa mémoire contient sept observations : deux
principales apprises de poids total 2, cinq shadow dont quatre apprises de
poids total 0,116667. Le scaler est figé et les matrices ont été mises à jour.

Le deuxième lancement affiche `loaded=true`, reprend **exactement** cette
action finale au démarrage et conserve le même modèle. Il finit avec
`(14, trailing=true, k=1,125)`. La mémoire cumulée contient seize observations,
avec quatre principales de poids total 4 et douze shadow dont neuf apprises de
poids total 0,254167, réparties entre plusieurs groupes plafonnés séparément.
Le rapport indique `knowledge_at_or_after_run_start=true` et
`independent_validation=false`. Son PF n'est donc pas un résultat hors
échantillon. Le test d'apprentissage sur marché synthétique démontre une
amélioration par rapport à une action fixe, sans en déduire une rentabilité réelle.

Les fichiers [initial.json](initial.json), [reload.json](reload.json),
[max_lookback.json](max_lookback.json) et leurs logs publient les contextes,
actions, récompenses, budgets, expositions et positions encore ouvertes. Les
checkpoints ne forcent aucune liquidation finale.

## Surcoût mesuré

Chaque mesure vient d'un processus neuf. Le temps porte sur `engine.run` ; le
pic mémoire est le pic du processus entier, imports et données inclus.

| Run | Temps mural | Temps CPU | Pic mémoire du processus |
| --- | ---: | ---: | ---: |
| Témoin | 0,440 s | 0,453 s | 92,50 MiB |
| Mémoire neuve, six slots | 3,539 s | 3,313 s | 98,00 MiB |
| Rechargement, six slots | 3,479 s | 3,188 s | 98,15 MiB |
| Lookback maximal 19, six slots | 3,643 s | 3,266 s | 97,83 MiB |

Sur cet échantillon, les simulations et la persistance ajoutent environ
3,0–3,2 s et 5,3–5,6 MiB de pic mémoire. Le rapport
[overhead.json](overhead.json) donne les ratios. Les features sont mises en
cache par chandelle close ; matrices et politique sont mises à jour après les
clôtures. Aucun apprentissage matriciel ni écriture de modèle n'a lieu à chaque
tick. Ces mesures sur deux heures de marché peu actif ne constituent pas un
benchmark de charge de production ; le coût croît avec les analyses,
positions, clôtures et variantes.

## Validations externes non réalisées

Les tests de `BrokerBanditBridge` utilisent des événements broker synthétiques
explicitement identifiés : devise du compte, risque confirmé, net incluant les
frais, deals partiels, ordre déterministe et replays après redémarrage. Aucun
transport d'ordres live, accès de compte, réconciliation avec un broker connecté
ou conversion FX cotée n'existe dans ce dépôt ou n'a été testé. L'hôte doit
raccorder ces services et son ledger de portefeuille aux contrats fournis.
Aucun compte, dépôt de fonds ou trade réel n'a été créé pour cette validation.
Slippage, latence et impact de marché restent des limites de la simulation
bid/ask existante.
