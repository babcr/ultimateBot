# Spécification — Optimisation continue de ultimateBot par bandit contextuel

Document à transmettre à Codex dans VS Code. Les exigences utilisateur priment sur les valeurs par défaut proposées ici. Les noms de composants sont indicatifs : les adapter à l'architecture réelle du dépôt.

## 1. Mission et résultat attendu

Implémenter entièrement une optimisation contextuelle incrémentale dans le robot existant, utilisable en backtest, en développement, en démo et en production. Ne pas livrer un prototype isolé : intégrer les événements, la simulation, la sauvegarde, les options de lancement, les rapports et les tests.

Le robot ajuste trois paramètres :

1. Le nombre de chandelles de la fenêtre de décision (`--regression-lookback-bars`).
2. L'activation du trailing stop existant (`--trailing-stop`).
3. Une distance supplémentaire positive ou nulle ajoutée à la distance du stop structurel (`protective_stop_buffer_atr`).

Le take profit n'est pas une quatrième variable optimisée. Son calcul reste celui du robot ; si ce calcul dépend déjà de la distance du stop et du ratio reward/risk, cette dépendance doit continuer à fonctionner.

L'apprentissage et la révision de la configuration active interviennent après la clôture de chaque trade, réel ou simulé. Les nouvelles configurations s'appliquent aux prochaines entrées. Une position déjà ouverte conserve sa configuration d'ouverture et les règles de gestion correspondantes jusqu'à sa clôture.

La mémoire est persistante et commune aux différents environnements. Le lancement suivant doit reprendre effectivement l'état appris. Deux modèles possédant des paramètres d'entrée fixes différents doivent avoir des mémoires distinctes. La production apprend sur les trades réels ET sur des trades simulés en parallèle ; elle peut explorer de nouvelles configurations sur le réel dans les limites déjà autorisées au robot.

Le profit factor (PF) existant, sa fenêtre glissante et son traitement des coûts doivent être réutilisés. Le seuil 1,01 est uniquement un repère : ne pas en faire une condition d'entrée, de suspension, de récompense ou de convergence.

## 2. Inspection obligatoire du dépôt

Avant de modifier le code, lire les instructions du dépôt et identifier précisément :

- Les stratégies et leurs paramètres, l'optimiseur existant et les bornes/pas du lookback.
- La construction des chandelles, les données tick et la sémantique des volumes bid/ask.
- Les calculs de tendance, volatilité, ATR, stop structurel et distance supplémentaire existants.
- Le trailing stop fonctionnel et la manière dont il est activé pour chaque position.
- Le sizing, les ordres, les limites de risque, les stops minimaux du broker et les conversions monétaires.
- Le moteur de backtest, la gestion des positions et les événements de clôture, y compris partielle.
- Le calcul du résultat net, le profit factor et sa fenêtre glissante.
- Les options de lancement, les journaux, la persistance et les tests existants.

Documenter les points d'intégration trouvés avec leurs fichiers et fonctions. Réutiliser leurs implementations plutôt que dupliquer les formules. Si une fonctionnalité indispensable manque, la développer dans le composant approprié. Ne demander une précision que si une ambiguïté métier restante ne peut être résolue par lecture du dépôt et par cette spécification.

## 3. Identité du modèle et mémoire commune

Définir une identité stable à partir de : symbole canonique, timeframe, stratégie/version de comportement, paramètres fixes effectifs et versions des contrats de features, actions et récompense.

Calculer `model_id = SHA256(canonical_json(identity))`. Canonicaliser les clés, les valeurs par défaut réellement résolues, les unités et les types. Persister le JSON lisible à côté du hash. Deux valeurs effectives identiques doivent donner la même identité, quelle que soit la syntaxe de lancement.

Les paramètres fixes comprennent notamment ceux qui influencent les entrées, sorties, le sizing, les contraintes de risque et les bornes de recherche. Les paramètres appris (`lookback`, activation trailing, supplément de stop) sont exclus de ce fingerprint ; leurs bornes et la définition de leur domaine y figurent. Une valeur initiale utilisée uniquement comme amorçage est exclue, sauf si elle influence encore la politique après cet amorçage.

Ne pas inclure le mode backtest/démo/production, les dates du backtest, le chemin des données, le run_id ou le solde courant dans l'identité. Les conserver comme provenance. Les différences de sources, de broker et de coûts effectivement appliqués restent visibles dans les observations. Une incompatibilité d'unités ou de sémantique ne doit pas être fusionnée silencieusement.

Même identité => même mémoire en lecture et écriture par défaut, tous environnements confondus. Un changement incompatible crée une nouvelle identité ou une migration explicite documentée, jamais une réinitialisation silencieuse.

Au premier lancement seulement, si la mémoire est vide, utiliser la configuration statique effective comme baseline et initialiser le bandit avec un prior neutre. Après rechargement, conserver la configuration active, les statistiques et l'état du générateur aléatoire.

## 4. Contexte observable

À chaque sélection de configuration, construire un instantané ne contenant que les informations disponibles à cet instant :

- Pente de la droite de tendance existante.
- Volatilité glissante existante.
- Volume bid moyen par tick et volume ask moyen par tick.
- Volume bid moyen par chandelle et volume ask moyen par chandelle.
- Masques de disponibilité de ces grandeurs et terme constant.

Le contexte de décision enregistré lors de la sélection d'une configuration est celui utilisé pour apprendre ses résultats ultérieurs. Ne pas remplacer ce contexte par celui de clôture. Conserver aussi le contexte d'entrée et le contexte de clôture pour les diagnostics, en les distinguant explicitement.

Pour comparer les lookbacks sans changer simultanément la définition du contexte, utiliser une fenêtre de contexte fixe, issue des paramètres existants. Elle fait partie des paramètres fixes du modèle. Le lookback choisi ne modifie que la fenêtre de décision de la stratégie.

Formaliser les moyennes : pour chaque côté, moyenne de ses volumes tick valides ; par chandelle, somme de ces volumes dans la chandelle, puis moyenne sur les chandelles closes de la fenêtre. Si le dépôt utilise déjà une autre convention, l'identifier, conserver la convention cohérente et l'inscrire dans le contrat de features. Ne pas confondre volume disponible dans une quote et volume de transactions exécutées. Ne pas inventer de volumes quand la source n'en fournit pas ; masquer les features indisponibles et journaliser la convention.

La pente doit avoir une unité documentée ; les volumes sont transformables par `log1p`. Toute normalisation repose sur des observations passées seulement. Pour une première version robuste : collecter un warmup configurable, figer ensuite les paramètres de standardisation et les persister. Pendant le warmup, utiliser la baseline et conserver les observations pour construire le modèle une fois le scaler figé. Ne pas modifier continuellement le scaler sans transformer aussi les statistiques du modèle.

Un historique insuffisant ou des features non finies provoquent un repli explicite vers la dernière configuration valide/baseline ; aucune matrice ne reçoit NaN ou infini.

## 5. Domaine des configurations

Une action est un triplet stable `(lookback, trailing_enabled, stop_extra_atr_multiplier)`.

### Lookback

Utiliser exactement les bornes et contraintes de l'optimiseur existant. Respecter son pas entier, les minima nécessaires aux indicateurs et l'historique réellement disponible. Si le domaine est petit, le reprendre intégralement ; sinon sélectionner initialement jusqu'à neuf valeurs régulièrement réparties sur les valeurs autorisées, incluant les extrêmes et la baseline. Rendre la grille configurable et persistante.

### Trailing stop

Les valeurs sont `false` et `true`. Réutiliser strictement le moteur de trailing existant ; ne pas optimiser ses distances ou son seuil d'activation dans cette mission.

### Distance supplémentaire du stop

Exprimer l'action par un coefficient ATR non négatif. À l'entrée :

`extra_distance = k * ATR_entry`

`stop_distance = structural_stop_distance + extra_distance`

Pour un achat, le stop initial se situe sous l'entrée ; pour une vente, au-dessus. Réutiliser la convention d'entrée bid/ask du moteur. Figer l'ATR de conversion à l'entrée ; ne pas élargir ensuite le stop initial en fonction d'un ATR futur.

Le paramètre de distance supplémentaire déjà présent constitue la borne maximale. Inspecter son unité :

- S'il représente un coefficient ATR, imposer `0 <= k <= configured_max_k`.
- S'il représente une distance absolue, convertir dans l'unité de prix existante et utiliser `extra_distance = min(k * ATR_entry, configured_max_distance)`.
- Si sa convention est différente, écrire un adaptateur explicite et testé ; ne pas traiter un nombre de points comme un coefficient ATR.

Proposer initialement cinq valeurs incluant zéro et la borne maximale, avec conversion cohérente. Si la borne maximale est zéro, réduire cette dimension à une seule action. La grille et la sémantique de plafonnement font partie de l'identité.

Recalculer la quantité avec le stop effectivement choisi et le sizing existant. Le bandit ne peut pas augmenter les limites de risque, de marge ou de positions. Réutiliser les règles de tick size, arrondi et distance minimale du broker ; invalider une action irréalisable plutôt que dépasser sa borne maximale. Si plusieurs coefficients deviennent équivalents après plafonnement, éviter de leur attribuer artificiellement plusieurs essais indépendants.

Avec les valeurs proposées, on obtient au plus 9 × 2 × 5 = 90 actions. Leur identifiant est dérivé de leurs valeurs et unités, jamais de leur indice dans une liste.

## 6. Bandit et apprentissage

Implémenter une première version LinUCB avec un modèle linéaire distinct par action. Cette version finie est inspectable, légère et permet l'exploration par incertitude. Il s'agit d'une adaptation au trading et aux observations simulées ; aucune garantie théorique de rentabilité ne doit être revendiquée.

Pour chaque action a, avec features x et régularisation lambda > 0 :

`A_a = lambda * I + sum(w_i * x_i * x_i.T)`

`b_a = sum(w_i * r_i * x_i)`

`theta_a = solve(A_a, b_a)`

`score_a(x) = x.T * theta_a + alpha * sqrt(x.T * solve(A_a, x))`

Sélectionner l'action valide maximisant le score. Départager les égalités avec un RNG seedé et persisté. Utiliser des résolutions numériques stables, idéalement Cholesky ; éviter une inversion explicite à chaque décision. Lambda, alpha, la seed et la normalisation sont configurables, validés et persistés. Le terme d'exploration permet de tester des actions nouvelles aussi sur le réel, conformément au choix utilisateur.

La récompense de base est le résultat net rapporté au risque monétaire initial de la position :

`R = pnl_net_account_currency / initial_risk_account_currency`

Le risque initial est celui du sizing après application de l'action, figé à l'entrée et calculé avec les conversions existantes. Le résultat net reprend exactement le moteur actuel, sans double déduction des frais. Si le risque est nul, inconnu ou non fini, conserver le résultat en audit mais ne pas apprendre une récompense invalide.

Pour stabiliser LinUCB, utiliser `r = clip(R, -R_clip, +R_clip) / R_clip`, avec `R_clip = 5` par défaut configurable. Conserver R brut et le PnL brut/net dans les rapports. Le PF ne remplace pas cette récompense par trade : il dépend de plusieurs résultats et ne permet pas une attribution simple à chaque action. Le PF existant reste un indicateur de contrôle du comportement global.

Garder une mémoire cumulative par défaut. Ne pas activer de facteur d'oubli arbitraire ni trier les observations de runs historiques indépendants comme un seul flux temporel. Une extension future pourra pondérer la récence, mais elle doit avoir un contrat temporel explicite.

### Pondération réel/simulé

Enregistrer séparément les statistiques et récompenses des observations exécutées et shadow, tout en permettant leur contribution à un même modèle. Une observation exécutée a un poids de base 1 ; une observation shadow a un poids de base 0,25 configurable.

Les simulations du même scénario sont corrélées : leur multiplication ne doit pas créer artificiellement une forte confiance. Répartir le poids shadow entre les variantes du groupe et plafonner la contribution totale d'un groupe à son budget (0,25 par défaut), sur toute sa durée. En backtest, un trade de la trajectoire principale est classé `backtest_executed`, jamais `live_real`. En démo, utiliser `demo_executed`. La provenance reste distincte même si ces exécutions ont un poids de base 1.

Pour l'action exécutée, sa copie shadow est un diagnostic de fidélité : ne pas apprendre deux fois exactement la même action et le même scénario. Les matrices pondérées constituent une heuristique d'exploration ; le score d'incertitude n'est pas une probabilité calibrée de gain.

## 7. Calendrier précis des décisions

Au démarrage, charger la mémoire et sa configuration active, ou la baseline en mémoire vide.

À chaque entrée principale, utiliser la configuration active. Enregistrer de manière immuable : action, version de politique, contexte de sélection, contexte d'entrée, ATR, stop initial, risque initial, quantité et identifiants de provenance. Aucune nouvelle sélection n'est faite à chaque tick/chandelle ou signal si aucun trade n'a clôturé.

À chaque clôture complète d'un trade exécuté ou shadow :

1. Finaliser son résultat selon le moteur actuel.
2. Dédupliquer l'événement et vérifier l'éligibilité à l'apprentissage.
3. Appliquer sa récompense à l'action et au contexte de sélection qui lui étaient attachés.
4. Mettre à jour les diagnostics et le PF de la trajectoire concernée.
5. Construire le contexte observable courant, sélectionner une nouvelle configuration principale valide, et persister observation, modèle et politique atomiquement.
6. Utiliser cette configuration pour les prochaines entrées seulement.

Une clôture partielle est accumulée ; elle ne produit qu'une observation finale quand la position est complètement clôturée. Les ordres rejetés/annulés et les signaux sans position ne sont pas des trades perdants fictifs. Les liquidations forcées en fin de backtest sont identifiées ; par défaut, elles figurent au bilan mais n'entraînent pas le bandit, sauf si cette sortie est une règle normale de la stratégie.

En cas d'événements simultanés, traiter un lot dans un ordre déterministe, mettre à jour tous les résultats, puis sélectionner une seule politique pour les entrées suivant ce lot. Une position encore ouverte ne change pas d'action lorsque d'autres positions clôturent.

La mémoire de plusieurs processus est partagée via une persistance transactionnelle ; un processus doit relire les observations nouvelles et le modèle courant avant sa mise à jour. Les décisions propres à chaque run restent enregistrées. Ne pas imposer le contexte de marché d'un backtest historique à un processus live : synchroniser les connaissances, puis sélectionner dans le contexte local à la prochaine clôture.

## 8. Simulation parallèle de plusieurs configurations

Construire un moteur shadow utilisant les mêmes ticks/chandelles que le robot, avec comptes, états de stratégie et positions virtuels isolés. Il ne peut pas accéder au chemin d'envoi d'ordres réels. Réutiliser le calcul des coûts, les bid/ask, le sizing, les stops, le take profit et le trailing du moteur existant.

Le lookback influence les signaux eux-mêmes : chaque variante doit donc analyser le marché et générer ses propres entrées. Ne pas simplement reprendre les entrées réelles en changeant le stop et présenter cela comme une évaluation complète du lookback.

Initialement limiter le pool à six variantes shadow configurables, choisies parmi la baseline, les meilleures actions estimées et les actions incertaines. Lors d'une rotation déclenchée par une clôture, laisser les positions virtuelles existantes terminer avec leur configuration ; les nouveaux bras reçoivent un historique de warmup constitué uniquement des données déjà disponibles. Ne pas fermer artificiellement leurs trades pour accélérer l'apprentissage.

Pour préserver l'attribution malgré des entrées et durées différentes, définir un groupe de comparaison borné, lancé à un instant connu avec plusieurs variantes figées. Chaque groupe possède son budget de poids shadow ; sa contribution totale reste plafonnée, quel que soit le nombre de trades/variantes. Répartition initiale explicite : pour un budget B et K variantes apprenantes, le j-ième trade clôturé de chaque variante reçoit `w = (B / K) * 2^(-j)`, avec j démarrant à 1. La somme des poids est au plus B ; une variante sans trade ne reçoit pas de récompense artificielle et son budget inutilisé n'est pas redistribué après observation des résultats. Cette règle est une heuristique configurable, distincte des statistiques de performance non pondérées. Les groupes ont une durée d'ouverture des nouvelles entrées configurable, par défaut la durée de la fenêtre de PF existante si elle est temporelle, sinon son nombre de chandelles ou de trades principaux selon sa convention ; après cette limite, leurs positions ouvertes terminent normalement. Un groupe nouveau utilise seulement les données déjà observées. Ne pas comparer uniquement les variantes déjà clôturées en oubliant les positions encore ouvertes ; publier aussi exposition et PnL latent.

La limite de positions est appliquée séparément à chaque compte shadow. Ses positions ne consomment pas la marge réelle. La simulation ne prétend pas reproduire exactement slippage, latence ou impact de marché : enregistrer ses hypothèses et comparer régulièrement l'action exécutée à sa copie shadow.

En l'absence de trades principaux, les clôtures shadow peuvent alimenter l'apprentissage et déclencher les sélections. Si aucun moteur ne génère de trade, rester en état d'observation ; ne pas inventer une récompense ou forcer une entrée hors stratégie.

## 9. Persistance, reprise et déduplication

Utiliser SQLite transactionnel, adapté au fonctionnement Windows local du projet, sauf si le dépôt fournit déjà un stockage équivalent. Prévoir un emplacement configurable commun à tous les lanceurs, exclu du suivi Git. Ne pas utiliser uniquement une variable globale ou un fichier JSON réécrit sans verrou.

Persister au minimum : identités lisibles, versions, grille, scaler, matrices/statistiques, configuration active par run et dernière configuration connue, RNG, observations, groupes shadow, décisions, provenance, positions et événements en attente nécessaires à la reprise.

Chaque observation a une clé stable de déduplication. En live, réutiliser les identifiants broker de position/deal et le modèle ; pour les clôtures partielles, agréger les deals. En shadow, utiliser groupe, variante et identifiant de position. En backtest, conserver aussi une empreinte des données, de la configuration et de la trajectoire/action effective ; ne pas utiliser uniquement un run_id fraîchement généré pour contourner la déduplication.

Rejouer exactement la même observation ne doit pas multiplier les statistiques. Un nouveau passage sur les mêmes données peut produire de nouvelles actions/trajectoires et les apprendre ; signaler explicitement qu'il s'agit d'un entraînement répété sur une période connue. Permettre une répétition pondérée uniquement par option explicite, jamais par défaut.

Transaction atomique : insertion d'observation unique + mise à jour des statistiques + mise à jour de politique. Une panne ne peut pas enregistrer la récompense deux fois ou la perdre entre observation et modèle. À la reprise, réconcilier les positions live avec le broker et restaurer les états shadow quand leur historique est disponible ; sinon marquer l'observation incomplète, sans inventer sa clôture.

Détecter verrouillage, corruption et version incompatible ; sauvegarder les erreurs et refuser l'apprentissage sans stockage fiable. Ne pas supprimer une mémoire existante automatiquement. Prévoir export/import, inspection et reset explicite ciblé sur un modèle ; aucun reset global implicite.

## 10. Options et rapports

Adapter les noms à la CLI/config actuelle. Fournir : activation du bandit ; chemin mémoire ; mode lecture seule optionnel ; alpha/lambda/seed/warmup ; grille ; limite shadow ; pondération shadow ; clipping ; export/inspection de mémoire et réinitialisation explicite ciblée.

Le fonctionnement normal activé charge et enrichit la mémoire commune dans tous les environnements. Le mode lecture seule est un outil de diagnostic, pas une séparation imposée des mémoires. En mode désactivé, conserver le comportement historique sans accès en écriture à la mémoire du bandit.

À chaque run, afficher : modèle chargé/créé, provenance de mémoire, nombre d'observations par source, présence d'apprentissage provenant de dates postérieures au début du backtest, bornes effectives, configuration initiale/finale, nombre de changements, budget shadow et repli éventuel.

Pour chaque décision : instant, run, action, contexte, score moyen prédit, bonus d'exploration, état des données et version. Pour chaque clôture : identifiants, action d'origine, résultat net, R, récompense, poids, source, état de déduplication et cause de sortie.

Rapporter PF existant et résultat net de la trajectoire principale, drawdown et nombre de trades. Séparer les bilans shadow par variante/groupe. Ne pas sommer les résultats shadow et réels comme une performance de portefeuille réalisable. Si le PF est indéfini ou infini faute de pertes, utiliser la convention existante et indiquer le nombre de trades.

Le rapport doit distinguer explicitement entraînement et validation : recharger une mémoire ayant appris sur la période testée ou après celle-ci est autorisé par l'utilisateur, mais ce run n'est pas une validation indépendante hors échantillon. Conserver la provenance nécessaire pour le vérifier. Aucun objectif de PF ou rendement n'est garanti.

## 11. Composants et contrats suggérés

Séparer clairement : `ModelIdentity`, `ContextBuilder`, `ActionSpace`, `LinUCBPolicy`, `RewardBuilder`, `ShadowRunner`, `LearningStore` et un coordinateur d'intégration.

Le contrat d'ouverture doit figer un `DecisionSnapshot` et un `PositionLearningRecord`. Le contrat de clôture fournit un `ClosedTradeObservation` portant toutes les unités et sa provenance. Le store accepte une opération transactionnelle/idempotente d'apprentissage. Le moteur shadow reçoit uniquement des données de marché et une interface d'exécution virtuelle.

Injecter horloge de marché, RNG et store pour obtenir des runs déterministes. Éviter de dépendre de l'heure murale en backtest. Réutiliser les interfaces présentes plutôt que les renommer inutilement. Ne pas ajouter de dépendance ML lourde si NumPy et SQLite suffisent.

## 12. Tests indispensables et critères d'acceptation

1. Identité stable pour mêmes paramètres effectifs ; isolation pour symbole, timeframe, stratégie, borne ou paramètre fixe différent ; partage entre modes.
2. Respect des bornes lookback et conversion/plafonnement ATR/points sur achats et ventes ; sizing recalculé sans dépassement de risque.
3. Récompense calculée avec résultat net existant ; frais déduits une seule fois ; rejet des risques nuls et non finis.
4. Apprentissage incrémental LinUCB déterministe sur un marché synthétique où l'action optimale dépend du contexte ; amélioration par rapport à une baseline sous seed fixée, sans assertion de profit réel.
5. Aucun changement de configuration hors clôture ; aucune modification des positions ouvertes ; attribution au contexte et à l'action d'origine malgré clôtures retardées et multiples positions.
6. Partiels agrégés et observation finale unique ; lot de clôtures simultanées déterministe ; fin de backtest identifiée.
7. Rechargement reproduisant décisions/statistiques à données et seed identiques ; déduplication des événements broker et des reprises ; écriture concurrente sans perte de mise à jour.
8. Budget shadow plafonné et action exécutée non doublement apprise ; ajout de variantes sans multiplication automatique de confiance.
9. Variantes de lookback produisant leurs propres signaux ; aucune possibilité d'envoi d'ordres réels par le shadow, vérifiée avec un client d'exécution espion.
10. Absence de données futures dans features, ATR et warmup ; visibilité de la contamination historique provenant de mémoire réutilisée.
11. Scénarios features manquantes, marché sans trades, ordres rejetés, historique insuffisant, crash et reprise ; repli observable et pas de NaN.
12. Bandit désactivé : résultat et ordres historiques préservés sur un jeu de référence approprié.

Exécuter un backtest court local avec les données disponibles, puis un second lancement qui prouve le chargement et l'utilisation de la mémoire du premier. Si les données/broker nécessaires manquent, fournir un scénario synthétique reproductible et signaler précisément les validations non réalisées. Ne pas ouvrir de compte, déposer de fonds ni exécuter de trades live pour valider cette mission.

Mesurer le surcoût CPU/mémoire sur un échantillon représentatif, notamment avec six variantes shadow et le lookback maximal. Pas d'entraînement matriciel ou d'écriture SQLite à chaque tick : seul le suivi de marché/positions est tick par tick. Le pipeline apprentissage/politique/persistance est déclenché par les clôtures.

## 13. Livrables attendus de Codex VS Code

- Implémentation complète intégrée au dépôt, avec modes existants fonctionnels.
- Options et valeurs effectives documentées, exemple de lancement adapté aux vraies commandes du projet.
- Tests et résultats des vérifications ; migrations de stockage si nécessaires.
- Court document d'architecture avec points d'intégration, conventions d'unités, attribution des récompenses et limites de simulation.
- Rapport d'un run initial et de sa reprise, démontrant l'apprentissage et la persistance.
- Liste précise des éventuels blocages externes ; aucune simulation factice présentée comme une intégration réelle.

Avancer jusqu'au bout de l'implémentation et des vérifications. Les choix de grille, d'algorithme et de pondération proposés sont des valeurs initiales configurables, pas des constantes cachées. Ne pas remplacer l'objectif par une recherche exhaustive à chaque clôture ni relancer l'ancien optimiseur global en boucle.

## Référence technique

Li, Chu, Langford et Schapire, « A Contextual-Bandit Approach to Personalized News Article Recommendation », 2010 : https://arxiv.org/abs/1003.0146. Référence pour LinUCB ; la récompense en R, les groupes shadow, leur pondération et l'intégration de trading ci-dessus sont des choix d'ingénierie de cette spécification.
