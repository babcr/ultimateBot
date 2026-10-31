# Recherche statistique sur les swings collectés

`research_statistical_swings.py` recherche des règles explicites associées à une
espérance de gain positive, sans TS2Vec, XGBoost ou entraînement de modèle.
Il peut conclure qu'aucune règle ne satisfait les critères. Le meilleur résultat
d'une exploration ne devient pas automatiquement une stratégie.

## Exécution

Depuis `ultimateBot`, pour la collecte actuelle :

```powershell
py research_statistical_swings.py `
  --run-id 88a154e4d58a4cd7bb0c6c8a10cfa588 `
  --output artifacts/statistical_swings_new
```

L'extraction lit uniquement les colonnes scalaires de `swing_events` avec une
connexion SQLite en lecture seule (`mode=ro`, `query_only`). Elle ne charge ni
les gros blobs de fenêtres, ni les poids d'un modèle. La base n'est pas modifiée.
La reconstruction des antécédents est faite une seule fois, dans `features.npz`.

Pour reproduire la recherche sur un instantané existant :

```powershell
py research_statistical_swings.py `
  --features artifacts/statistical_swings_88a154e4d58a_v2/features.npz `
  --output artifacts/statistical_swings_reproduction
```

Il faut un nouveau dossier de sortie lorsqu'une règle a déjà été figée.
`--extract-only` permet de s'arrêter après la création de l'instantané.
`--help` affiche les paramètres de protocole. Les modifier après examen des
résultats constitue une nouvelle exploration, et non une confirmation indépendante.
NumPy, SciPy et les dépendances de lecture des collectes doivent être installés ;
ni Torch ni XGBoost ne sont utilisés par cette recherche.

## Variables disponibles au moment du signal

Le contexte contient les **16 derniers swings alternés confirmés**. Une nouvelle
confirmation du même type remplace la précédente, comme dans la collecte. Deux
swings opposés simultanés sont ambigus : le contexte est réinitialisé sans leur
inventer un ordre. L'extraction groupe ces confirmations par date **et symbole**.

| Variable | Définition |
|---|---|
| `leg_bps` | Amplitude logarithmique de la dernière jambe, en points de base |
| `leg_ratio` | Amplitude absolue de la dernière jambe / celle de la précédente |
| `previous_leg_ratio` | Même rapport sur les deux jambes précédentes |
| `gap_ratio` | Délai de confirmation de la dernière jambe / médiane des délais précédents |
| `same_kind_change` | Déplacement du pivot par rapport au précédent de même type, orienté dans le sens du trade et divisé par la jambe précédente |
| `structure` | `1` si les derniers hauts et bas se déplacent dans le sens du trade, `-1` dans le sens opposé, `0` sinon |
| `path_efficiency` | Distance entre premier et dernier pivot / somme des amplitudes des jambes |
| `span_bars` | Temps couvert par les 16 swings, en unités de bougies |
| `volume_ratio` | Volume moyen du swing actuel / médiane des cinq swings précédents du même type |
| `opposite_volume_ratio` | Même calcul avec les cinq swings précédents du type opposé |
| Autres | Paire, sens du trade, jour UTC, tranche horaire UTC de six heures, délai de confirmation du pivot |

Les volumes sont ceux de la collecte : moyenne bid pour un swing haut, ask pour
un swing bas. Le swing actuel est exclu des références de volume. Les valeurs
absentes ou les divisions impossibles ne sont pas imputées. Le PnL, le MFE,
le MAE, le trailing réalisé et le prix d'exécution ne sont jamais des variables
d'entrée.

## Protocole par défaut

1. **Découverte 2015–2021** : découper chaque variable numérique en trois classes
   selon les tertiles de cette période uniquement ; tester toutes les conditions
   simples et les conjonctions de deux conditions sur des variables différentes.
2. Retenir les règles avec au moins 1 000 opportunités, moyenne positive,
   profit factor ≥ 1,05, au moins 60 % d'années positives et q BY ≤ 0,10.
3. **Validation 2022–2023** : examiner uniquement les règles précédemment
   retenues. Exiger au moins 200 opportunités, les deux années positives,
   profit factor ≥ 1,05, q BY ≤ 0,10 et borne basse de l'intervalle à 95 % positive.
4. Choisir au maximum une règle selon cette borne basse de validation, puis la
   figer dans `frozen_rule.json` **avant** d'évaluer la période réservée.
5. **Test 2024–2025** : évaluer cette seule règle. Exiger au moins 200 observations,
   les deux années positives, PF ≥ 1,05 et une borne basse positive avec des blocs
   de 14, 28 et 56 jours. Si aucune règle ne passe la validation, ne pas consulter
   les performances de test. Les signaux de **2026 sont exclus de l'extraction**.

Les observations sont des opportunités clôturées dont le résultat est connu
avant la fin de chaque période. Les trades encore ouverts à cette borne, les
ordres expirés et les résultats manquants sont exclus ; les fenêtres historiques
peuvent traverser une frontière de période puisqu'elles ne contiennent que du passé.
La période de test est réservée **à cette analyse** : cela n'efface pas les
consultations de ces années dans d'autres recherches du projet.

Les opportunités se chevauchent. Le test de moyenne utilise donc des blocs
calendaires de 28 jours qui regroupent toutes les paires, plutôt que de considérer
chaque trade comme une observation indépendante. La correction
[Benjamini–Yekutieli](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html)
porte sur l'ensemble de la famille testée à chaque étape. Son contrôle dépend
de la validité des p-values ; ici les tests par blocs restent des approximations.

Les intervalles sont obtenus avec 3 000 rééchantillonnages de blocs calendaires
contigus, avec une graine fixe. Les sommes de R et les effectifs de toutes les
paires sont rééchantillonnés ensemble. Un intervalle exploratoire du meilleur
essai n'intègre pas à lui seul le biais lié au choix de ce meilleur essai.

## Fichiers et limites

`protocol.json` décrit les critères, la provenance et la configuration des
sorties. `discovery_rules.json` et `validation_rules.json` contiennent les essais.
`frozen_rule.json` contient soit la règle retenue, soit la raison de l'absence de
sélection. `report.json`, `rapport.md` et `discovery_leaderboard.csv` présentent
les résultats. L'instantané comporte une empreinte des événements lus et sa
version d'extraction. Les résultats actuels corrigés sont dans
`artifacts/statistical_swings_88a154e4d58a_v2`.

**R = PnL réalisé / risque initial.** Les sorties sont celles réellement
simulées par la collecte, avec ses quotes bid/ask. Le programme n'invente pas les
résultats d'un autre stop, d'un autre trailing ou d'un take profit à partir du
seul MFE/MAE. Commissions et slippage supplémentaire ne sont pas ajoutés.

Les sommes de R décrivent des opportunités chevauchantes, pas le rendement
d'un compte. Une règle retenue demanderait ensuite un backtest avec les limites
de positions, la taille des ordres et les mêmes sorties. L'absence de règle
qualifiée ne prouve pas l'absence d'un edge dans toute autre famille de patterns
ou configuration de sorties.

## Résultat sur la collecte actuelle

Sur le run `88a154e4d58a4cd7bb0c6c8a10cfa588` (bougies de 20 minutes,
stop avec marge de 10 pips, début du trailing à 1,5R), **aucune des 1 118 règles
ne passe les critères de découverte**. Les performances de 2024–2025 n'ont
donc pas été évaluées et aucune règle opérationnelle n'a été retenue.

| Échantillon sans filtre | Opportunités clôturées | Moyenne R | Profit factor |
|---|---:|---:|---:|
| Découverte 2015–2021 | 323 215 | −0,08583 | 0,8670 |
| Validation 2022–2023 | 90 038 | −0,10665 | 0,8365 |

Seules cinq règles ont une moyenne exploratoire positive. La meilleure est
« USDJPY le lundi » : 12 961 observations, +0,02823R en moyenne, PF 1,0459.
L'intervalle descriptif par blocs à 95 % est [−0,03568R ; +0,09108R] et le
q BY vaut 1. Ce résultat ne suffit pas à établir une espérance positive.
Le détail est conservé dans `artifacts/statistical_swings_88a154e4d58a_v2/rapport.md`.

Vérification depuis le dossier parent de `ultimateBot` :

```powershell
py -m unittest ultimateBot.tests.test_swing_statistics -q
```
