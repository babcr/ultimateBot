Export historique JForex 4
=========================

Dans JForex 4, ouvrir et recompiler `DukascopyHistoricalTickExport.java`, puis
lancer la strategie avec les parametres suivants. Verifier les valeurs dans
la fenetre de lancement : JForex peut conserver celles de l'ancien export.

| Parametre | Valeur |
| --- | --- |
| Instruments CSV | `USDJPY` |
| Dossier de sortie | `C:\jforex_history` |
| Debut GMT | `2015-01-01` |
| Fin GMT | Vide : dernier jour UTC cloture, calcule a chaque lancement |
| Reecrire les jours existants | `false` pour reprendre en verifiant chaque fichier |
| Stopper la strategie a la fin | `true` |

La selection par defaut exporte uniquement USDJPY du 1er janvier 2015 au
dernier jour UTC cloture. Le jour UTC en cours est exclu afin de ne pas publier
un fichier journalier incomplet. Les deux dates et les instruments restent
configurables. Pour chaque paire, la strategie interroge la date du premier tick
disponible annoncee par JForex (`IDataService.getTimeOfFirstTick`). Elle ignore
les annees anterieures avec un message `skip_year`, puis passe a l'annee
suivante. Si l'historique commence en cours d'annee, elle reprend au premier
jour disponible (`partial_year`), en conservant les ticks de ce jour.
Si aucun historique n'existe dans la plage, la paire n'a aucun jour a exporter
et les autres paires continuent. Aucun fichier journalier vide n'est cree pour
ces periodes indisponibles. Le bilan separe les annees/jours indisponibles des
fichiers existants valides ignores a la reprise.

Ce controle porte sur le debut de disponibilite annonce par la plateforme ;
il ne deduit pas l'absence d'une annee a partir d'un jour ferme. Les erreurs
de connexion ou les chargements incomplets restent des erreurs d'export.

La strategie ecrit un fichier `.tsv.part` par paire et par jour. Elle refuse un
instrument inattendu, une cotation invalide, des volumes negatifs ou non finis,
un tick hors du jour demande et un horodatage decroissant. Plusieurs ticks a la
meme milliseconde restent autorises. L'ecriture et l'arret sont synchronises.
Un verrou empeche deux strategies d'export de travailler dans le meme dossier.

Un jour n'est publie en `.tsv` qu'apres `allDataLoaded=true`, fermeture du fichier
et relecture integrale : neuf champs par ligne, fin de ligne presente, un START
initial, une session stable, des sequences consecutives et un STOP final dont
la sequence correspond au nombre de QUOTE. Les jours UTC en cours ou futurs
sont exclus, meme si la date de fin les inclut. Les jours sans cotation, par
exemple lorsque le marche est ferme, contiennent START et STOP avec sequence 0.

En cas d'erreur, l'export s'arrete et supprime le fichier temporaire du jour.
Il ne saute aucun tick incorrect. Le journal indique la paire, le jour et
l'erreur. A la reprise, les jours valides sont ignores ; les jours corrompus
sont recharges et leur ancien fichier n'est remplace qu'apres validation.
Si la meme cotation source est invalide a chaque tentative, l'export s'arretera
de nouveau : il faudra examiner le message JForex plutot que supprimer ce tick.

Pour verifier les exports sans creer de parquet ni installer pyarrow, depuis
la racine du depot (adapter les dates a la plage effectivement exportee) :

```powershell
python backtester/other_tick_busters/import_jforex_history.py --validate-only --symbols USDJPY --start-date 2015-01-01
```

Le dossier source est celui indique ci-dessus. Le script d'import conserve
son ancien groupe de quatre paires par defaut ; preciser `--symbols USDJPY`
pour cet export.
Utiliser `--source-dir`, `--symbols` et `--end-date` pour une autre plage ou un
export partiel. Pour importer les fichiers valides dans la base parquet :

```powershell
python backtester/other_tick_busters/import_jforex_history.py --output-dir C:\ticks --symbols USDJPY --start-date 2015-01-01
```

Cette conversion necessite pyarrow. Elle verifie aussi les sessions, sequences
et bornes de chaque jour ; un mois invalide n'est pas publie. Un parquet deja
present est ignore par defaut ; `--overwrite` permet de le reconstruire.

Ces controles garantissent l'integrite des trames recues et enregistrees. Ils
ne prouvent pas que le serveur ou le cache Dukascopy fournit chaque tick du
marche : la sequence est attribuee localement et `allDataLoaded` est le signal
de fin fourni par l'API.

Verification du code
-------------------

Les tests Python se lancent depuis le dossier parent du depot :

```powershell
python -m unittest ultimateBot.tests.test_jforex_history -v
```

Le test Java utilise la vraie API JForex avec un historique simule. Depuis la
racine du depot, avec un JDK 8 ou plus recent et le chemin de l'API installee :

```powershell
$apiJar = 'C:\Program Files\JForex4\libs\demo\4.9.7\jforex-api-4.9.5.jar'
$apiDirectory = Split-Path $apiJar
$buildDir = Join-Path $env:TEMP 'ultimateBot-jforex-tests'
$fixtureDir = Join-Path $env:TEMP ('ultimateBot-jforex-fixtures-' + [guid]::NewGuid())
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
javac -cp $apiJar -d $buildDir backtester/other_tick_busters/DukascopyHistoricalTickExport.java tests/java/DukascopyHistoricalTickExportTest.java
java -cp "$buildDir;$apiDirectory\*" DukascopyHistoricalTickExportTest $fixtureDir
```
