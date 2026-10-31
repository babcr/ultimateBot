-- PURGE TOTALE des donnees de recherche swing.
-- Cible : C:\TradingBotData\swing_ts2vec\research.sqlite3
-- Conserve les tables, index et vues. Ne touche pas aux fichiers de ticks/modeles.
-- Arreter les collectes/recherches et fermer les autres connexions a cette base.
-- DBeaver : Auto-commit active, aucune transaction deja ouverte.
-- Selectionner TOUT le script (Ctrl+A), puis Executer le script SQL (Alt+X).
-- Ctrl+Entree peut executer uniquement l'instruction sous le curseur.
-- Executer le script sur une seule connexion ; arreter a la premiere erreur.

PRAGMA busy_timeout = 60000;

-- Checkpoint initial : la premiere colonne doit valoir 0 (pas de verrou bloquant).
PRAGMA wal_checkpoint(TRUNCATE);
PRAGMA journal_mode = DELETE;

-- Ces reglages concernent la connexion de maintenance.
-- Le VACUUM final retire physiquement les pages liberees.
PRAGMA foreign_keys = OFF;
PRAGMA secure_delete = OFF;

BEGIN IMMEDIATE;

DELETE FROM main.swing_predictions;
DELETE FROM main.swing_assignments;
DELETE FROM main.swing_embeddings;
DELETE FROM main.swing_tests;
DELETE FROM main.swing_resolvers;
DELETE FROM main.swing_experiments;
DELETE FROM main.swing_embedding_models;
DELETE FROM main.swing_context_build_checkpoints;
DELETE FROM main.swing_context_coverage;
DELETE FROM main.swing_context_builds;
DELETE FROM main.swing_window_searches;
DELETE FROM main.swing_run_extensions;
DELETE FROM main.swing_contexts;
DELETE FROM main.swing_events;
DELETE FROM main.swing_runs;

COMMIT;

-- Doit etre execute hors transaction : rend l'espace au systeme de fichiers.
VACUUM main;

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA wal_checkpoint(TRUNCATE);

-- Verification rapide, meme si cette seule instruction est lancee par erreur.
-- Les trois indicateurs doivent valoir 0 (aucune ligne restante).
SELECT
    EXISTS(SELECT 1 FROM main.swing_runs LIMIT 1) AS contient_des_collectes,
    EXISTS(SELECT 1 FROM main.swing_events LIMIT 1) AS contient_des_evenements,
    EXISTS(SELECT 1 FROM main.swing_contexts LIMIT 1) AS contient_des_contextes;
