import com.dukascopy.api.*;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.Calendar;
import java.util.Date;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TimeZone;
import java.util.UUID;

/**
 * Export historique authentifie depuis JForex vers des TSV journaliers sur disque.
 *
 * Chaque fichier journalier contient des trames START/QUOTE/STOP compatibles avec
 * le contrat tabulaire du bridge live, afin qu'un importeur Python puisse ensuite
 * reconstituer la base parquet mensuelle sous D:\ticks sans repasser par le flux
 * historique public Dukascopy.
 */
@RequiresFullAccess
public class DukascopyHistoricalTickExport implements IStrategy {
    private static final TimeZone GMT = TimeZone.getTimeZone("GMT");
    private static final long DAY_MS = Period.DAILY.getInterval();

    @Configurable("Instruments CSV")
    public String instrumentsCsv = "EURGBP,EURCHF,EURUSD,EURCAD,EURAUD,EURNZD,USDCHF,USDCAD,GBPUSD,GBPCAD,GBPCHF,GBPAUD,GBPNZD,BTCUSD,ETHUSD";

    @Configurable("Dossier de sortie")
    public String outputDirectory = "D:\\jforex_history";

    @Configurable("Debut GMT yyyy-MM-dd")
    public String startDate = "2020-01-01";

    @Configurable("Fin GMT yyyy-MM-dd inclus (vide = dernier jour UTC cloture)")
    public String endDate = "";

    @Configurable("Reecrire les jours existants")
    public boolean overwriteExistingDays = false;

    @Configurable("Stopper la strategie a la fin")
    public boolean stopWhenDone = true;

    private final Deque<ExportJob> pendingJobs = new ArrayDeque<ExportJob>();
    private final Map<Instrument, Integer> exportedDaysByInstrument = new LinkedHashMap<Instrument, Integer>();
    private final Map<Instrument, Long> exportedTicksByInstrument = new LinkedHashMap<Instrument, Long>();
    private final String session = UUID.randomUUID().toString();

    private IContext context;
    private IHistory history;
    private IConsole console;
    private Path baseOutputDirectory;

    private int queuedDays = 0;
    private int skippedDays = 0;
    private int completedDays = 0;
    private long completedQuotes = 0;
    private boolean stopping = false;
    private String asyncFailureMessage = null;
    private Exception asyncFailure = null;
    private DayWriter activeWriter = null;

    @Override
    public void onStart(IContext context) throws JFException {
        this.context = context;
        this.history = context.getHistory();
        this.console = context.getConsole();

        try {
            baseOutputDirectory = Paths.get(outputDirectory);
            Files.createDirectories(baseOutputDirectory);

                Set<Instrument> instruments = parseInstruments(instrumentsCsv);
            long startInclusive = parseUtcDate(startDate);
            long endExclusive = computeEndExclusive(endDate);
            if (startInclusive >= endExclusive) {
                throw new IllegalArgumentException("La fenetre demandee est vide");
            }

                Set<Instrument> subscribedInstruments = subscribeAndFilterInstruments(instruments);
                planJobs(subscribedInstruments, startInclusive, endExclusive);

            print(String.format(
                    Locale.US,
                    "[history-export] instruments=%d queued_days=%d skipped_days=%d start=%s end_exclusive=%s output=%s",
                    subscribedInstruments.size(),
                    queuedDays,
                    skippedDays,
                    formatUtcDay(startInclusive),
                    formatUtcDay(endExclusive),
                    baseOutputDirectory));

            if (pendingJobs.isEmpty()) {
                print("[history-export] Rien a exporter.");
                printSummary();
                if (stopWhenDone) {
                    context.stop();
                }
                return;
            }

            startNextJob();
        } catch (Exception error) {
            throw new JFException("Impossible de demarrer l'export historique : " + error.getMessage(), error);
        }
    }

    private synchronized void startNextJob() {
        if (stopping) {
            return;
        }
        if (pendingJobs.isEmpty()) {
            print("[history-export] Export termine.");
            printSummary();
            if (stopWhenDone) {
                context.stop();
            }
            return;
        }

        final ExportJob job = pendingJobs.removeFirst();
        final DayWriter writer;
        try {
            writer = new DayWriter(job.targetPath, job.instrument, session, job.startInclusiveMs);
            activeWriter = writer;
        } catch (IOException error) {
            abortStrategy("Impossible d'ouvrir le fichier cible " + job.targetPath, error);
            return;
        }

        print(String.format(
                Locale.US,
                "[history-export] start instrument=%s day=%s file=%s remaining=%d",
                job.instrument,
                formatUtcDay(job.startInclusiveMs),
                job.targetPath,
                pendingJobs.size()));

        try {
            history.readTicks(
                    job.instrument,
                    job.startInclusiveMs,
                    job.endExclusiveMs - 1,
                    new LoadingDataListener() {
                        @Override
                        public void newTick(Instrument instrument, long time, double ask, double bid,
                                            double askVol, double bidVol) {
                            try {
                                writer.writeQuote(time, bid, ask, bidVol, askVol);
                            } catch (IOException error) {
                                recordAsyncFailure(
                                        "Ecriture impossible pour " + job.instrument + " " + formatUtcDay(job.startInclusiveMs),
                                        error);
                            }
                        }

                        @Override
                        public void newBar(Instrument instrument, Period period, OfferSide side, long time,
                                           double open, double close, double low, double high, double vol) {
                            // Aucun bar attendu ici.
                        }
                    },
                    new LoadingProgressListener() {
                        @Override
                        public void dataLoaded(long start, long end, long currentPosition, String information) {
                            // Progression volontairement silencieuse pour ne pas noyer le journal.
                        }

                        @Override
                        public void loadingFinished(boolean allDataLoaded, long start, long end, long currentPosition) {
                            completeJob(job, writer, allDataLoaded);
                        }

                        @Override
                        public boolean stopJob() {
                            return stopping;
                        }
                    });
        } catch (JFException error) {
            writer.abortQuietly();
            activeWriter = null;
            abortStrategy(
                    "Lecture historique impossible pour " + job.instrument + " " + formatUtcDay(job.startInclusiveMs),
                    error);
        }
    }

    private synchronized void completeJob(ExportJob job, DayWriter writer, boolean allDataLoaded) {
        if (asyncFailure != null) {
            writer.abortQuietly();
            activeWriter = null;
            abortStrategy(asyncFailureMessage, asyncFailure);
            return;
        }
        if (!allDataLoaded) {
            writer.abortQuietly();
            activeWriter = null;
            abortStrategy(
                    "Chargement incomplet pour " + job.instrument + " " + formatUtcDay(job.startInclusiveMs),
                    null);
            return;
        }

        try {
            writer.finish(job.endExclusiveMs - 1);
            activeWriter = null;
        } catch (IOException error) {
            writer.abortQuietly();
            activeWriter = null;
            abortStrategy(
                    "Finalisation impossible pour " + job.instrument + " " + formatUtcDay(job.startInclusiveMs),
                    error);
            return;
        }

        completedDays++;
        completedQuotes += writer.getQuoteCount();
        exportedDaysByInstrument.put(job.instrument, exportedDaysByInstrument.get(job.instrument) + 1);
        exportedTicksByInstrument.put(job.instrument, exportedTicksByInstrument.get(job.instrument) + writer.getQuoteCount());

        print(String.format(
                Locale.US,
                "[history-export] done instrument=%s day=%s quotes=%d file=%s",
                job.instrument,
                formatUtcDay(job.startInclusiveMs),
                writer.getQuoteCount(),
                job.targetPath));

        startNextJob();
    }

    private void planJobs(Set<Instrument> instruments, long startInclusive, long endExclusive) throws IOException {
        for (Instrument instrument : instruments) {
            exportedDaysByInstrument.put(instrument, 0);
            exportedTicksByInstrument.put(instrument, 0L);
            for (long dayStart = startInclusive; dayStart < endExclusive; dayStart += DAY_MS) {
                ExportJob job = buildJob(instrument, dayStart);
                Path tempPath = tempPath(job.targetPath);
                if (!overwriteExistingDays && Files.exists(job.targetPath)) {
                    Files.deleteIfExists(tempPath);
                    skippedDays++;
                    continue;
                }
                if (Files.exists(tempPath)) {
                    Files.delete(tempPath);
                }
                pendingJobs.addLast(job);
                queuedDays++;
            }
        }
    }

    private ExportJob buildJob(Instrument instrument, long dayStart) {
        Calendar calendar = Calendar.getInstance(GMT);
        calendar.setTimeInMillis(dayStart);
        String symbol = instrument.toString().replace("/", "");
        String year = String.format(Locale.US, "%04d", calendar.get(Calendar.YEAR));
        String month = String.format(Locale.US, "%02d", calendar.get(Calendar.MONTH) + 1);
        String filename = String.format(
                Locale.US,
                "%04d-%02d-%02d.tsv",
                calendar.get(Calendar.YEAR),
                calendar.get(Calendar.MONTH) + 1,
                calendar.get(Calendar.DAY_OF_MONTH));
        Path targetPath = baseOutputDirectory.resolve(symbol).resolve(year).resolve(month).resolve(filename);
        return new ExportJob(instrument, dayStart, dayStart + DAY_MS, targetPath);
    }

    private synchronized void recordAsyncFailure(String message, Exception error) {
        if (asyncFailure != null) {
            return;
        }
        asyncFailureMessage = message;
        asyncFailure = error;
        stopping = true;
    }

    private synchronized void abortStrategy(String message, Exception error) {
        if (stopping && asyncFailure == null && error == null) {
            return;
        }
        stopping = true;
        if (activeWriter != null) {
            activeWriter.abortQuietly();
            activeWriter = null;
        }
        err("[history-export] " + message);
        if (error != null) {
            err(error.toString());
        }
        if (context != null) {
            context.stop();
        }
    }

    private void printSummary() {
        print(String.format(
                Locale.US,
                "[history-export] summary queued_days=%d skipped_days=%d completed_days=%d completed_quotes=%d",
                queuedDays,
                skippedDays,
                completedDays,
                completedQuotes));
        for (Instrument instrument : exportedDaysByInstrument.keySet()) {
            print(String.format(
                    Locale.US,
                    "[history-export] instrument=%s exported_days=%d exported_quotes=%d",
                    instrument,
                    exportedDaysByInstrument.get(instrument),
                    exportedTicksByInstrument.get(instrument)));
        }
    }

    private Set<Instrument> subscribeAndFilterInstruments(Set<Instrument> requestedInstruments) {
        context.setSubscribedInstruments(requestedInstruments, true);

        LinkedHashSet<Instrument> subscribedInstruments = new LinkedHashSet<Instrument>(context.getSubscribedInstruments());
        subscribedInstruments.retainAll(requestedInstruments);

        LinkedHashSet<String> blockedCrypto = new LinkedHashSet<String>();
        LinkedHashSet<String> blockedOther = new LinkedHashSet<String>();
        for (Instrument instrument : requestedInstruments) {
            if (subscribedInstruments.contains(instrument)) {
                continue;
            }
            String symbol = instrument.toString();
            if (isCryptoInstrument(symbol)) {
                blockedCrypto.add(symbol);
            } else {
                blockedOther.add(symbol);
            }
        }

        if (!blockedCrypto.isEmpty()) {
            err("[history-export] cryptos non abonnees: " + blockedCrypto
                    + ". Accepter d'abord la \"Cryptocurrencies Disclaimer\" dans JForex puis relancer la strategie.");
        }
        if (!blockedOther.isEmpty()) {
            err("[history-export] instruments refuses ou indisponibles apres tentative d'abonnement: " + blockedOther);
        }
        if (subscribedInstruments.isEmpty()) {
            throw new IllegalArgumentException("Aucun instrument effectivement abonne par JForex");
        }
        return subscribedInstruments;
    }

    private Set<Instrument> parseInstruments(String csv) {
        LinkedHashSet<Instrument> instruments = new LinkedHashSet<Instrument>();
        LinkedHashSet<String> skippedInstruments = new LinkedHashSet<String>();
        for (String raw : csv.split(",")) {
            String token = raw.trim().toUpperCase(Locale.US);
            if (token.isEmpty()) {
                continue;
            }
            if (!token.contains("/") && token.length() != 6) {
                throw new IllegalArgumentException("Instrument invalide: " + raw);
            }
            String normalized = token.contains("/") ? token : token.substring(0, 3) + "/" + token.substring(3);
            if (normalized.length() < 7) {
                throw new IllegalArgumentException("Instrument invalide: " + raw);
            }
            Instrument instrument = Instrument.fromString(normalized);
            if (instrument == null) {
                skippedInstruments.add(normalized);
                continue;
            }
            instruments.add(instrument);
        }
        if (instruments.isEmpty()) {
            if (!skippedInstruments.isEmpty()) {
                throw new IllegalArgumentException("Aucun instrument supporte par JForex: " + skippedInstruments);
            }
            throw new IllegalArgumentException("Aucun instrument fourni");
        }
        if (!skippedInstruments.isEmpty()) {
            err("[history-export] instruments ignores car indisponibles ou inconnus dans ce JForex: " + skippedInstruments);
        }
        return instruments;
    }

    private static boolean isCryptoInstrument(String symbol) {
        return "BTC/USD".equals(symbol) || "ETH/USD".equals(symbol) || "XRP/USD".equals(symbol);
    }

    private static long parseUtcDate(String text) throws ParseException {
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd", Locale.US);
        formatter.setLenient(false);
        formatter.setTimeZone(GMT);
        return formatter.parse(text.trim()).getTime();
    }

    private static long floorUtcDay(long timestampMs) {
        Calendar calendar = Calendar.getInstance(GMT);
        calendar.setTimeInMillis(timestampMs);
        calendar.set(Calendar.HOUR_OF_DAY, 0);
        calendar.set(Calendar.MINUTE, 0);
        calendar.set(Calendar.SECOND, 0);
        calendar.set(Calendar.MILLISECOND, 0);
        return calendar.getTimeInMillis();
    }

    private static long computeEndExclusive(String configuredEndDate) throws ParseException {
        String trimmed = configuredEndDate == null ? "" : configuredEndDate.trim();
        if (trimmed.isEmpty()) {
            return floorUtcDay(System.currentTimeMillis());
        }
        return parseUtcDate(trimmed) + DAY_MS;
    }

    private static String formatUtcDay(long timestampMs) {
        SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd", Locale.US);
        formatter.setTimeZone(GMT);
        return formatter.format(new Date(timestampMs));
    }

    private static Path tempPath(Path finalPath) {
        return Paths.get(finalPath.toString() + ".part");
    }

    private static void moveReplacing(Path source, Path target) throws IOException {
        try {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException error) {
            Files.move(source, target, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private void print(String message) {
        console.getOut().println(message);
    }

    private void err(String message) {
        console.getErr().println(message);
    }

    @Override
    public void onTick(Instrument instrument, ITick tick) throws JFException {}

    @Override
    public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar) throws JFException {}

    @Override
    public void onMessage(IMessage message) throws JFException {}

    @Override
    public void onAccount(IAccount account) throws JFException {}

    @Override
    public void onStop() throws JFException {
        if (activeWriter != null) {
            activeWriter.abortQuietly();
            activeWriter = null;
        }
        if (asyncFailure != null) {
            err("[history-export] stop apres erreur: " + asyncFailureMessage);
        }
    }

    private static final class ExportJob {
        private final Instrument instrument;
        private final long startInclusiveMs;
        private final long endExclusiveMs;
        private final Path targetPath;

        private ExportJob(Instrument instrument, long startInclusiveMs, long endExclusiveMs, Path targetPath) {
            this.instrument = instrument;
            this.startInclusiveMs = startInclusiveMs;
            this.endExclusiveMs = endExclusiveMs;
            this.targetPath = targetPath;
        }
    }

    private static final class DayWriter {
        private final Path finalPath;
        private final Path tempPath;
        private final String symbol;
        private final String session;
        private final BufferedWriter writer;
        private long sequence = 0;
        private long quoteCount = 0;
        private boolean closed = false;

        private DayWriter(Path finalPath, Instrument instrument, String session, long dayStartMs) throws IOException {
            this.finalPath = finalPath;
            this.tempPath = tempPath(finalPath);
            this.symbol = instrument.toString().replace("/", "");
            this.session = session;
            Files.createDirectories(finalPath.getParent());
            if (Files.exists(tempPath)) {
                Files.delete(tempPath);
            }
            writer = Files.newBufferedWriter(
                    tempPath,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE);
            writeFrame("START", dayStartMs, 0.0, 0.0, 0.0, 0.0);
        }

        private void writeQuote(long timestampMs, double bid, double ask, double bidSize, double askSize) throws IOException {
            sequence++;
            writeFrame("QUOTE", timestampMs, bid, ask, bidSize, askSize);
            quoteCount++;
        }

        private void finish(long stopTimestampMs) throws IOException {
            if (closed) {
                return;
            }
            writeFrame("STOP", stopTimestampMs, 0.0, 0.0, 0.0, 0.0);
            writer.close();
            closed = true;
            moveReplacing(tempPath, finalPath);
        }

        private void abortQuietly() {
            if (!closed) {
                try {
                    writer.close();
                } catch (IOException ignored) {
                    // Best effort only.
                }
                closed = true;
            }
            try {
                Files.deleteIfExists(tempPath);
            } catch (IOException ignored) {
                // Best effort only.
            }
        }

        private long getQuoteCount() {
            return quoteCount;
        }

        private void writeFrame(String type, long timestampMs, double bid, double ask,
                                double bidSize, double askSize) throws IOException {
            writer.write(type);
            writer.write('\t');
            writer.write(symbol);
            writer.write('\t');
            writer.write(Long.toString(timestampMs));
            writer.write('\t');
            writer.write(session);
            writer.write('\t');
            writer.write(Long.toString(sequence));
            writer.write('\t');
            writer.write(Double.toString(bid));
            writer.write('\t');
            writer.write(Double.toString(ask));
            writer.write('\t');
            writer.write(Double.toString(bidSize));
            writer.write('\t');
            writer.write(Double.toString(askSize));
            writer.newLine();
        }
    }
}