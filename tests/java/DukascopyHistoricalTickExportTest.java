import com.dukascopy.api.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.reflect.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

/** Tests executables sans JUnit, avec la vraie API JForex et un historique simule. */
public class DukascopyHistoricalTickExportTest {
    private static final long START = 1577836800000L;
    private static final long STOP = START + 86400000L - 1;
    private static int checks;
    private static final Class<?> WRITER;
    private static final Constructor<?> CONSTRUCTOR;
    private static final Method WRITE, FINISH, ABORT, VALIDATE;
    static {
        try {
            WRITER = Class.forName("DukascopyHistoricalTickExport$DayWriter");
            CONSTRUCTOR = WRITER.getDeclaredConstructor(Path.class, Instrument.class, String.class, long.class);
            CONSTRUCTOR.setAccessible(true);
            WRITE = method(WRITER, "writeQuote", long.class, double.class, double.class, double.class, double.class);
            FINISH = method(WRITER, "finish", long.class);
            ABORT = method(WRITER, "abortQuietly");
            VALIDATE = method(DukascopyHistoricalTickExport.class, "validateDayFile", Path.class, String.class, long.class);
        } catch (Exception error) { throw new ExceptionInInitializerError(error); }
    }
    private static Method method(Class<?> type, String name, Class<?>... args) throws Exception {
        Method result = type.getDeclaredMethod(name, args);
        result.setAccessible(true);
        return result;
    }
    private static Object call(Method method, Object target, Object... args) throws Exception {
        try { return method.invoke(target, args); }
        catch (InvocationTargetException error) { throw (Exception) error.getCause(); }
    }
    private static Object field(Object target, String name) throws Exception {
        Field field = target.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(target);
    }
    private static void check(boolean condition, String description) {
        if (!condition) { throw new AssertionError(description); }
        checks++;
    }
    private interface Action { void run() throws Exception; }
    private static void expectIOException(Action action) throws Exception {
        try { action.run(); throw new AssertionError("IOException attendue"); }
        catch (IOException expected) { checks++; }
    }
    private static Path target(Path root, String symbol) {
        return root.resolve(symbol + "/2020/01/2020-01-01.tsv");
    }
    private static String frame(String kind, long time, long sequence, String values) {
        return kind + "\tEURUSD\t" + time + "\tsession\t" + sequence + "\t" + values + "\n";
    }
    private static String fixture() {
        return frame("START", START, 0, "0\t0\t0\t0")
                + frame("QUOTE", START + 1, 1, "1.1\t1.2\t2\t3")
                + frame("STOP", STOP, 1, "0\t0\t0\t0");
    }
    @SuppressWarnings("unchecked")
    private static final class Platform {
        final DukascopyHistoricalTickExport strategy = new DukascopyHistoricalTickExport();
        final List<LoadingDataListener> ticks = new ArrayList<>();
        final List<LoadingProgressListener> progress = new ArrayList<>();
        final List<Instrument> instruments = new ArrayList<>();
        final List<Long> starts = new ArrayList<>();
        final Map<Instrument, Long> firstTicks = new HashMap<>();
        final ByteArrayOutputStream log = new ByteArrayOutputStream();
        Set<Instrument> subscribed;
        int stops;
        final IContext context;
        Platform(Path root, String symbols) {
            strategy.outputDirectory = root.toString();
            strategy.instrumentsCsv = symbols;
            strategy.startDate = strategy.endDate = "2020-01-01";
            PrintStream output = new PrintStream(log);
            IConsole console = (IConsole) Proxy.newProxyInstance(IConsole.class.getClassLoader(),
                    new Class<?>[]{IConsole.class}, (proxy, method, args) -> output);
            IHistory history = (IHistory) Proxy.newProxyInstance(IHistory.class.getClassLoader(),
                    new Class<?>[]{IHistory.class}, (proxy, method, args) -> {
                        if (method.getName().equals("readTicks")) {
                            instruments.add((Instrument) args[0]);
                            starts.add((Long) args[1]);
                            ticks.add((LoadingDataListener) args[3]);
                            progress.add((LoadingProgressListener) args[4]);
                        }
                        return null;
                    });
            IDataService dataService = (IDataService) Proxy.newProxyInstance(IDataService.class.getClassLoader(),
                    new Class<?>[]{IDataService.class}, (proxy, method, args) -> {
                        if (method.getName().equals("getTimeOfFirstTick")) {
                            return firstTicks.getOrDefault((Instrument) args[0], 0L);
                        }
                        return null;
                    });
            context = (IContext) Proxy.newProxyInstance(IContext.class.getClassLoader(),
                    new Class<?>[]{IContext.class}, (proxy, method, args) -> {
                        switch (method.getName()) {
                            case "getHistory": return history;
                            case "getDataService": return dataService;
                            case "getConsole": return console;
                            case "setSubscribedInstruments": subscribed = (Set<Instrument>) args[0]; return null;
                            case "getSubscribedInstruments": return subscribed;
                            case "stop": stops++; strategy.onStop(); return null;
                            default: return null;
                        }
                    });
        }
        void start() throws Exception { strategy.onStart(context); }
        void quote(int job, long time, double bid, double ask) {
            ticks.get(job).newTick(instruments.get(job), time, ask, bid, 3, 2);
        }
        void finish(int job, boolean complete) {
            progress.get(job).loadingFinished(complete, START, STOP, STOP);
        }
    }
    public static void main(String[] args) throws Exception {
        Path root = Paths.get(args[0]);
        Files.createDirectories(root);
        check(new DukascopyHistoricalTickExport().instrumentsCsv.equals("USDJPY"),
                "USDJPY uniquement par defaut");
        DukascopyHistoricalTickExport defaults = new DukascopyHistoricalTickExport();
        check(defaults.startDate.equals("2015-01-01") && defaults.endDate.isEmpty(),
                "Plage par defaut depuis 2015 jusqu'au dernier jour UTC cloture");
        Method parseInstruments = method(DukascopyHistoricalTickExport.class, "parseInstruments", String.class);
        check(call(parseInstruments, defaults, defaults.instrumentsCsv).equals(Collections.singleton(Instrument.USDJPY)),
                "La selection par defaut contient uniquement l'instrument JForex USD/JPY");
        Method endExclusive = method(DukascopyHistoricalTickExport.class, "computeEndExclusive", String.class);
        Method parseDate = method(DukascopyHistoricalTickExport.class, "parseUtcDate", String.class);
        check(((Long) call(endExclusive, null, "2019-12-31")) == START,
                "Le dernier jour de 2019 est inclus, aucun jour de 2020");
        long closedDayBoundary = ((Long) call(endExclusive, null, ""));
        check(((Long) call(endExclusive, null, defaults.endDate)) == closedDayBoundary,
                "La fin par defaut suit automatiquement la borne UTC des jours clotures");
        check(((Long) call(endExclusive, null, "2099-01-01")) == closedDayBoundary,
                "Une date future ne publie pas de jour UTC en cours");

        Platform historical = new Platform(root.resolve("historical"), "EURUSD,USDCHF");
        historical.strategy.startDate = defaults.startDate;
        historical.strategy.endDate = "2019-12-31";
        long firstDay = (Long) call(parseDate, null, "2017-06-07");
        long first2015 = (Long) call(parseDate, null, defaults.startDate);
        historical.firstTicks.put(Instrument.EURUSD, firstDay + 12 * 3600000L);
        historical.firstTicks.put(Instrument.USDCHF, first2015);
        historical.start();
        check(historical.starts.get(0) == firstDay, "Premier jour partiel conserve, annees 2015 et 2016 ignorees");
        check(((Integer) field(historical.strategy, "unavailableYears")) == 2,
                "Les annees indisponibles sont comptees par paire");
        check(((Integer) field(historical.strategy, "queuedDays")) == (START - firstDay + START - first2015) / 86400000L,
                "Tous les jours disponibles sont planifies, y compris 2016 bissextile");
        Deque<?> jobs = (Deque<?>) field(historical.strategy, "pendingJobs");
        boolean usdStarts2015 = false, eurEnds2019 = false;
        for (Object job : jobs) {
            Instrument instrument = (Instrument) field(job, "instrument");
            long from = (Long) field(job, "startInclusiveMs");
            long to = (Long) field(job, "endExclusiveMs");
            check(from >= first2015 && to <= START, "Aucun jour hors 2015-2019");
            if (instrument == Instrument.EURUSD) {
                check(from >= firstDay, "Aucun fichier EURUSD avant le premier tick disponible");
                eurEnds2019 |= to == START;
            } else {
                usdStarts2015 |= from == first2015;
            }
        }
        check(usdStarts2015 && eurEnds2019, "Disponibilite par paire et dernier jour inclus");
        String historicalLog = historical.log.toString("UTF-8");
        check(historicalLog.contains("year=2015 reason=no_tick_history")
                && historicalLog.contains("year=2016 reason=no_tick_history")
                && historicalLog.contains("first_available_day=2017-06-07"), "Annees ignorees et annee partielle journalisees");
        historical.strategy.onStop();

        Platform unavailable = new Platform(root.resolve("unavailable"), "EURUSD,USDCHF");
        unavailable.strategy.startDate = defaults.startDate;
        unavailable.strategy.endDate = "2019-12-31";
        unavailable.firstTicks.put(Instrument.EURUSD, Long.MAX_VALUE);
        unavailable.firstTicks.put(Instrument.USDCHF, START);
        unavailable.start();
        check(unavailable.ticks.isEmpty() && unavailable.stops == 1,
                "Aucun historique ou historique commencant en 2020 : aucune lecture ni blocage");
        check(((Integer) field(unavailable.strategy, "unavailableYears")) == 10,
                "Les cinq annees absentes sont ignorees pour chaque paire");
        check(!Files.exists(root.resolve("unavailable/EURUSD")) && !Files.exists(root.resolve("unavailable/USDCHF")),
                "Aucun faux fichier vide cree pour une annee indisponible");

        Platform yen = new Platform(root.resolve("yen"), defaults.instrumentsCsv);
        yen.start();
        yen.quote(0, START + 1, 150., 150.01);
        yen.finish(0, true);
        check(yen.stops == 1 && yen.instruments.equals(Collections.singletonList(Instrument.USDJPY)),
                "Export USDJPY seul termine");
        check(((Long) call(VALIDATE, null, target(root.resolve("yen"), "USDJPY"), "USDJPY", START)) == 1,
                "Cotation USDJPY publiee avec le vrai symbole");

        // Plusieurs producteurs: aucune trame melangee ni sequence perdue.
        Path concurrent = root.resolve("concurrent.tsv");
        Object writer = CONSTRUCTOR.newInstance(concurrent, Instrument.EURUSD, "session", START);
        ExecutorService pool = Executors.newFixedThreadPool(8);
        List<Future<?>> workers = new ArrayList<>();
        for (int thread = 0; thread < 8; thread++) {
            workers.add(pool.submit(() -> {
                try {
                    for (int tick = 0; tick < 500; tick++) { call(WRITE, writer, START + 1, 1.1, 1.2, 2.0, 3.0); }
                } catch (Exception error) { throw new RuntimeException(error); }
            }));
        }
        for (Future<?> worker : workers) { worker.get(); }
        pool.shutdown();
        call(FINISH, writer, STOP);
        check(((Long) call(VALIDATE, null, concurrent, "EURUSD", START)) == 4000L, "4000 trames completes");
        expectIOException(() -> call(WRITE, writer, START, 1.1, 1.2, 2.0, 3.0));

        double[][] invalid = {{0, 1.2, 2, 3}, {1.2, 1.1, 2, 3}, {Double.NaN, 1.2, 2, 3},
                {1.1, Double.POSITIVE_INFINITY, 2, 3}, {1.1, 1.2, -1, 3}, {1.1, 1.2, 2, Double.NaN}};
        for (int i = 0; i < invalid.length; i++) {
            Path path = root.resolve("invalid-" + i + ".tsv");
            Object bad = CONSTRUCTOR.newInstance(path, Instrument.EURUSD, "session", START);
            double[] v = invalid[i];
            expectIOException(() -> call(WRITE, bad, START, v[0], v[1], v[2], v[3]));
            expectIOException(() -> call(FINISH, bad, STOP));
            call(ABORT, bad);
            check(!Files.exists(path) && !Files.exists(Paths.get(path + ".part")), "Pas de jour invalide publie");
        }
        for (long time : new long[]{START - 1, STOP + 1}) {
            Object bad = CONSTRUCTOR.newInstance(root.resolve("boundary-" + time + ".tsv"),
                    Instrument.EURUSD, "session", START);
            expectIOException(() -> call(WRITE, bad, time, 1.1, 1.2, 2.0, 3.0));
            call(ABORT, bad);
        }
        Object backwards = CONSTRUCTOR.newInstance(root.resolve("backwards.tsv"), Instrument.EURUSD, "session", START);
        call(WRITE, backwards, START + 2, 1.1, 1.2, 2.0, 3.0);
        expectIOException(() -> call(WRITE, backwards, START + 1, 1.1, 1.2, 2.0, 3.0));
        call(ABORT, backwards);

        String valid = fixture();
        String[] corrupted = {"", valid.substring(0, valid.length() - 1),
                valid.replace("QUOTE\tEURUSD", "QUOTE\tGBPUSD"), valid.replace("\tsession\t1\t1.1", "\tother\t1\t1.1"),
                valid.replace("\tsession\t1\t1.1", "\tsession\t2\t1.1"),
                valid.substring(valid.indexOf('\n') + 1), valid.substring(0, valid.lastIndexOf("STOP")),
                valid + valid.substring(0, valid.indexOf('\n') + 1), valid.replace("1.1\t1.2", "NaN\t1.2")};
        for (int i = 0; i < corrupted.length; i++) {
            Path path = root.resolve("corrupt-" + i + ".tsv");
            Files.write(path, corrupted[i].getBytes(StandardCharsets.UTF_8));
            expectIOException(() -> call(VALIDATE, null, path, "EURUSD", START));
        }
        Platform all = new Platform(root.resolve("all"), "USDCHF,GBPUSD,USDCAD,EURUSD");
        all.start();
        for (int i = 0; i < 4; i++) { all.quote(i, START + 1, 1.1, 1.2); all.finish(i, true); }
        check(all.stops == 1 && all.ticks.size() == 4, "Export des quatre paires termine");
        for (Instrument instrument : all.instruments) {
            String symbol = instrument.toString().replace("/", "");
            check(((Long) call(VALIDATE, null, target(root.resolve("all"), symbol), symbol, START)) == 1,
                    "Publication valide de " + symbol);
        }
        Platform resume = new Platform(root.resolve("all"), "USDCHF,GBPUSD,USDCAD,EURUSD");
        resume.start();
        check(resume.ticks.isEmpty() && resume.stops == 1, "Reprise ignore uniquement les jours valides");

        Platform incomplete = new Platform(root.resolve("incomplete"), "EURUSD");
        incomplete.start(); incomplete.quote(0, START, 1.1, 1.2); incomplete.finish(0, false);
        check(incomplete.stops == 1 && !Files.exists(target(root.resolve("incomplete"), "EURUSD")),
                "Chargement incomplet jamais publie");

        Platform stopped = new Platform(root.resolve("stopped"), "EURUSD,USDCHF");
        stopped.start(); stopped.quote(0, START, 1.1, 1.2); stopped.strategy.onStop();
        stopped.quote(0, START + 1, 1.1, 1.2); stopped.finish(0, true);
        check(stopped.progress.get(0).stopJob() && stopped.ticks.size() == 1
                && !Files.exists(target(root.resolve("stopped"), "EURUSD")), "Callbacks apres arret ignores");

        Platform brokenTick = new Platform(root.resolve("broken-tick"), "EURUSD,USDCHF");
        brokenTick.start(); brokenTick.quote(0, START, 1.2, 1.1); brokenTick.finish(0, true);
        check(brokenTick.stops == 1 && brokenTick.ticks.size() == 1
                && !Files.exists(target(root.resolve("broken-tick"), "EURUSD")), "Tick invalide arrete immediatement");

        Path repairRoot = root.resolve("repair");
        Path repairTarget = target(repairRoot, "EURUSD");
        Files.createDirectories(repairTarget.getParent());
        Files.write(repairTarget, "ancien fichier corrompu".getBytes(StandardCharsets.UTF_8));
        Platform repair = new Platform(repairRoot, "EURUSD");
        repair.start();
        check(repair.ticks.size() == 1 && new String(Files.readAllBytes(repairTarget), StandardCharsets.UTF_8)
                .equals("ancien fichier corrompu"), "Jour corrompu planifie sans suppression");
        repair.quote(0, START, 1.1, 1.2); repair.finish(0, true);
        check(((Long) call(VALIDATE, null, repairTarget, "EURUSD", START)) == 1, "Jour corrompu remplace apres validation");

        byte[] previousDay = Files.readAllBytes(repairTarget);
        Platform failedOverwrite = new Platform(repairRoot, "EURUSD");
        failedOverwrite.strategy.overwriteExistingDays = true;
        failedOverwrite.start(); failedOverwrite.quote(0, START, 1.1, 1.2); failedOverwrite.finish(0, false);
        check(Arrays.equals(previousDay, Files.readAllBytes(repairTarget)), "Echec preserve le jour deja publie");
        check(!Files.exists(Paths.get(repairTarget + ".part")), "Echec nettoie le temporaire");

        Platform wrongInstrument = new Platform(root.resolve("wrong-instrument"), "EURUSD");
        wrongInstrument.start();
        wrongInstrument.ticks.get(0).newTick(Instrument.USDCHF, START, 1.2, 1.1, 3, 2);
        check(wrongInstrument.stops == 1 && !Files.exists(target(root.resolve("wrong-instrument"), "EURUSD")),
                "Un instrument incorrect n'est jamais ecrit");

        Platform first = new Platform(root.resolve("lock"), "EURUSD");
        Platform second = new Platform(root.resolve("lock"), "EURUSD");
        first.start();
        try { second.start(); throw new AssertionError("Deux exports dans le meme dossier"); }
        catch (JFException expected) { checks++; }
        first.finish(0, true); // Jour vide legitime, par exemple marche ferme.
        check(((Long) call(VALIDATE, null, target(root.resolve("lock"), "EURUSD"), "EURUSD", START)) == 0,
                "Jour vide complet et premier export intact");
        System.out.println("JForex export: " + checks + " checks passed");
    }
}
