package jforex;

import com.dukascopy.api.*;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/** Export local JForex 4; démarrer depuis le client JForex (démo ou réel). */
@RequiresFullAccess
public class DukascopyTickBridge implements IStrategy {
    private static final String CONTROL_SYMBOL = "SYSTEM";
    private static final String OUTPUT_FILE_NAME = "live_ticks.tsv";

    @Configurable("Instruments CSV")
    public String instrumentsCsv = "BTCUSD,ETHUSD,EURGBP,EURCHF,EURUSD,EURCAD,EURAUD,EURNZD,USDCHF,USDCAD,GBPUSD,GBPCAD,GBPCHF,GBPAUD,GBPNZD";
    @Configurable("Dossier des ticks")
    public String outputDirectory = System.getProperty("user.home");

    private final Object lock = new Object();
    private final String session = UUID.randomUUID().toString();
    private BufferedWriter writer;
    private ScheduledExecutorService heartbeat;
    private long sequence = 0;
    private Set<Instrument> subscribedInstruments = new LinkedHashSet<Instrument>();
    private volatile boolean failed = false;
    private IConsole console;

    @Override
    public void onStart(IContext context) throws JFException {
        console = context.getConsole();
        try {
            Set<Instrument> requestedInstruments = parseInstruments(instrumentsCsv);
            subscribedInstruments = subscribeAndFilterInstruments(context, requestedInstruments);

            Path folder = Paths.get(outputDirectory);
            Files.createDirectories(folder);
            writer = Files.newBufferedWriter(folder.resolve(OUTPUT_FILE_NAME),
                    StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
            write("START", CONTROL_SYMBOL, System.currentTimeMillis(), 0, 0, 0, 0);
            console.getOut().println(String.format(
                    Locale.US,
                    "[bridge] subscribed=%d output=%s instruments=%s",
                    subscribedInstruments.size(),
                    folder.resolve(OUTPUT_FILE_NAME),
                    subscribedInstruments));

            heartbeat = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread thread = new Thread(r, "dukascopy-bridge-heartbeat");
                thread.setDaemon(true);
                return thread;
            });
            heartbeat.scheduleAtFixedRate(() -> {
                try {
                    write("HEARTBEAT", CONTROL_SYMBOL, System.currentTimeMillis(), 0, 0, 0, 0);
                } catch (IOException error) {
                    failed = true;
                }
            }, 0, 5, TimeUnit.SECONDS);
        } catch (Exception error) {
            throw new JFException("Impossible de démarrer le pont : " + error.getMessage());
        }
    }

    private Set<Instrument> subscribeAndFilterInstruments(IContext context, Set<Instrument> requestedInstruments) {
        context.setSubscribedInstruments(requestedInstruments, true);

        LinkedHashSet<Instrument> actualInstruments = new LinkedHashSet<Instrument>(context.getSubscribedInstruments());
        actualInstruments.retainAll(requestedInstruments);

        LinkedHashSet<String> blockedCrypto = new LinkedHashSet<String>();
        LinkedHashSet<String> blockedOther = new LinkedHashSet<String>();
        for (Instrument instrument : requestedInstruments) {
            if (actualInstruments.contains(instrument)) {
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
            console.getErr().println("[bridge] cryptos non abonnees: " + blockedCrypto
                    + ". Accepter d'abord la \"Cryptocurrencies Disclaimer\" dans JForex puis relancer la strategie.");
        }
        if (!blockedOther.isEmpty()) {
            console.getErr().println("[bridge] instruments refuses ou indisponibles apres tentative d'abonnement: " + blockedOther);
        }
        if (actualInstruments.isEmpty()) {
            throw new IllegalArgumentException("Aucun instrument effectivement abonne par JForex");
        }
        return actualInstruments;
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
            console.getErr().println("[bridge] instruments ignores car indisponibles ou inconnus dans ce JForex: " + skippedInstruments);
        }
        return instruments;
    }

    private static boolean isCryptoInstrument(String symbol) {
        return "BTC/USD".equals(symbol) || "ETH/USD".equals(symbol) || "XRP/USD".equals(symbol);
    }

    private static String symbolText(Instrument instrument) {
        return instrument.toString().replace("/", "");
    }

    private void write(String type, String symbol, long timestamp, double bid, double ask,
                       double bidSize, double askSize) throws IOException {
        synchronized (lock) {
            if (writer == null) throw new IOException("Pont fermé");
            writer.write(type + "\t" + symbol + "\t" + timestamp + "\t" + session + "\t"
                    + sequence + "\t" + bid + "\t" + ask + "\t" + bidSize + "\t" + askSize);
            writer.newLine();
            writer.flush();
        }
    }

    @Override
    public void onTick(Instrument instrument, ITick tick) throws JFException {
        if (!subscribedInstruments.contains(instrument)) return;
        if (failed) throw new JFException("Écriture du pont JForex interrompue");
        try {
            synchronized (lock) {
                sequence++;
                write("QUOTE", symbolText(instrument), tick.getTime(), tick.getBid(), tick.getAsk(),
                      tick.getBidVolume(), tick.getAskVolume());
            }
        } catch (IOException error) {
            failed = true;
            throw new JFException("Écriture tick JForex impossible : " + error.getMessage());
        }
    }

    @Override
    public void onStop() throws JFException {
        if (heartbeat != null) heartbeat.shutdownNow();
        synchronized (lock) {
            if (writer == null) return;
            try {
                write("STOP", CONTROL_SYMBOL, System.currentTimeMillis(), 0, 0, 0, 0);
                writer.close();
                writer = null;
            } catch (IOException error) {
                throw new JFException("Arrêt du pont JForex : " + error.getMessage());
            }
        }
    }

    @Override public void onAccount(IAccount account) throws JFException {}
    @Override public void onMessage(IMessage message) throws JFException {}
    @Override public void onBar(Instrument instrument, Period period, IBar askBar, IBar bidBar)
            throws JFException {}
}
