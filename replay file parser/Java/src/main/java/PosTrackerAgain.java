/**
 * Dota 2 replay file parser for use with the turtling hero predictor AI.
 * 
 * heroes.json must be located in the same folder as your compiled executable
 * https://api.opendota.com/api/constants/heroes
 * 
 * give the path to your demo file in argv
 * 
 * Written by jasper law using clarity.
 * 
 */

import skadistats.clarity.event.Insert;
import skadistats.clarity.model.Entity;
import skadistats.clarity.model.FieldPath;
import skadistats.clarity.model.StringTable;
import skadistats.clarity.processor.entities.Entities;
import skadistats.clarity.processor.entities.OnEntityPropertyChanged;
import skadistats.clarity.processor.entities.UsesEntities;
import skadistats.clarity.processor.reader.OnTickStart;
import skadistats.clarity.processor.runner.Context;
import skadistats.clarity.processor.runner.SimpleRunner;
import skadistats.clarity.source.MappedFileSource;
import skadistats.clarity.util.Predicate;
import skadistats.clarity.wire.common.proto.DotaUserMessages.DOTA_COMBATLOG_TYPES;
import skadistats.clarity.model.CombatLogEntry;
import skadistats.clarity.processor.gameevents.OnCombatLogEntry;
import skadistats.clarity.processor.stringtables.OnStringTableCreated;
import skadistats.clarity.processor.stringtables.UsesStringTable;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Scanner;
import org.apache.commons.text.WordUtils;

import utils.Util;
import java.util.HashMap;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
// import HeroStats;

import it.unimi.dsi.fastutil.Hash;

@UsesEntities
public class PosTrackerAgain {

    /**
     * Holds a list of strings representing lines in a csv table
     * 
     */
    public class CSVHolder {
        // The lines
        public LinkedList<String> data;

        public CSVHolder() {
            data = new LinkedList<String>();
        }

        /**
         * Add a new line. Don't end it with a carriage return, that's done by compile().
         * Values must already be comma separated
         * 
         * @param d the line to add
         */
        public void add(String d) {
            data.add(d);
        }

        /**
         * Add a new line. Don' t end it with a carriage return, that's done by compile().
         * Takes a list of values joins them into a single string (line), separated by commas.
         * 
         * @param d list of values to add, constituting a single line 
         */
        public void add(LinkedList<String> d) {
            data.add(String.join(",", d));
        }

        /**
         * Join all lines with carriage return characters and return the whole dataset as a string
         * 
         * @return the whole dataset as a single string with lines separated by carriage returns
         */
        public String compile() {
            return String.join("\n", data);
        }
    }

    // The tick at which the pick/ban phase begins
    int gameStartTick;
    // The tick at which the horn blows (time in phase == 0)
    int hornTick;
    // A record associating engine hero names (not source hero names/CDOTA...) with their hero IDs
    private HashMap<String, Integer> heroNamesIDs;

    @Insert
    // State of the game at the current tick
    private Context ctx;
    // File reader
    private SimpleRunner runner;
    // Util functions - instanced as some are not static
    private Util util;

    // Stat trackers for each hero in the match
    private HashMap<Entity, HeroStats> heroStats;

    private Util.FloatTriple[] radiantTowers;
    private Util.FloatTriple[] direTowers;
    private Util.FloatTriple radiantFountain;
    private Util.FloatTriple direFountain;
    private HashMap<String, Integer> heroNameIDOverrides;
    private String fname;
    private final int TICK_RATE = 30;

    @UsesEntities
    @OnTickStart()
    /**
     * No functionality currently
     * 
     * @param ctx
     * @param synthetic
     */
    public void onTickStart(Context ctx, boolean synthetic) {
        // Here debugger can get all the objects in the game
        ctx.getProcessor(Entities.class);
    }

    
    /**
     * Instance this class. Some initialization (e.g hero gathering) is performed whilst running,
     * so on its own this object has little purpose straight away after construction
     * 
     * @param fileName Path to the demo to read
     * @throws IOException When failing to read the demo file or heroes.json
     */
    public PosTrackerAgain(String fileName) throws IOException {
        // Start ticks currently unknown
        gameStartTick = -1;
        hornTick = -1;
        // Empty class instances to be populated
        util = new Util();
        heroNamesIDs = new HashMap<String, Integer>();
        heroStats = new HashMap<Entity, HeroStats>();

        radiantTowers = new Util.FloatTriple[11];
        direTowers = new Util.FloatTriple[11];
        radiantFountain = null;
        direFountain = null;

        heroNameIDOverrides = new HashMap<String,Integer>();
        heroNameIDOverrides.put("npc_dota_hero_sand_king", 17);
        heroNameIDOverrides.put("npc_dota_hero_antimage", 2);
        heroNameIDOverrides.put("npc_dota_hero_earthshaker", 8);
        heroNameIDOverrides.put("npc_dota_hero_lich", 26);
        heroNameIDOverrides.put("npc_dota_hero_ancient_apparition", 69);
        heroNameIDOverrides.put("npc_dota_hero_drow_ranger", 7);
        heroNameIDOverrides.put("npc_dota_hero_oracle", 109);
        heroNameIDOverrides.put("npc_dota_hero_sven", 19);
        heroNameIDOverrides.put("npc_dota_hero_crystal_maiden", 6);
        heroNameIDOverrides.put("npc_dota_hero_ember_spirit", 105);
        heroNameIDOverrides.put("npc_dota_hero_slardar", 29);
        heroNameIDOverrides.put("npc_dota_hero_vengefulspirit", 21);
        heroNameIDOverrides.put("npc_dota_hero_abyssal_underlord", 114);
        heroNameIDOverrides.put("npc_dota_hero_pudge", 15);
        heroNameIDOverrides.put("npc_dota_hero_shadow_shaman", 28);
        heroNameIDOverrides.put("npc_dota_hero_razor", 16);
        heroNameIDOverrides.put("npc_dota_hero_juggernaut", 9);
        heroNameIDOverrides.put("npc_dota_hero_kunkka", 24);
        heroNameIDOverrides.put("npc_dota_hero_axe", 3);
        heroNameIDOverrides.put("npc_dota_hero_monkey_king", 115);
        heroNameIDOverrides.put("npc_dota_hero_zuus", 23);
        heroNameIDOverrides.put("npc_dota_hero_puck", 14);
        heroNameIDOverrides.put("npc_dota_hero_witch_doctor", 31);
        fname = fileName.split("/")[fileName.split("/").length-1].split(".dem")[0];

        // Read in hero constants
        Scanner fReader = new Scanner(new File("heroes.json")).useDelimiter("\\Z");
        String heroesData = fReader.next();
        fReader.close();
        Map<String, Object> heroesJson = new Gson().fromJson(
            heroesData, new TypeToken<HashMap<String, HashMap<String, Object>>>() {}.getType());
        
        // Associate each hero name with their ID
        for (String id: heroesJson.keySet()) {
            heroNamesIDs.put((String) ((Map<String, Object>) heroesJson.get(id)).get("name"),
                                Integer.parseInt(id) - 1); // heroes.json is 1 indexed for hero IDs, clarity is 0 indexed
        }

        // Read the demo file
        Path pathToFile = Paths.get(fileName);
        runner = new SimpleRunner(new MappedFileSource(pathToFile.toAbsolutePath()));
    }

    /**
     * Main func, parses a demo file.
     * 
     * @param args ignored
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        File file = new File(args[0]);
        if (!file.exists()) {
            throw new Error("Given path does not exist");
        } else if (file.isFile()) {
            if (args[0].endsWith(".dem")) {
                new PosTrackerAgain(args[0]).run();
            } else {
                throw new Error("Given path does not point to a .dem file");
            }
        } else if (file.isDirectory()) {
            File[] files = file.listFiles((d, name) -> name.endsWith(".dem"));
            if (files.length == 0) {
                System.out.println("Given a directory that contains no .dem files");
            } else {
                // System.out.println("Parse " + files.length + " files? (y/n)");
                // BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
                // if (!reader.readLine().toLowerCase().equals("y")) {
                //     System.out.println("Batch parse cancelled.");
                // } else {
                //     for (File currentDemo : files) {
                //         new PosTrackerAgain(currentDemo.getAbsolutePath()).run();
                //     }
                // }
                System.out.println("Parsing " + files.length + " files");
                for (File currentDemo : files) {
                    System.out.println("Parsing " + currentDemo.getPath());
                    new PosTrackerAgain(currentDemo.getAbsolutePath()).run();
                }
            }
        } else {
            throw new Error("Unrecognised path type");
        }
    }

    /**
     * Predicate deciding if an Entity is a hero or not based on their source name
     * 
     */
    private class IsHeroPred implements Predicate<Entity> {
        public boolean apply(Entity value) {
            return value.getDtClass().getDtName().startsWith("CDOTA_Unit_Hero_");
        }
    }


    private class IsTowerPred implements Predicate<Entity> {
        public boolean apply(Entity value) {
            return value.getDtClass().getDtName().equals("CDOTA_BaseNPC_Tower");
        }
    }


    private class IsFountainPred implements Predicate<Entity> {
        public boolean apply(Entity value) {
            return value.getDtClass().getDtName().equals("CDOTA_Unit_Fountain");
        }
    }


    /**
     * Entity objects differ across different Contexts (ticks)
     * This function searches the tracked Entity objects for one of the same name, allowing the stat tracker
     * for a given hero can be accessed at any tick
     * 
     * @param ent hero whose equivilent to look up
     * @return an Entity with the same dtname as ent
     * @throws Error if no stored entity could be found
     */
    private Entity getStoredHero(Entity ent) {
        for (Entity e: heroStats.keySet()) {
            if (e.getDtClass().getDtName().equals(ent.getDtClass().getDtName())) {
                return e;
            }
        }
        throw new Error("Failed to find stored hero for " + ent.getDtClass().getDtName());
    }


    /**
     * Searches the tracked Entity objects for one of the given hero ID
     * 
     * @param id hero ID whose Entity to look up
     * @return an Entity with the given hero ID
     * @throws Error if no stored entity could be found
     */
    private Entity getStoredHeroByID(Integer id) {
        for (Entity e: heroStats.keySet()) {
            if (((Integer) e.getProperty("m_iUnitNameIndex")).equals(id)) {
                // System.out.println("Found hero: " + e.getDtClass().getDtName());
                return e;
            }
        }
        throw new Error("Failed to find stored hero with ID " + id.toString());
    }


    /**
     * Searches the tracked Entity objects for one with the given dtName
     * 
     * @param dtName hero dtName whose Entity to look up
     * @return an Entity with the given dtName
     * @throws Error if no stored entity could be found
     */
    private Entity getStoredHeroByDTName(String dtName) {
        for (Entity e: heroStats.keySet()) {
            if (e.getDtClass().getDtName().replace("_", "").toLowerCase().equals(dtName.replace("_", "").toLowerCase())) {
                // System.out.println("Found hero: " + e.getDtClass().getDtName());
                return e;
            }
        }
        throw new Error("Failed to find stored hero with dtName " + dtName);
    }


    private String npcToCDota(String npcName) {
        return "CDOTA_Unit_Hero_" + WordUtils.capitalizeFully(npcName.substring(14).replace('_', ' ')).replace(" ", "");
    }


    /**
     * Get the stored hero which corresponds to the player referenced by a data spectator.
     * 
     * @param e An instance of CDOTA_DataSpectator, referencing a player whose hero to look up in heroStats
     * @param fp The path to the hero ID property on e
     * @return The hero being controlled by the player referenced in e
     * @throws Error if no corresponding hero could be found
     */
    private Entity heroFromDataSpectator(Entity e, FieldPath fp) {
        Integer playerID = Integer.valueOf(e.getDtClass().getNameForFieldPath(fp).substring(15));
        for (Entity heroEnt: heroStats.keySet()) {
            if (heroEnt.getProperty("m_iPlayerID").equals(playerID)) {
                return heroEnt;
                // heroNamesIDs.get(cle.getAttackerName())
            }
        }
        throw new Error("Unable to find hero for player " + playerID.toString());
    }


    /**
     * Parse the demo, construct csv from the recorded data, and export to file.
     * 
     */
    public void run() throws IOException, InterruptedException {
        // Parse data
        runner.runWith(this);
        
        boolean debugItems = false;
        // CSV data organiser
        CSVHolder data = new CSVHolder();
        // Current line - data for all heroes for the current tick
        LinkedList<String> line = new LinkedList<String>();
        // Is this the first tick we've written?
        boolean first = true;
        // // The last written positions, used in calculating position deltas
        // HashMap<Entity, Util.FloatTriple> lastCoords = new HashMap<Entity, Util.FloatTriple>();

        // Add header row wih column names
        line.add("tick");
        // line.add("mapControlRadiant");
        // line.add("mapControlDire");
        for (int i = 0; i < 10; i++) {
            line.add("heroID" + Integer.toString(i));
            line.add("heroName" + Integer.toString(i));
            line.add("heroTeam" + Integer.toString(i));
            line.add("posX" + Integer.toString(i));
            line.add("posXPerSecond" + Integer.toString(i));
            line.add("posY" + Integer.toString(i));
            line.add("posYPerSecond" + Integer.toString(i));
            line.add("posZ" + Integer.toString(i));
            line.add("posZPerSecond" + Integer.toString(i));
            line.add("netWorth" + Integer.toString(i));
            line.add("netWorthPerSecond" + Integer.toString(i));
            line.add("XP" + Integer.toString(i));
            line.add("XPPerSecond" + Integer.toString(i));
            line.add("kills" + Integer.toString(i));
            line.add("deaths" + Integer.toString(i));
            line.add("lastHits" + Integer.toString(i));
            line.add("lastHitsPerSecond" + Integer.toString(i));
            line.add("closestFriendlyHeroDist" + Integer.toString(i));
            line.add("closestEnemyHeroDist" + Integer.toString(i));
            line.add("closestFriendlyTowerDist" + Integer.toString(i));
            line.add("closestEnemyTowerDist" + Integer.toString(i));
        }
        for (int j = 0; j < 10; j++) {
            line.add("isTurtling" + Integer.toString(j));
        }
        data.add(line);

        // Need a list of ticks to record to file, but recorded ticks may differ from hero to hero
        // So, pick the hero with the most ticks and use their ticks. May result in *marginally* lower resolution data.
        Iterator<Entity> heroIterator = heroStats.keySet().iterator();
        Entity driverHero = heroIterator.next();
        while (heroIterator.hasNext()) {
            Entity nextHero = heroIterator.next();
            if (heroStats.get(nextHero).ticks.size() > heroStats.get(driverHero).ticks.size()) {
                driverHero = nextHero;
            }
        }
        HashMap<Entity, HashMap<Entity, Float>> radiantHeroDistances = new HashMap<Entity, HashMap<Entity, Float>>();
        HashMap<Entity, HashMap<Entity, Float>> direHeroDistances = new HashMap<Entity, HashMap<Entity, Float>>();

        for (Entity currentHero: heroStats.keySet()) {
            if (currentHero.getProperty("m_iTeamNum").equals(Integer.valueOf(3))) {
                radiantHeroDistances.put(currentHero, new HashMap<Entity, Float>());
            } else {
                direHeroDistances.put(currentHero, new HashMap<Entity, Float>());
            }
        }

        System.out.println("Driver hero " + driverHero.getDtClass().getDtName() + " has " + heroStats.get(driverHero).ticks.get(HeroStats.StatType.coords).size() + " ticks.");
        
        // Ticks may also vary from stat to stat, coords is the one that will change the most though so lets use that
        for (Integer tick: heroStats.get(driverHero).ticks.get(HeroStats.StatType.coords)) {
        // Integer tick = (Integer) heroStats.get(driverHero).ticks.get(HeroStats.StatType.coords).toArray()[0];
            // Make a new line
            line.clear();
            // Add the current tick
            line.add(tick.toString());

            // Calculate all distances between heroes
            for (Entity currentHero: radiantHeroDistances.keySet()) {
                radiantHeroDistances.get(currentHero).clear();
            }
            for (Entity currentHero: direHeroDistances.keySet()) {
                direHeroDistances.get(currentHero).clear();
            }
            for (Entity currentHero: heroStats.keySet()) {
                HeroStats currentHeroStats = heroStats.get(currentHero);
                if (currentHero.getProperty("m_iTeamNum").equals(Integer.valueOf(3))) { // radiant
                    for (Entity otherHero: heroStats.keySet()) {
                        if (currentHero != otherHero) {
                            HeroStats otherHeroStats = heroStats.get(otherHero);
                            radiantHeroDistances.get(currentHero).put(otherHero, currentHeroStats.getCoords(tick).distanceTo(otherHeroStats.getCoords(tick)));
                        }
                    }
                } else { // dire
                    for (Entity otherHero: heroStats.keySet()) {
                        if (currentHero != otherHero) {
                            HeroStats otherHeroStats = heroStats.get(otherHero);
                            direHeroDistances.get(currentHero).put(otherHero, currentHeroStats.getCoords(tick).distanceTo(otherHeroStats.getCoords(tick)));
                        }
                    }
                }
            }

            // Infer level of map control as the 2D area between all heroes on the team
            Util.FloatTriple[] heroCoords = new Util.FloatTriple[5];
            int heroNum = 0;
            for (Entity currentHero: radiantHeroDistances.keySet()) {
                heroCoords[heroNum] = heroStats.get(currentHero).getCoords(tick);
                heroNum++;
            }
            // Scale map control estimation by distance from polygon centroid to enemy fountain
            // line.add((debugItems ? "radiantMapControl " : "") + String.valueOf(Util.polygonArea(heroCoords) / util.polygonCentroid(heroCoords).distanceTo(direFountain)));

            heroNum = 0;
            for (Entity currentHero: direHeroDistances.keySet()) {
                heroCoords[heroNum] = heroStats.get(currentHero).getCoords(tick);
                heroNum++;
            }
            // line.add((debugItems ? "direMapControl " : "") + String.valueOf(Util.polygonArea(heroCoords) / util.polygonCentroid(heroCoords).distanceTo(radiantFountain)));

            // Add data in this tick for all heroes
            for (Entity currentHero: heroStats.keySet()) {
                HeroStats currentHeroStats = heroStats.get(currentHero);
                // Add hero ID
                line.add((debugItems ? "heroID " : "") + ((Integer) currentHero.getProperty("m_iUnitNameIndex")).toString());
                // Add hero name
                line.add((debugItems ? "heroName " : "") + currentHero.getDtClass().getDtName());
                // Add hero team
                line.add((debugItems ? "heroTeam " : "") + (currentHero.getProperty("m_iTeamNum").equals(Integer.valueOf(3)) ? "dire" : "radiant"));
                // Add current position
                line.add((debugItems ? "heroPosition " : "") + currentHeroStats.getCoords(tick).toString());
                // Write the change in position since the previous tick
                if (first) {
                    // There will be no previous coords for the first tick, so write 
                    line.add((debugItems ? "firstTickPosChange " : "") + util.new FloatTriple(0f, 0f, 0f).toString());
                } else {
                    // Add current rate of change in x position
                    Util.FloatTriple posChange = currentHeroStats.getCoords(tick);
                    int closestTick = Math.max(tick - 2 * TICK_RATE, 0);
                    posChange.minus(currentHeroStats.getCoords(closestTick));
                    posChange.divide(util.new FloatTriple((float) 2f, (float) 2f, (float) 2f));
                    line.add((debugItems ? "posXPerSecond " : "") + String.valueOf(posChange.x));
                    line.add((debugItems ? "posYPerSecond " : "") + String.valueOf(posChange.y));
                    line.add((debugItems ? "posZPerSecond " : "") + String.valueOf(posChange.z));
                }
                int closestTick = Math.max(tick - 10 * TICK_RATE, 0);
                // // Save the current coords for comparison in the next tick
                // lastCoords.put(currentHero, currentHeroStats.getCoords(tick));
                // Add current net worth
                line.add((debugItems ? "netWorth " : "") + currentHeroStats.getNetWorth(tick).toString());
                // Add current rate of change in net worth
                line.add((debugItems ? "netWorthPerSecond " : "") + String.valueOf(((float) (currentHeroStats.getNetWorth(tick) - currentHeroStats.getNetWorth(closestTick)))/10f));
                // Add current XP
                line.add((debugItems ? "XP " : "") + currentHeroStats.getXP(tick).toString());
                // Add current rate of change in XP
                line.add((debugItems ? "XPPerSecond " : "") + String.valueOf(((float) (currentHeroStats.getXP(tick) - currentHeroStats.getXP(closestTick)))/10f));
                // Add current kills
                line.add((debugItems ? "kills " : "") + currentHeroStats.getKills(tick).toString());
                // Add current deaths
                line.add((debugItems ? "deaths " : "") + currentHeroStats.getDeaths(tick).toString());
                // Add current lastHits
                line.add((debugItems ? "lastHits " : "") + currentHeroStats.getLastHits(tick).toString());
                // Add current rate of change in last hits
                line.add((debugItems ? "lastHitsPerSecond " : "") + String.valueOf(((float) (currentHeroStats.getLastHits(tick) - currentHeroStats.getLastHits(closestTick)))/10f));
                // Find closest hero distances
                Float closestFriendlyHeroDistance = Float.POSITIVE_INFINITY;
                Float closestEnemyHeroDistance = Float.POSITIVE_INFINITY;
                if (currentHero.getProperty("m_iTeamNum").equals(Integer.valueOf(3))) {
                    for (Entity otherHero: radiantHeroDistances.get(currentHero).keySet()) {
                        if (otherHero.getProperty("m_iTeamNum").equals(Integer.valueOf(3)) && currentHero != otherHero && radiantHeroDistances.get(currentHero).get(otherHero) < closestFriendlyHeroDistance) {
                            closestFriendlyHeroDistance = radiantHeroDistances.get(currentHero).get(otherHero);
                        }
                    }
                    for (Entity otherHero: radiantHeroDistances.get(currentHero).keySet()) {
                        if (otherHero.getProperty("m_iTeamNum").equals(Integer.valueOf(2)) && radiantHeroDistances.get(currentHero).get(otherHero) < closestEnemyHeroDistance) {
                            closestEnemyHeroDistance = radiantHeroDistances.get(currentHero).get(otherHero);
                        }
                    }
                } else { // Add friendly hero distances
                    for (Entity otherHero: direHeroDistances.get(currentHero).keySet()) {
                        if (otherHero.getProperty("m_iTeamNum").equals(Integer.valueOf(2)) && currentHero != otherHero && direHeroDistances.get(currentHero).get(otherHero) < closestFriendlyHeroDistance) {
                            closestFriendlyHeroDistance = direHeroDistances.get(currentHero).get(otherHero);
                        }
                    }
                    for (Entity otherHero: direHeroDistances.get(currentHero).keySet()) {
                        if (otherHero.getProperty("m_iTeamNum").equals(Integer.valueOf(3)) && direHeroDistances.get(currentHero).get(otherHero) < closestEnemyHeroDistance) {
                            closestEnemyHeroDistance = direHeroDistances.get(currentHero).get(otherHero);
                        }
                    }
                }
                // Write closest hero distances
                line.add((debugItems ? "closestFriendlyHeroDistance " : "") + closestFriendlyHeroDistance.toString());
                line.add((debugItems ? "closestEnemyHeroDistance " : "") + closestEnemyHeroDistance.toString());
                
                // // write all tower distances
                // if (currentHero.getProperty("m_iTeamNum").equals(Integer.valueOf(3))) {
                //     for (Util.FloatTriple towerPos: radiantTowers) {
                //         line.add((debugItems ? "friendlyTowerDist " : "") + String.valueOf(currentHeroStats.getCoords(tick).distanceTo(towerPos)));
                //     }
                //     for (Util.FloatTriple towerPos: direTowers) {
                //         line.add((debugItems ? "enemyTowerDist " : "") + String.valueOf(currentHeroStats.getCoords(tick).distanceTo(towerPos)));
                //     }
                // } else {
                //     for (Util.FloatTriple towerPos: direTowers) {
                //         line.add((debugItems ? "friendlyTowerDist " : "") + String.valueOf(currentHeroStats.getCoords(tick).distanceTo(towerPos)));
                //     }
                //     for (Util.FloatTriple towerPos: radiantTowers) {
                //         line.add((debugItems ? "enemyTowerDist " : "") + String.valueOf(currentHeroStats.getCoords(tick).distanceTo(towerPos)));
                //     }
                // }

                // Find closest radiant tower
                float radiantTowerDist = currentHeroStats.getCoords(tick).distanceTo(radiantTowers[0]);
                for (int towerNum = 1; towerNum < 11; towerNum++) {
                    float towerDist = currentHeroStats.getCoords(tick).distanceTo(radiantTowers[0]);
                    if (towerDist < radiantTowerDist) {
                        radiantTowerDist = towerDist;
                    }
                }
                // Find closest dire tower
                float direTowerDist = currentHeroStats.getCoords(tick).distanceTo(direTowers[0]);
                for (int towerNum = 1; towerNum < 11; towerNum++) {
                    float towerDist = currentHeroStats.getCoords(tick).distanceTo(direTowers[0]);
                    if (towerDist < direTowerDist) {
                        direTowerDist = towerDist;
                    }
                }
                // Write closest friendly tower first, then closest enemy tower
                if (currentHero.getProperty("m_iTeamNum").equals(Integer.valueOf(3))) {
                    line.add(String.valueOf(radiantTowerDist));
                    line.add(String.valueOf(direTowerDist));
                } else {
                    line.add(String.valueOf(direTowerDist));
                    line.add(String.valueOf(radiantTowerDist));
                }
            }
            // Add placeholders for isTurtling
            for (int j = 0; j < 10; j++) {
                line.add("0");
            }

            // Save the line
            data.add(line);
            // Update first tick indicator
            if (first) {
                first = false;
            }
        }
        
        // Write the recorded data to file
        saveData(data, fname + ".csv");
    }


    /**
     * Record when the game starts.
     * This is primary used for creating hero stat trackers, as no data can be recorded until all heroes are known.
     * 
     * @param context current state of the game
     * @param e the game rules entity
     * @param fp path to the property that changed
     * @throws Error when any number of heroes are in the game other than 10
     */
    @OnEntityPropertyChanged(classPattern = "CDOTAGamerulesProxy", propertyPattern = "m_pGameRules.m_flGameStartTime")
    public void onGameStart(Context context, Entity e, FieldPath fp) {
        // Ignore the game rule change if the start of the game has already been found
        if (hornTick == -1) {
            // Record the start of the demo file
            if (gameStartTick == -1) {
                gameStartTick = ctx.getTick();
                System.out.println("game started at tick " + ctx.getTick());
            // Record the horn blow
            } else {
                Iterator<Entity> fountains = runner.getContext().getProcessor(Entities.class).getAllByPredicate(new IsFountainPred());
                while (fountains.hasNext()) {
                    Entity f = fountains.next();
                    if (f.getProperty("m_iTeamNum").equals(Integer.valueOf(3))) {
                        radiantFountain = util.getPosition(f);
                    } else {
                        direFountain = util.getPosition(f);
                    }
                }
                if (direFountain == null) {
                    throw new Error("Couldnt find dire fountain");
                }
                if (radiantFountain == null) {
                    throw new Error("Couldnt find radiant fountain");
                }

                Iterator<Entity> towers = runner.getContext().getProcessor(Entities.class).getAllByPredicate(new IsTowerPred());
                int currentRadTower = 0;
                int currentDirTower = 0;
                if (towers.hasNext()) {
                    while (towers.hasNext()) {
                        Entity tower = towers.next();
                        if (tower.getProperty("m_iTeamNum").equals(Integer.valueOf(3))) {
                            radiantTowers[currentRadTower] = util.getPosition(tower);
                            currentRadTower++;
                        } else {
                            direTowers[currentDirTower] = util.getPosition(tower);
                            currentDirTower++;
                        }
                    }
                } else {
                    throw new Error("Unable to find tower entities");
                }

                hornTick = ctx.getTick();
                System.out.println("horn blew at tick " + ctx.getTick());

                // Once the horn has blown, all heroes must be in the game.
                // At this point, construct hero stat trackers.
                Iterator<Entity> allHeroes = runner.getContext().getProcessor(Entities.class).getAllByPredicate(new IsHeroPred());
                for (Iterator<Entity> iter = allHeroes; iter.hasNext(); ) {
                    // Create a stat tracker for each hero
                    Entity ent = iter.next();
                    // Make sure its not an illusion/clone
                    if (ent.hasProperty("m_iPlayerID") && !ent.getProperty("m_iIsControllableByPlayer64").equals(Long.valueOf(0))) {
                        heroStats.put(ent, new HeroStats(ent));
                        System.out.println("Player for " + ent.getDtClass().getDtName() + "(" + ent.getProperty("m_iUnitNameIndex") + "): " + ent.getProperty("m_iPlayerID"));
                    }
                }
                
                // Ensure exactly 10 heroes are in the game
                if (heroStats.size() != 10) {
                    throw new Error("incorrect number of heroes: " + Integer.toString(heroStats.size()));
                }
            }
        }
    }

    
    /**
     * Record changes in hero XP
     * 
     * @param context state of the game at the current tick
     * @param e the hero whose XP changed
     * @param fp path to the XP property on hero
     */
    @OnEntityPropertyChanged(classPattern = "CDOTA_Unit_Hero_.*", propertyPattern = "m_iCurrentXP")
    public void onGainXP(Context context, Entity e, FieldPath fp) {
        if (hornTick != -1 && e.hasProperty("m_iPlayerID")) {
            HeroStats stats = heroStats.get(getStoredHero(e));
            Integer newXP = (Integer) e.getPropertyForFieldPath(fp);
            try {
                if (newXP >= stats.getXP(ctx.getTick())) {
                    stats.addXP(ctx.getTick(), newXP);
                }
            } catch (Error ex) {
                stats.addXP(ctx.getTick(), newXP);
            }
        }
    }


    /**
     * Record changes in hero net worth 
     * 
     * @param context state of the game at the current tick
     * @param e the hero whose net worth changed
     * @param fp path to the net worth property on hero
     */
    @OnEntityPropertyChanged(classPattern = "CDOTA_DataSpectator", propertyPattern = "m_iNetWorth.*")
    public void onGainGold(Context context, Entity e, FieldPath fp) {
        if (hornTick != -1) {
            try {
                heroStats.get(heroFromDataSpectator(e, fp)).addNetWorth(ctx.getTick(), (Integer) e.getPropertyForFieldPath(fp));
            } catch (Error er) {
                System.out.println(er.toString());
            }
            
        }
    }


    /**
     * Record changes in hero position
     * 
     * @param context state of the game at the current tick
     * @param e the hero whose position changed
     * @param fp path to the position property on hero
     */
    @OnEntityPropertyChanged(classPattern = "CDOTA_Unit_Hero_.*", propertyPattern = "CBodyComponent.*")
    public void onPositionChange(Context context, Entity e, FieldPath fp) {
        // System.out.println(context.getTick() + " position change for " + e.getDtClass().getDtName());
        if (hornTick != -1 && e.hasProperty("m_iPlayerID")) {
            heroStats.get(getStoredHero(e)).addCoords(ctx.getTick(), util.getPosition(e));
        }
    }


    /**
     * Record hero kills and deaths
     * 
     * @param cle
     */
    @OnCombatLogEntry
    public void onCombatLogEntry(CombatLogEntry cle) {
        if (hornTick != -1 && cle.getType().equals(DOTA_COMBATLOG_TYPES.DOTA_COMBATLOG_DEATH)) {
            if (cle.getAttackerName().startsWith("npc_dota_hero") && (!cle.hasAttackerIllusion() || !cle.isAttackerIllusion()) && (!cle.hasTargetIllusion() || !cle.isTargetIllusion())) {
                // Integer heroID = heroNameIDOverrides.containsKey(cle.getAttackerName()) ? heroNameIDOverrides.get(cle.getAttackerName()) : Integer.valueOf((heroNamesIDs.get(cle.getAttackerName()) + 1));
                // // System.out.println("Finding Entity for hero '" + cle.getAttackerName() + "' with ID " + heroID);
                // try {
                //     getStoredHeroByID(heroID);
                // } catch (Error ex) {
                //     System.out.println("ERR: " + heroID.toString() + " " + cle.getAttackerName());
                //     throw ex;
                // }
                // Entity attackerHero = getStoredHeroByID(heroID);
                try {
                    getStoredHeroByDTName(npcToCDota(cle.getAttackerName()));
                } catch (Error er) {
                    System.out.println("WARN: " + er.toString());
                    return;
                }
                Entity attackerHero = getStoredHeroByDTName(npcToCDota(cle.getAttackerName()));
                heroStats.get(attackerHero).addLastHit(runner.getTick());
                if (cle.getTargetName().startsWith("npc_dota_hero")) {
                    heroStats.get(attackerHero).addKill(runner.getTick());
                }
            }
            if (cle.getTargetName().startsWith("npc_dota_hero") && (!cle.hasAttackerIllusion() || !cle.isAttackerIllusion()) && (!cle.hasTargetIllusion() || !cle.isTargetIllusion())) {
                // Integer heroID = heroNameIDOverrides.containsKey(cle.getTargetName()) ? heroNameIDOverrides.get(cle.getTargetName()) : Integer.valueOf((heroNamesIDs.get(cle.getTargetName()) + 1));
                // // System.out.println("Finding Entity for hero '" + cle.getTargetName() + "' with ID " + heroID);
                // try {
                //     getStoredHeroByID(heroID);
                // } catch (Error ex) {
                //     System.out.println("ERR: " + heroID.toString() + " " + cle.getTargetName());
                //     throw ex;
                // }
                // Entity targetHero = getStoredHeroByID(heroID);
                try {
                    getStoredHeroByDTName(npcToCDota(cle.getTargetName()));
                } catch (Error er) {
                    System.out.println("WARN: " + er.toString());
                    return;
                }
                Entity targetHero = getStoredHeroByDTName(npcToCDota(cle.getTargetName()));
                heroStats.get(targetHero).addDeath(runner.getTick());
                // System.out.println(cle.getAttackerName() + " killed " + cle.getTargetName());
            }
        }
    }


    // @OnStringTableCreated
    // @UsesStringTable(value="*")
    // public void onStringTableCreated(int tableNum, StringTable table) throws IOException {
    //     // System.out.println("STRING TABLE RECEIVED");
    //     // System.out.println(table.toString());
    //     // int i = 0;
    //     // while (table.hasIndex(i)) {
    //     //     if (table.getValueByIndex(i) != null) {
    //     //         System.out.println(i + ": " + table.getValueByIndex(i).toStringUtf8());
    //     //         i++;
    //     //     } else {
    //     //         break;
    //     //     }
    //     // }
    //     LinkedList<String> data = new LinkedList<String>();
    //     data.add(table.toString());
    //     Path file = Paths.get("TABLES\\" + table.getName() + ".txt");
    //     Files.write(file, data, StandardCharsets.UTF_8);
    //     // System.exit(0);
    // }


    public void saveData(CSVHolder d, String fname) {
        try {
            Path file = Paths.get(fname);
            Files.write(file, d.data, StandardCharsets.UTF_8);
            // FileWriter myWriter = new FileWriter("output.txt");
            // myWriter.write(d.compile());
            // myWriter.close();
            System.out.println("Successfully wrote to the file.");
          } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
          }
    }
}
