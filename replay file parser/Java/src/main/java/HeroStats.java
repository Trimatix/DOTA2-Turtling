import java.util.HashMap;

import utils.Util.FloatTriple;
import skadistats.clarity.model.Entity;
import java.util.TreeSet;

public class HeroStats {
    public HashMap<Integer, FloatTriple> coords;
    public Entity ent;
    public HashMap<Integer, Integer> netWorths;
    public HashMap<Integer, Integer> XPs;
    public HashMap<StatType, TreeSet<Integer>> ticks;
    public HashMap<Integer, Integer> kills;
    public int currentKills;
    public HashMap<Integer, Integer> deaths;
    public int currentDeaths;
    public HashMap<Integer, Integer> lastHits;
    public int currentLastHits;

    public enum StatType {
        coords,
        netWorth,
        XP,
        kills,
        deaths,
        lastHits
    }


    public HeroStats(Entity ent) {
        this.ent = ent;
        netWorths = new HashMap<Integer,Integer>();
        XPs = new HashMap<Integer,Integer>();
        coords = new HashMap<Integer, FloatTriple>();
        kills = new HashMap<Integer, Integer>();
        deaths = new HashMap<Integer, Integer>();
        lastHits = new HashMap<Integer, Integer>();
        ticks = new HashMap<StatType, TreeSet<Integer>>();
        currentKills = 0;
        currentDeaths = 0;
        currentLastHits = 0;
        for (StatType t: StatType.values()) {
            ticks.put(t, new TreeSet<Integer>());
        }
    }

    private void updateTick(StatType stat, Integer tick) {
        ticks.get(stat).add(tick);
    }


    public void addCoords(Integer tick, FloatTriple coords) {
        this.coords.put(tick, coords);
        updateTick(StatType.coords, tick);
    }

    public void addCoords(int tick, FloatTriple coords) {
        addCoords(Integer.valueOf(tick), coords);
    }



    public void addNetWorth(Integer tick, Integer worth) {
        this.netWorths.put(tick, worth);
        updateTick(StatType.netWorth, tick);
    }

    public void addNetWorth(int tick, Integer worth) {
        addNetWorth(Integer.valueOf(tick), worth);
    }

    public void addNetWorth(Integer tick, int worth) {
        addNetWorth(tick, Integer.valueOf(worth));
    }

    public void addNetWorth(int tick, int worth) {
        addNetWorth(Integer.valueOf(tick), Integer.valueOf(worth));
    }

    
    
    public void addXP(Integer tick, Integer xp) {
        this.XPs.put(tick, xp);
        updateTick(StatType.XP, tick);
    }

    public void addXP(int tick, Integer xp) {
        addXP(Integer.valueOf(tick), xp);
    }

    public void addXP(Integer tick, int xp) {
        addXP(tick, Integer.valueOf(xp));
    }

    public void addXP(int tick, int xp) {
        addXP(Integer.valueOf(tick), Integer.valueOf(xp));
    }


    public void addKill(Integer tick) {
        currentKills++;
        this.kills.put(tick, currentKills);
        updateTick(StatType.kills, tick);
    }

    public void addKill(int tick) {
        addKill(Integer.valueOf(tick));
    }

    
    public void addDeath(Integer tick) {
        currentDeaths++;
        this.deaths.put(tick, currentDeaths);
        updateTick(StatType.deaths, tick);
    }

    public void addDeath(int tick) {
        addDeath(Integer.valueOf(tick));
    }


    public void addLastHit(Integer tick) {
        // if (currentLastHits == 0) {
        //     System.out.println("First last hit recorded for " + ent.getDtClass().getDtName());
        // }
        currentLastHits++;
        this.lastHits.put(tick, currentLastHits);
        updateTick(StatType.lastHits, tick);
    }

    public void addLastHit(int tick) {
        addLastHit(Integer.valueOf(tick));
    }


    private Integer getTick(Integer tick, StatType stat) {
        Integer floor = ticks.get(stat).floor(tick);
        Integer ceiling = ticks.get(stat).ceiling(tick);
        if (floor == null && ceiling != null) {
            return ceiling;
        } else if (ceiling == null && floor != null) {
            return floor;
        } else if (ceiling == null && floor == null) {
            return null;
        } else if (tick - floor < ceiling - tick) {
            return floor;
        } else {
            return ceiling;
        }
    }


    public FloatTriple getCoords(Integer tick) {
        Integer foundTick = getTick(tick, StatType.coords);
        if (foundTick == null) {
            throw new Error("Couldnt find floor for " + ent.getDtClass().getDtName() + " coords, tick " + tick.toString());
        } else {
            return coords.get(foundTick);
        }
    }

    public Integer getNetWorth(Integer tick) {
        Integer foundTick = getTick(tick, StatType.netWorth);
        if (foundTick == null) {
            throw new Error("Couldnt find floor for " + ent.getDtClass().getDtName() + " netWorths, tick " + tick.toString());
        } else {
            return netWorths.get(foundTick);
        }
    }

    public Integer getXP(Integer tick) {
        Integer foundTick = getTick(tick, StatType.XP);
        if (foundTick == null) {
            throw new Error("Couldnt find floor for " + ent.getDtClass().getDtName() + " XP, tick " + tick.toString());
        } else {
            return XPs.get(foundTick);
        }
    }

    public Integer getKills(Integer tick) {
        Integer foundTick = getTick(tick, StatType.kills);
        if (foundTick == null) {
            return 0;
        } else {
            return kills.get(foundTick);
        }
    }

    public Integer getDeaths(Integer tick) {
        Integer foundTick = getTick(tick, StatType.deaths);
        if (foundTick == null) {
            return 0;
        } else {
            return deaths.get(foundTick);
        }
    }

    public Integer getLastHits(Integer tick) {
        Integer foundTick = getTick(tick, StatType.lastHits);
        if (foundTick == null) {
            return 0;
        } else {
            return lastHits.get(foundTick);
        }
    }
}
