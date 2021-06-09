package utils;

import java.lang.reflect.Field;
import java.util.LinkedList;

import skadistats.clarity.model.Entity;
import skadistats.clarity.model.Vector;
import skadistats.clarity.processor.entities.Entities;
import java.lang.Math;

public class Util {

    public class FloatTriple {
        public float x;
        public float y;
        public float z;

        public FloatTriple(float x, float y, float z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public void set(float x, float y, float z) {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public void set(FloatTriple t) {
            this.x = t.x;
            this.y = t.y;
            this.z = t.z;
        }

        public void plus(FloatTriple t) {
            this.x += t.x;
            this.y += t.y;
            this.z += t.z;
        }

        public void minus(FloatTriple t) {
            this.x -= t.x;
            this.y -= t.y;
            this.z -= t.z;
        }

        public void divide(FloatTriple t) {
            this.x /= t.x;
            this.y /= t.y;
            this.z /= t.z;
        }

        public void multiply(FloatTriple t) {
            this.x *= t.x;
            this.y *= t.y;
            this.z *= t.z;
        }

        public FloatTriple newPlus(FloatTriple t) {
            return new FloatTriple(this.x + t.x, this.y + t.y, this.z + t.z);
        }

        public FloatTriple newMinus(FloatTriple t) {
            return new FloatTriple(this.x - t.x, this.y - t.y, this.z - t.z);
        }

        public FloatTriple newDivide(FloatTriple t) {
            return new FloatTriple(this.x / t.x, this.y / t.y, this.z / t.z);
        }

        public FloatTriple newMultiply(FloatTriple t) {
            return new FloatTriple(this.x * t.x, this.y * t.y, this.z * t.z);
        }

        public String toString() {
            return Float.toString(this.x) + "," + Float.toString(this.y) + "," + Float.toString(this.z);
        }

        public float distanceTo(FloatTriple o) {
            return (float) Math.sqrt(Math.pow(o.x - x, 2)
                                    + Math.pow(o.y - y, 2)
                                    + Math.pow(o.z - z, 2));
        }
    }


    /**
     * Calculate the area of a polygon
     * 
     * https://www.geeksforgeeks.org/area-of-a-polygon-with-given-n-ordered-vertices/
     * 
     * @param points The vertices of the polygon
     * @return the area of the polygon
     */
    public static float polygonArea(FloatTriple[] points) {
        // Initialze area
        float area = 0f;
     
        // Calculate value of shoelace formula
        int j = points.length - 1;
        for (int i = 0; i < points.length; i++)
        {
            area += (points[j].x + points[i].x) * (points[j].y - points[i].y);
             
            // j is previous vertex to i
            j = i;
        }
     
        // Return absolute value
        return (float) Math.abs(area / 2.0);
    }


    /**
     * Calculate the centroid of a polygon
     * 
     * @param points the vertices of the polygon
     * @return A coordinate representing the centroid of the polygon
     */
    public FloatTriple polygonCentroid(FloatTriple[] points) {
        FloatTriple centroid = new FloatTriple(0f, 0f, 0f);
        for (FloatTriple p: points) {
            centroid.plus(p);
        }
        centroid.x /= points.length;
        centroid.y /= points.length;
        centroid.z /= points.length;

        return centroid;
    }
    
    
    /**
     * Get the team name from an id
     * @param team
     * @return
     */
    public static String getTeamName(int team) {
        switch(team) {
            case 2:
                return "Radiant";
            case 3:
                return "Dire";
            default:
                return "Unkown";
        }
    }

    /**
     * Get the respwan time of an enitity, based on its level
     * @param e
     * @return
     */
    public static float getRespawnTime(Entity e) {
        int[] respawnTimes = {12,15,18,21,24,26,28,30,32,34,36,44,46,48,50,52,54,65,70,75,80,85,90,95,100,100,100,100,100,100};
        int heroLevel = e.getProperty("m_iCurrentLevel");
        return respawnTimes[heroLevel - 1];
    }

    /**
     * Get the in-game position of an entity
     * @param e
     * @return
     */
    public FloatTriple getPosition(Entity e) {
        if (!e.hasProperty("CBodyComponent.m_cellX")) {
            return null;
        }
        return new FloatTriple(
                getPositionComponent(e, "X"),
                getPositionComponent(e, "Y"),
                getPositionComponent(e, "Z")
        );
    }

    private static float getPositionComponent(Entity e, String which) {
        int cell = e.getProperty("CBodyComponent.m_cell" + which);
        float vec = e.getProperty("CBodyComponent.m_vec" + which);
        return cell * 128.0f + vec;
    }

    /**
     * Gets the current game time in seconds
     * @param entities
     * @return
     */
    public static Float getGameTime(Entities entities) {
        if (entities == null) {return -90.0f;}
        Entity entity = entities.getByDtName("CDOTAGamerulesProxy");
        Float gameTime, startTime, preGameTime, transitionTime;
        // before the match starts, there's CDOTAGamerulesProxy
        if (entity.getDtClass().getDtName().equals("CDOTAGamerulesProxy")) {
            gameTime = entity.getProperty("m_pGameRules.m_fGameTime");
            // before the match starts, there's no game "time"
            if (gameTime != null) {
                preGameTime = entity.getProperty("m_pGameRules.m_flPreGameStartTime");
                // before hero picking and strategy time are finished, the
                //  pre-game countdown is still at 0, i.e. nothing has happened
                //  in the match
                if (preGameTime > 0) {
                    startTime = entity.getProperty("m_pGameRules.m_flGameStartTime");
                    // time after the clock hits 0:00
                    if (startTime > 0) {
                        return gameTime - startTime;
                    }
                    // between the pre-game and 0:00 time of the match, the
                    //  transition time reflects when the match is supposed to
                    //  start (i.e. hit 0:00 on the clock), and gives a good
                    //  approximation of when the match will start. Up to that
                    //  point, the start time is set to 0.
                    else {
                        transitionTime = entity.getProperty("m_pGameRules.m_flStateTransitionTime");
                        return gameTime - transitionTime;
                    }
                }
            }
        }
        return -90.0f;
    }


    public static String classToString(Object o) {
        StringBuilder result = new StringBuilder();
        String newLine = System.getProperty("line.separator");
      
        result.append( o.getClass().getName() );
        result.append( " Object {" );
        result.append(newLine);
      
        //determine fields declared in o class only (no fields of superclass)
        Field[] fields = o.getClass().getDeclaredFields();
      
        //print field names paired with their values
        for ( Field field : fields  ) {
          result.append("  ");
          try {
            result.append( field.getName() );
            result.append(": ");
            //requires access to private field:
            result.append( field.get(o) );
          } catch ( IllegalAccessException ex ) {
            System.out.println(ex);
          }
          result.append(newLine);
        }
        result.append("}");
      
        return result.toString();
      }
}
