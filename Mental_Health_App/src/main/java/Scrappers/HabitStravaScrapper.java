package Scrappers;

import java.util.HashMap;
import java.util.Map;

public class HabitStravaScrapper implements Scrapper {
    @Override
    public Map<String, Object> scrapeUserData() throws Exception {
        Map<String, Object> data = new HashMap<>();
        try {
            // Mock API call
            String stravaJson = "{ \"distance_km\": 12.5, \"active_days\": 3 }";

            data.put("distance_km", 12.5);
            data.put("active_days", 3);
        } catch (Exception e) {
            System.err.println("Failed to scrape Strava data: " + e.getMessage());
            throw e;
        }
        return data;
    }

    @Override
    public String getSourceName() {
        return "Strava";
    }
}
