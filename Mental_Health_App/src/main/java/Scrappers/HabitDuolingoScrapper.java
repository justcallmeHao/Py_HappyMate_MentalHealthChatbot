package Scrappers;

import java.util.HashMap;
import java.util.Map;

public class HabitDuolingoScrapper implements Scrapper {
    @Override
    public Map<String, Object> scrapeUserData() throws Exception {
        Map<String, Object> data = new HashMap<>();
        try {
            // Mocked API call and data retrieval
            String duolingoJson = "{ \"streak\": 45, \"minutes_today\": 20 }";

            // Simulate processing
            data.put("streak", 45);
            data.put("minutes_today", 20);
        } catch (Exception e) {
            System.err.println("Failed to scrape Duolingo data: " + e.getMessage());
            throw e;
        }
        return data;
    }

    @Override
    public String getSourceName() {
        return "Duolingo";
    }
}

