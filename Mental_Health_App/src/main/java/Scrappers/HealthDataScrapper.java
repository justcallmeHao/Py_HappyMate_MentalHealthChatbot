package Scrappers;

import java.util.HashMap;
import java.util.Map;

public class HealthDataScrapper implements Scrapper {
    @Override
    public Map<String, Object> scrapeUserData() throws Exception {
        Map<String, Object> data = new HashMap<>();
        try {
            // Simulate API call to mobile health provider
            String healthJson = "{ \"steps\": 7000, \"heart_rate_avg\": 70 }";

            data.put("steps", 7000);
            data.put("heart_rate_avg", 70);
        } catch (Exception e) {
            System.err.println("Failed to scrape Health data: " + e.getMessage());
            throw e;
        }
        return data;
    }

    @Override
    public String getSourceName() {
        return "HealthAPI";
    }
}
