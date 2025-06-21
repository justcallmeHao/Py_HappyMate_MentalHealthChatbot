package Scrappers;

import java.util.Map;

public interface Scrapper {
    Map<String, Object> scrapeUserData() throws Exception;
    String getSourceName();
}