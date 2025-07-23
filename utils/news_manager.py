import pandas as pd
from datetime import datetime, timedelta, timezone
import logging
from .news_fetcher import fetch_forex_factory_calendar

logger = logging.getLogger(__name__)

class NewsManager:
    def __init__(self, config):
        self.config = config.get('news_filter', {})
        self.news_cache = pd.DataFrame()
        self.last_fetch_time = None
        self.cache_ttl = timedelta(hours=self.config.get('cache_ttl_hours', 4))

        if self.is_enabled():
            self.fetch_and_cache_events()

    def is_enabled(self):
        return self.config.get('enabled', False)

    def fetch_and_cache_events(self):
        """Fetches news from Forex Factory and caches it."""
        logger.info("Fetching and caching Forex Factory news events...")
        try:
            self.news_cache = fetch_forex_factory_calendar()
            self.last_fetch_time = datetime.now(timezone.utc)
            if not self.news_cache.empty:
                logger.info(f"Successfully cached {len(self.news_cache)} news events.")
            else:
                logger.warning("Failed to fetch or cache any news events.")
        except Exception as e:
            logger.error(f"An error occurred during news fetching and caching: {e}", exc_info=True)
            self.news_cache = pd.DataFrame()

    def _is_cache_stale(self):
        """Checks if the cache is older than its time-to-live (TTL)."""
        if not self.last_fetch_time:
            return True
        return (datetime.now(timezone.utc) - self.last_fetch_time) > self.cache_ttl

    def is_trade_prohibited(self, symbol: str):
        """
        Checks if a trade should be blocked.
        1. First, checks for global-blocking news (e.g., USD).
        2. If none, checks for news specific to the symbol's currency pair.
        """
        if not self.is_enabled():
            return False

        if self._is_cache_stale():
            self.fetch_and_cache_events()

        if self.news_cache.empty:
            logger.warning("News cache is empty, cannot perform news check. Allowing trade.")
            return False

        prohibited_impacts = self.config.get('prohibited_impacts', ['High'])
        minutes_before = self.config.get('minutes_before_news', 30)
        minutes_after = self.config.get('minutes_after_news', 30)
        now_utc = datetime.now(timezone.utc)

        # --- NEW: Global Currency Check ---
        global_block_currencies = self.config.get('block_all_on_currency', [])
        if global_block_currencies:
            global_news = self.news_cache[
                (self.news_cache['currency'].isin(global_block_currencies)) &
                (self.news_cache['impact'].isin(prohibited_impacts))
            ]
            for _, event in global_news.iterrows():
                event_time = event['datetime_utc']
                blackout_start = event_time - timedelta(minutes=minutes_before)
                blackout_end = event_time + timedelta(minutes=minutes_after)
                if blackout_start <= now_utc <= blackout_end:
                    logger.warning(
                        f"TRADE BLOCKED (GLOBAL): All trading paused due to high-impact '{event['event']}' "
                        f"for {event['currency']} at {event_time.strftime('%H:%M')} UTC."
                    )
                    return True
        # --- End of Global Check ---

        # --- Original Pair-Specific Check ---
        currencies_in_pair = []
        if len(symbol) >= 6:
            currencies_in_pair.append(symbol[:3].upper())
            currencies_in_pair.append(symbol[3:6].upper())
        else:
            return False # Cannot check news if symbol format is unknown

        pair_specific_news = self.news_cache[
            (self.news_cache['currency'].isin(currencies_in_pair)) &
            (self.news_cache['impact'].isin(prohibited_impacts))
        ]

        if pair_specific_news.empty:
            return False

        for _, event in pair_specific_news.iterrows():
            event_time = event['datetime_utc']
            blackout_start = event_time - timedelta(minutes=minutes_before)
            blackout_end = event_time + timedelta(minutes=minutes_after)

            if blackout_start <= now_utc <= blackout_end:
                logger.warning(
                    f"TRADE BLOCKED for {symbol}. Reason: High-impact '{event['event']}' "
                    f"for {event['currency']} at {event_time.strftime('%H:%M')} UTC. "
                    f"Current time {now_utc.strftime('%H:%M')} is within blackout window."
                )
                return True

        return False
