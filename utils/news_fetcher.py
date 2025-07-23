import pandas as pd
from datetime import timezone
import logging
import investpy
logger = logging.getLogger(__name__)

def fetch_forex_factory_calendar():
    """
    Fetches the economic calendar from Investing.com using the investpy library.
    This is a more robust alternative to scraping Forex Factory.

    Returns:
        pd.DataFrame: A DataFrame containing upcoming news events, or an empty DataFrame on error.
                      Columns: 'datetime_utc', 'currency', 'impact', 'event'
    """
    logger.info("Fetching economic calendar data using investpy...")
    try:
        # Fetch calendar data from today onwards
        # investpy fetches data based on the local timezone of the machine running the code.
        # We will convert it to UTC.
        news_df = investpy.economic_calendar()

        if news_df.empty:
            logger.warning("investpy returned an empty DataFrame for the economic calendar.")
            return pd.DataFrame()

        # --- Data Processing to match the required format ---

        # 1. Combine 'date' and 'time' into a single datetime column
        # The 'date' is in dd/mm/yyyy format.
        news_df['datetime_local'] = pd.to_datetime(news_df['date'] + ' ' + news_df['time'], format='%d/%m/%Y %H:%M')

        # 2. Convert to UTC
        # We assume the fetched time is in the local timezone of the server.
        # We then convert it to UTC for standardization.
        news_df['datetime_utc'] = news_df['datetime_local'].apply(
            lambda dt: dt.tz_localize('UTC') if dt.tzinfo is None else dt.astimezone(timezone.utc)
        )


        # 3. Rename 'importance' to 'impact' and format values
        news_df.rename(columns={'importance': 'impact'}, inplace=True)
        impact_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
        news_df['impact'] = news_df['impact'].map(impact_map).fillna('Low') # Default to 'Low' if mapping fails

        # 4. Select and reorder the final columns to match the old format
        final_df = news_df[['datetime_utc', 'currency', 'impact', 'event']].copy()

        logger.info(f"Successfully fetched and processed {len(final_df)} news events.")
        return final_df

    except Exception as e:
        # This can happen if investpy has issues (e.g., website changes, connection errors)
        logger.error(f"Failed to fetch or process economic calendar via investpy: {e}", exc_info=True)
        # Return an empty DataFrame to ensure the calling function handles the failure gracefully
        return pd.DataFrame()

if __name__ == '__main__':
    # Example usage for testing the new function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # You might see a UserWarning from investpy about timezones, which is expected.
    # Our code handles the conversion to UTC.
    calendar_df = fetch_forex_factory_calendar()

    if not calendar_df.empty:
        print("\n--- Upcoming High Impact News ---")
        high_impact_news = calendar_df[calendar_df['impact'] == 'High']
        print(high_impact_news.head())

        print("\n--- Upcoming Medium Impact News ---")
        medium_impact_news = calendar_df[calendar_df['impact'] == 'Medium']
        print(medium_impact_news.head())
    else:
        print("\nCould not fetch any news events.")
