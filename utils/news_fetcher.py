import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

def fetch_forex_factory_calendar(days=7):
    """
    Scrapes the Forex Factory calendar for the specified number of days.

    Returns:
        pd.DataFrame: A DataFrame containing upcoming news events, or an empty DataFrame on error.
    """
    url = "https://www.forexfactory.com/calendar"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch Forex Factory calendar: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.content, 'lxml')
    table = soup.find('table', class_='calendar__table')
    if not table:
        logger.warning("Could not find the calendar table on Forex Factory.")
        return pd.DataFrame()

    events = []
    rows = table.find_all('tr', class_='calendar__row')

    current_date_str = None
    for row in rows:
        # Date row
        date_cell = row.find('td', class_='calendar__date')
        if date_cell:
            current_date_str = date_cell.text.strip()
            continue

        if not current_date_str:
            continue

        # Event row
        time_cell = row.find('td', class_='calendar__time')
        currency_cell = row.find('td', class_='calendar__currency')
        impact_cell = row.find('td', class_='calendar__impact')
        event_cell = row.find('td', class_='calendar__event')

        if not all([time_cell, currency_cell, impact_cell, event_cell]):
            continue

        time_str = time_cell.text.strip()
        currency = currency_cell.text.strip()
        impact_title = impact_cell.find('span')['title'] if impact_cell.find('span') else 'Low Impact'
        event_name = event_cell.text.strip()

        if "All Day" in time_str:
            continue

        try:
            # Combine date and time to create a datetime object
            # Note: Forex Factory time needs to be handled carefully. Assuming it's EST/EDT.
            # A more robust solution would use their server time and convert.
            # For now, we parse it as-is and attach UTC for consistency.
            # This requires the user to be aware of the time difference.
            event_dt_str = f"{current_date_str} {datetime.now().year} {time_str}"
            # A simple way to handle time, assuming the website time is close to user's timezone context
            # A more robust solution would involve timezone libraries like pytz and knowing FF's server timezone
            event_dt = pd.to_datetime(event_dt_str, format='%a %b %d %Y %I:%M%p', errors='coerce')

            if pd.isna(event_dt):
                continue

            # For this implementation, we will assume the parsed time is in a known timezone (e.g., New York)
            # and convert to UTC. Let's assume 'America/New_York' for Forex Factory.
            try:
                from zoneinfo import ZoneInfo
                ny_tz = ZoneInfo("America/New_York")
                event_dt_localized = event_dt.tz_localize(ny_tz)
                event_dt_utc = event_dt_localized.astimezone(timezone.utc)
            except ImportError:
                # Fallback for older python
                event_dt_utc = event_dt.tz_localize('UTC') # Less accurate fallback

            events.append({
                "datetime_utc": event_dt_utc,
                "currency": currency,
                "impact": impact_title.replace(" Impact", ""),
                "event": event_name
            })
        except Exception as e:
            logger.warning(f"Could not parse event row: {e}")
            continue

    return pd.DataFrame(events)

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    news_df = fetch_forex_factory_calendar()
    if not news_df.empty:
        print("Upcoming High Impact News:")
        print(news_df[news_df['impact'] == 'High'])
