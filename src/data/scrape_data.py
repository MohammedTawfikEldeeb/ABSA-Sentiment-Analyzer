import pandas as pd
from google_play_scraper import Sort , reviews_all
import logging
from src import config
import sys
logging.basicConfig(level=logging.INFO)
logging.info("Starting scraping")


def scrape_reviews(url: str):
    all_reviews = reviews_all(
        url,
        sleep_milliseconds=0,
        lang="ar",
        country="eg",
        sort=Sort.NEWEST,
    )

    df = pd.DataFrame(all_reviews)
    df_useful = df[['at', 'score', 'content']].copy()
    df_useful.rename(columns={
        'at': 'review_date',
        'score': 'rating',
        'content': 'review_text'
    }, inplace=True)

    return df_useful



if __name__ == '__main__':
    output_path = sys.argv[1]
    
    logging.info("Starting scraping process...")
    url = "com.egyptianbanks.instapay"
    reviews_df = scrape_reviews(url)
    
    reviews_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"Data saved successfully to {output_path}")