from datetime import datetime

import pandas as pd


def fetch_santiment(
    symbol: str,
    interval: str,
    beginning_date: datetime,
    ending_date: datetime,
) -> pd.DataFrame:
    import san

    assert symbol == "BTC"
    slug = "santiment"
    queries = [
        # "sentiment_balance_bitcointalk",
        # "sentiment_balance_reddit",
        # "sentiment_balance_telegram",
        "sentiment_balance_total",
        # "sentiment_balance_total_change_1d",
        # "sentiment_balance_total_change_30d",
        # "sentiment_balance_total_change_7d",
        "sentiment_balance_twitter",
        # "sentiment_balance_twitter_crypto",
        # "sentiment_balance_twitter_news",
        # "sentiment_balance_youtube_videos",
        # "sentiment_volume_consumed_bitcointalk",
        # "sentiment_volume_consumed_reddit",
        # "sentiment_volume_consumed_telegram",
        "sentiment_volume_consumed_total",
        # "sentiment_volume_consumed_total_change_1d",
        # "sentiment_volume_consumed_total_change_30d",
        # "sentiment_volume_consumed_total_change_7d",
        # "sentiment_volume_consumed_twitter",
        # "sentiment_volume_consumed_twitter_crypto",
        # "sentiment_volume_consumed_twitter_news",
        # "sentiment_volume_consumed_youtube_videos",
    ]
    list_df = []
    for query in queries:
        df = san.get(
            f"{query}/{slug}",
            from_date=beginning_date,
            to_date=ending_date,
            interval=interval,
        )
        df.index.names = ["Datetime"]
        df = df.rename(columns={"value": query})

        list_df.append(df)
    sentiment = pd.concat(list_df, axis=1).astype("float32")
    return sentiment
