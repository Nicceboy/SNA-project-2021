import pytest
import pathlib
import logging
import analyse
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from analyse import TweetData, Sentiment
from analyse.__main__ import read_csv_into_obj


DATA_PATH = pathlib.Path("data/isis_twitter_data.csv")


def list_lower(inputs: List[str]):
    return list(map(lambda x: x.lower(), inputs))

@pytest.fixture(autouse=True)
def init_logger():
    logging.basicConfig(
        format=f"%(levelname)s - %(name)s: %(message)s",
        level=logging.DEBUG,
    )
    analyse.__main__.GLOBAL_LOGGER = logging.getLogger("pytest")


@pytest.fixture(scope="function")
def tweet_subset(tmp_path):
    """Get only 100 first items for some tests"""
    data = read_csv_into_obj(DATA_PATH)
    yield data[:100]


def test_data_load():

    data = read_csv_into_obj(DATA_PATH)
    assert len(data) == 17410


def test_hashtags(tweet_subset):
    """Look for .csv to match corresponding tweet ids and contents for hashtags"""
    assert tweet_subset[21].get_hashtags() == ["jn"]
    assert tweet_subset[24].get_hashtags() == ["jn", "is"]
    assert tweet_subset[33].get_hashtags() == ["is", "dawlah", "cubs"]


def test_mentions(tweet_subset):
    """Look for .csv to match corresponding tweet ids and contents for mentions"""
    assert tweet_subset[10].get_mentions() == list_lower([
        "KhalidMaghrebi",
        "seifulmaslul123",
        "CheerLeadUnited",
        "IbnNabih1",
    ])

    assert tweet_subset[14].get_mentions() == list_lower(["IbnNabih1", "MuwMedia", "Dawlat_islam7"])
    # Also retweet
    assert tweet_subset[27].get_mentions() == list_lower(["GIIMedia_CH004"])


def test_retweets(tweet_subset):
    """Check if tweet is retweet"""
    assert tweet_subset[27].is_retweet()

    assert not tweet_subset[36].is_retweet()
    assert tweet_subset[63].is_retweet()


def test_sentiment_types(tweet_subset):
    """Test sentiment types"""
    sample = tweet_subset[0]

    sample.sentiment = "NEUTRAL"
    with pytest.raises(ValueError):
        sample.sentiment = "NEUTRALL"
    assert sample.sentiment == Sentiment.NEUTRAL
    sample.sentiment = Sentiment.NEUTRAL
    assert sample.sentiment == Sentiment.NEUTRAL
    sample.sentiment = "POSITIVE"
    assert sample.sentiment == Sentiment.POSITIVE
    sample.sentiment = "NEGATIVE"
    assert sample.sentiment == Sentiment.NEGATIVE


def test_sentiment_values(tweet_subset):

    analyzer = SentimentIntensityAnalyzer()
    for t in range(0, 5):
        tweet_subset[t].set_sentiment_value(analyzer)
    assert tweet_subset[0].sentiment == Sentiment.POSITIVE
    assert tweet_subset[1].sentiment == Sentiment.POSITIVE
    assert tweet_subset[2].sentiment == Sentiment.NEUTRAL
    assert tweet_subset[3].sentiment == Sentiment.POSITIVE
    assert tweet_subset[4].sentiment == Sentiment.NEGATIVE
