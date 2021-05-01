import pytest
import pathlib
import logging
import analyse
from analyse import TweetData
from analyse.__main__ import read_csv_into_obj

DATA_PATH = pathlib.Path("data/isis_twitter_data.csv")


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
    assert tweet_subset[21].get_hashtags() == ["JN"]
    assert tweet_subset[24].get_hashtags() == ["JN", "IS"]
    assert tweet_subset[33].get_hashtags() == ["IS", "Dawlah", "cubs"]


def test_mentions(tweet_subset):
    """Look for .csv to match corresponding tweet ids and contents for mentions"""
    assert tweet_subset[10].get_mentions() == [
        "KhalidMaghrebi",
        "seifulmaslul123",
        "CheerLeadUnited",
        "IbnNabih1",
    ]

    assert tweet_subset[14].get_mentions() == ["IbnNabih1", "MuwMedia", "Dawlat_islam7"]
    # Also retweet
    assert tweet_subset[27].get_mentions() == ["GIIMedia_CH004"]


def test_retweets(tweet_subset):
    """Check if tweet is retweet"""
    assert tweet_subset[27].is_retweet()

    assert not tweet_subset[36].is_retweet()
    assert tweet_subset[63].is_retweet()
