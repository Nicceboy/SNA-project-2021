import datetime
import re
from enum import Enum
from typing import List, Union, Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Sentiment(Enum):
    """Sentiment can have three different values
    Scoring is defined here https://github.com/cjhutto/vaderSentiment#about-the-scoring
    """

    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"

    @classmethod
    def _missing_(cls, value: Dict):
        compound = value.get("compound")
        if isinstance(compound, int) or isinstance(compound, float):
            if compound <= -0.05:
                return cls(cls.NEGATIVE)
            elif compound < 0.05 and compound > -0.05:
                return cls(cls.NEUTRAL)
            elif compound >= 0.05:
                return cls(cls.POSITIVE)
            raise ValueError("Value out of range, should not happend.")
        else:
            ValueError("Invalid value type provided for Sentiment instancing.")


class TweetData:
    """Class representing single row from the .csv data set"""

    def __init__(
        self,
        row_id: int,
        name: str,
        username: str,
        description: str,
        location: str,
        followers: int,
        number_statuses: int,
        time: datetime.datetime,
        tweet: str,
    ):
        self.row_id: int = row_id
        self.name: str = name
        self.username: str = username
        self.description: str = description
        self.location: str = location
        self.followers: int = int(followers)
        self.number_statuses: int = int(number_statuses)
        if time:
            self.time: datetime.datetime = self.__format_str_time(time)
        else:
            self.time = None
        self.tweet: str = tweet
        self._sentiment: Sentiment = None

    def __format_str_time(self, time: str) -> datetime.datetime:
        """Convert string time to datetime object
        Example date: month/day/year 1/6/2015 21:07
        """
        return datetime.datetime.strptime(time, "%m/%d/%Y %H:%M")

    def get_hashtags(self) -> List[str]:
        """Get list of hashtags from single tweet"""
        return re.findall(r"#(\w+)", self.tweet)

    def get_mentions(self) -> List[str]:
        """Get list of mentions from single tweet"""
        return re.findall(r"@(\w+)", self.tweet)

    def is_retweet(self) -> bool:
        """Check if tweet starts with RT, remove quotes and whitespaces"""
        return self.tweet.strip('"').strip().startswith("RT")

    @property
    def sentiment(self) -> Sentiment:
        if not self._sentiment:
            # Not provided, calculate it for this tweet
            analyzer = SentimentIntensityAnalyzer()
            vs = analyzer.polarity_scores(self.tweet)
            self._sentiment = Sentiment(vs)
        return self._sentiment

    @sentiment.setter
    def sentiment(self, value: Union[str, Sentiment] = ""):
        if value not in Sentiment.__members__ and not isinstance(value, Sentiment):
            raise ValueError("Sentiment value must be POSITIVE, NEGATIVE or NEUTRAL")
        self._sentiment = Sentiment(value)
