import datetime
import re
from typing import List


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
