#!/usr/bin/env python3
import csv
import networkx
import pathlib
import datetime
from typing import Union, List


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
        self.followers: int = followers
        self.number_statuses: int = number_statuses
        self.time: datetime.datetime = time
        self.tweet: str = tweet


def read_csv_into_obj(path: pathlib) -> List[TweetData]:
    """Convert CSV file into list of TweetData objects"""
    tweets = []
    with path.open(newline="") as csvfile:
        csv_tweets = csv.reader(csvfile, delimiter=",", quotechar='"')
        for index, row in enumerate(csv_tweets):
            # Skip column names
            if index == 0:
                continue
            tweets.append(TweetData(*row))
    return tweets


def main():

    DATA_PATH = pathlib.Path("data/isis_twitter_data.csv")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"CSV dataset not found from the path {DATA_PATH}. Please, download and convert it."
        )

    data = read_csv_into_obj(DATA_PATH)


if __name__ == "__main__":
    main()
