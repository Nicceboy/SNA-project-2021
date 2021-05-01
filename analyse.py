#!/usr/bin/env python3
import csv
import networkx
import pathlib
import datetime
import logging
import argparse
import sys
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
        self.followers: int = int(followers)
        self.number_statuses: int = int(number_statuses)
        if time:
            self.time: datetime.datetime = self.format_str_time(time)
        self.tweet: str = tweet

    def format_str_time(self, time: str) -> datetime.datetime:
        """Convert string time to datetime object
        Example date: month/day/year 1/6/2015 21:07
        """
        return datetime.datetime.strptime(time, "%m/%d/%Y %H:%M")


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
            if not tweets[index - 1].username and not tweets[index - 1].tweet:
                GLOBAL_LOGGER.debug(f"Invalid data on row {tweets[index - 1].row_id}")
    GLOBAL_LOGGER.info("CSV file loaded.")
    return tweets


def main():

    DATA_PATH = pathlib.Path("data/isis_twitter_data.csv")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"CSV dataset not found from the path {DATA_PATH}. Please, download and convert it."
        )
    data = read_csv_into_obj(DATA_PATH)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-l",
        "--log",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
        default=None,
    )
    if len(sys.argv) >= 1 and len(sys.argv) < 3:
        args = parser.parse_args(args=sys.argv[1:])
    else:
        raise ValueError("Too many program arguments provided.")

    log_level = args.log_level if args.log_level else "INFO"
    if log_level not in {"DEBUG"}:
        sys.tracebacklimit = 0  # avoid track traces unless debugging
    logging.basicConfig(
        format=f"%(levelname)s - %(name)s: %(message)s",
        level=getattr(logging, log_level),
    )

    GLOBAL_LOGGER = logging.getLogger("main")

    main()
