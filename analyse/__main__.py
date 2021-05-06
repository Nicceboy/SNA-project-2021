#!/usr/bin/env python3
import re
import csv
import networkx as nx
import pathlib
import logging
import argparse
import sys
import matplotlib

# matplotlib.use("Qt5Agg")
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import Union, List, Dict
from .tweetdata import TweetData
from copy import deepcopy
from operator import itemgetter
from itertools import islice


USER_META = {
    "amount_tweets": 0,
    "amount_retweets": 0,
    "amount_mentions": 0,
    "amount_hashtags": 0,
    "tweets": [],
}


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
    GLOBAL_LOGGER.info(f"CSV file loaded from the path '{str(path)}'.")
    GLOBAL_LOGGER.info(f"{len(tweets)} tweets found.")
    return tweets


def sort_users(
    users: Dict, by_meta_value: str = "amount_tweets", amount_to_show: int = 10
) -> Dict:

    if by_meta_value not in USER_META.keys():
        raise ValueError("Key used for sorting must be found from Dict 'USER_META'")

    users_sorted = dict(
        sorted(users.items(), key=lambda item: item[1][by_meta_value], reverse=True)
    )
    GLOBAL_LOGGER.info(
        f"Users sorted by {by_meta_value}, showing first {amount_to_show} users:"
    )
    for index, key in enumerate(list(users_sorted.keys())[:amount_to_show]):
        GLOBAL_LOGGER.info(
            f"User {index + 1} ({key}): {by_meta_value}, {users_sorted[key][by_meta_value]}"
        )
    # Something weird going on if these are not equal
    assert len(users_sorted[next(iter(users_sorted))].get("tweets")) == users_sorted[
        next(iter(users_sorted))
    ].get("amount_tweets")

    return users_sorted


def plot(
    labels: List[str],
    data: List[int],
    x_label: str = "",
    y_label: str = "",
    description: str = "",
    plot_type: str = "bar",
):
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    if plot_type == "log":
        plt.semilogx(labels, data)
    elif plot_type == "bar":
        plt.bar(labels, data)
    # plt.axes().set_xlim(1, 1.5)
    plt.xlabel(x_label)
    # plt.x
    plt.ylabel(y_label)
    plt.title(description)
    plt.tick_params("x",labelrotation=-45)
    # for tick in plt.get_xticklabels():
    #     tick.set_rotation(90)
    plt.show()


def plot_users(
    users: Dict,
    plot_by: str = "amount_tweets",
    x_label: str = "",
    y_label: str = "",
    description: str = "",
    use_name: bool = False,
    plot_type: str = "bar",
):
    if plot_by not in USER_META.keys():
        raise ValueError("Key used for plotting must be found from Dict 'USER_META'")
    # Take 5 first characters, hide rest
    if use_name:
        names = [k[:5] + "*" for k in users.keys()]
    else:
        names = range(0, len(users.keys()))
    num = [users.get(k).get(plot_by) for k in users.keys()]
    plot(names, num, x_label, y_label, description, plot_type)


def group_users_by_tweet_meta(data: List[TweetData]) -> Dict:
    users = {}
    oldest = None
    latest = None
    for tweet in data:
        if tweet.username not in users:
            users[tweet.username] = deepcopy(USER_META)
        users[tweet.username]["amount_tweets"] += 1
        users[tweet.username]["amount_retweets"] += 1 if tweet.is_retweet() else 0
        users[tweet.username]["amount_mentions"] += len(tweet.get_mentions())
        users[tweet.username]["amount_hashtags"] += len(tweet.get_hashtags())
        users[tweet.username]["tweets"].append(tweet)
        if not latest:
            latest = tweet.time
        if not oldest:
            oldest = tweet.time
        if latest < tweet.time:
            latest = tweet.time
        if oldest > tweet.time:
            oldest = tweet.time

    GLOBAL_LOGGER.info(
        f"Total of {len(users)} different users with meta data constructed from original data."
    )
    GLOBAL_LOGGER.info(f"First tweet dates to {str(oldest)}")
    GLOBAL_LOGGER.info(f"The last tweet datest {str(latest)}")
    return users


def get_slice(data: Dict, n: int = 10) -> Dict:
    """Get slice from Dict, by default 10 first key:item pairs"""
    return dict(islice(data.items(), n))


def main():

    DATA_PATH = pathlib.Path("data/isis_twitter_data.csv")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"CSV dataset not found from the path {DATA_PATH}. Please, download and convert it."
        )
    data = read_csv_into_obj(DATA_PATH)

    users = group_users_by_tweet_meta(data)
    # Sort by tweets, get first 10 users
    users_sorted = sort_users(users)
    # users_sorted_10 = get_slice(users_sorted)
    # assert 10 == len(users_sorted_10.keys())

    # Plot all users, by amount of tweets
    plots_tweet_amount_all = False
    if plots_tweet_amount_all:
        plot_users(
            users_sorted,
            "amount_tweets",
            "User number",
            "Amount tweets",
            "Amount of tweets per user, sorted as descending",
        )
        # Plot all users, by amount of tweets, in logarithm scale
        plot_users(
            users_sorted,
            "amount_tweets",
            "User number",
            "Amount tweets",
            "Amount of tweets in logarithm scale, sorted as descending",
            plot_type="log",
        )
    users_sorted_10 = get_slice(users_sorted)
    # plot_users(
    #     users_sorted_10,
    #     "amount_tweets",
    #     "User number",
    #     "Amount tweets",
    #     "Amount of tweets per user, sorted as descending",
    #     use_name=True,
    # )
    assert 10 == len(users_sorted_10.keys())
    for sort_by, desc in zip(list(USER_META.keys())[1:], ["Amount retweets", "Amount use of mentions", "Amount use of hashtags"]):
        users_sorted = sort_users(users, by_meta_value=sort_by)
        users_sorted_10 = get_slice(users_sorted)
        plot_users(
            users_sorted_10,
            sort_by,
            "User number",
            desc,
            f"{desc} per user, sorted as descending",
            use_name=True,
        )
    # data[0].sentiment()
    # print(data[0].sentiment)
    # for tweet in data:
    #     G.add_node(tweet.username, obj=data)


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
