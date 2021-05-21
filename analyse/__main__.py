#!/usr/bin/env python3
import re
from sklearn.preprocessing import minmax_scale
import numpy as np
import csv
import networkx as nx
import pathlib
import logging
import argparse
import sys
import matplotlib
import pandas
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

matplotlib.use("Qt5Agg")
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import powerlaw
from typing import Union, List, Dict
from .tweetdata import TweetData, Sentiment
from copy import deepcopy
from operator import itemgetter
from itertools import islice, combinations
from collections import deque

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
    tmp_fig = None
    if plot_type == "log":
        # plt.semilogx(labels, data)
        tmp_fig = plot_powerlaw(data, x_label)
        y_label = "p(X)"
    elif plot_type == "bar":
        plt.bar(labels, data)
    # plt.axes().set_xlim(1, 1.5)
    plt.xlabel(x_label)
    # plt.x
    plt.ylabel(y_label)
    plt.title(description)
    plt.tick_params("x", labelrotation=-45)
    # for tick in plt.get_xticklabels():
    #     tick.set_rotation(90)
    # if tmp_fig:
    #     handles, labels = tmp_fig.get_legend_handles_labels()
    #     plt.legend(handles, labels, loc=3)
    plt.savefig(f'{description.replace(" ", "")}.png', bbox_inches='tight')
    plt.show()


def plot_powerlaw(data, x_label):
    """Method for attempting to fit data into powerlaw"""
    fit = powerlaw.Fit(data)
    # fit.set_xlabel(x_label)
    # fit.set_ylabel("p(X)")
    fig2 = fit.plot_pdf(color='b', linewidth=2, label=r"Probability density")
    fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig2, )
    fit.plot_ccdf(color='r', linewidth=2, ax=fig2)
    fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig2, label=r"Cumulative distribution")
    return fit
    # powerlaw.savefig
    # plt.show()

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


def plot_sentiments_with_ternary(tweets: List[TweetData]):
    # draw ternary plot
    negative_count = []
    neutral_count = []
    positive_count = []
    # This is really slow to instance, need to define outside of tweet class
    analyser = SentimentIntensityAnalyzer()
    sentiments = {
        "NEGATIVE": [],
        "NEUTRAL": [],
        "POSITIVE": [],
    }
    for tweet in tweets:
        sentiment = tweet.set_sentiment_value(analyser)
        if sentiment is Sentiment.NEGATIVE:
            negative_count.append(tweet.sentiment_value)
        elif sentiment is Sentiment.POSITIVE:
            positive_count.append(tweet.sentiment_value)
        elif sentiment is Sentiment.NEUTRAL:
            neutral_count.append(tweet.sentiment_value)
        sentiments["NEGATIVE"].append(tweet.sentiment_value.get("neg"))
        sentiments["NEUTRAL"].append(tweet.sentiment_value.get("neu"))
        sentiments["POSITIVE"].append(tweet.sentiment_value.get("pos"))

    GLOBAL_LOGGER.debug("Sentiment calculation done.")
    GLOBAL_LOGGER.info(f"Length of negative: {len(negative_count)}")
    GLOBAL_LOGGER.info(f"Length of neutral: {len(neutral_count)}")
    GLOBAL_LOGGER.info(f"Length of positive: {len(positive_count)}")
    df = pandas.DataFrame(sentiments)
    fig = px.scatter_ternary(df, b="NEGATIVE", a="NEUTRAL", c="POSITIVE")
    fig.show()


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


def main(_args: argparse.Namespace):
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
    if args.all and args.plot_tweets:
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
            "Tweet Frequency",
            "p(X)",
            "Amount of tweets fitted into probability density and cumulative distribution function",
            plot_type="log",
        )
    users_sorted_10 = get_slice(users_sorted)
    if not args.all and args.plot_tweets:
        plot_users(
            users_sorted_10,
            "amount_tweets",
            "User number",
            "Amount tweets",
            "Amount of tweets per user, sorted as descending",
            use_name=True,
        )
        assert 10 == len(users_sorted_10.keys())
        for sort_by, desc in zip(
                list(USER_META.keys())[1:],
                ["Amount retweets", "Amount use of mentions", "Amount use of hashtags"],
        ):
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

    GLOBAL_LOGGER.info("Plotting sentiments of all tweets...")
    show_vader_ternary_plot = False
    if show_vader_ternary_plot and args.all:
        plot_sentiments_with_ternary(data)
    elif show_vader_ternary_plot:
        for k in list(USER_META.keys())[1:4]:
            users_combined_10 = []
            users_sorted_10 = get_slice(sort_users(users, k))
            GLOBAL_LOGGER.info(f"Plotting ternary by {k}")
            for j in users_sorted_10.keys():
                users_combined_10 += users_sorted_10.get(j).get("tweets")
            plot_sentiments_with_ternary(users_combined_10)
    G = nx.Graph()

    create_edge_network = True
    # G = nx.petersen_graph()
    #
    # plt.subplot(121)
    #
    # nx.draw(G, with_labels=True, font_weight='bold')
    #
    # plt.subplot(122)
    #
    # nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
    # plt.show()
    if create_edge_network:
        for tweet in data:
            hashtags = tweet.get_hashtags()
            if hashtags:
                # Remove possible duplicates
                hashtags = set(hashtags)
                # print(hashtags)
                for tag in hashtags:
                    G.add_node(tag)
                if len(hashtags) > 1:
                    # Make unique pairs from hashtags, marking connection between hashtag
                    pairs = list(combinations(hashtags, 2))
                    G.add_edges_from(pairs)
        # Create secondary graph of isolated components, later used for removing isolated components from original graph
        GLOBAL_LOGGER.debug("All nodes and edges added for hashtag graph")
        GLOBAL_LOGGER.info(f"Number of isolates in the original graph: {nx.number_of_isolates(G)}")
        isolated = nx.isolates(G)
        GLOBAL_LOGGER.info("Removing isolates...")
        G.remove_nodes_from(list(isolated))
        GLOBAL_LOGGER.info(f"Number of isolates in the pruned graph: {nx.number_of_isolates(G)}")
        GLOBAL_LOGGER.info(f"There are {G.number_of_nodes()} nodes (unique hashtags) and {G.number_of_edges()}"
                           f" edges in the Graph")
        degrees = [v for (node, v) in G.degree()]
        degree_max = np.max(degrees)
        degree_min = np.min(degrees)
        # diameter = nx.diameter(G)
        average_c_coeff = nx.average_clustering(G)

        # Generate sorted list of connected components, the largest at first
        components = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
        print(components[0])
        GLOBAL_LOGGER.info(f"The maximum degree is {degree_max}")
        GLOBAL_LOGGER.info(f"The minimum degree is {degree_min}")
        GLOBAL_LOGGER.info(f"The average degree of the nodes is {np.mean(degrees):.2f}")
        GLOBAL_LOGGER.info(f"The number of connected components is {nx.number_connected_components(G)}")
        # GLOBAL_LOGGER.info(f"The diameter is {diameter:.2f}")
        GLOBAL_LOGGER.info(f"The average clustering coefficient for the network is {average_c_coeff:.2f}")

        # nx.drawing.nx_pylab.draw_networkx_nodes(G)
        # plt.show()


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
    parser.add_argument(
        "-a",
        "--all",
        dest="all",
        help="Provided when showcasing all of the user data. By default top10 used instead. "
             "Some user specific data left out",
        action="store_true"
    )
    parser.add_argument(
        "--plot-tweets",
        dest="plot_tweets",
        help="Plot tweets",
        action="store_true"
    )
    if 1 <= len(sys.argv):
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

    main(args)
