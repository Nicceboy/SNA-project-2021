#!/usr/bin/env python3
# from sklearn.preprocessing import minmax_scale
import argparse
import csv
import logging
import pathlib
import sys

import matplotlib
import networkx as nx
import numpy as np
import pandas
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

matplotlib.use("Qt5Agg")
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import powerlaw
from typing import List, Dict
from .tweetdata import TweetData, Sentiment
from copy import deepcopy
from itertools import islice, combinations

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
        show: bool = False
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
    if not description:
        f_name = "default"
    else:
        f_name = description.replace(" ", "")
    plt.savefig(f'{f_name}.png', bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_powerlaw(data, x_label):
    """Method for attempting to fit data into powerlaw"""
    fit = powerlaw.Fit(data, discrete=True)
    fit.distribution_compare('power_law', 'lognormal')
    fig = fit.plot_ccdf(linewidth=3, label='Tweet Data')
    fit.power_law.plot_ccdf(ax=fig, color='r', linestyle='--', label='Power law fit')
    fit.truncated_power_law.plot_ccdf(ax=fig, color='y', linestyle='--', label='Truncated Power law fit')
    fit.lognormal.plot_ccdf(ax=fig, color='g', linestyle='--', label='Lognormal fit')
    fit.exponential.plot_ccdf(ax=fig, color='b', linestyle='--', label='Exponential fit')
    ####
    fig.set_ylabel(u"p(X≥x)")
    # fig.set_xlabel("y")
    handles, labels = fig.get_legend_handles_labels()
    fig.legend(handles, labels, loc=3)
    R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    GLOBAL_LOGGER.info(f"Comparing distribution fit: power law vs exponential: {R:.2f} vs {p:.2f}")
    R, p = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    GLOBAL_LOGGER.info(f"Comparing distribution fit: power law vs lognormal: {R:.2f} vs {p:.2f}")
    R, p = fit.distribution_compare('exponential', 'lognormal', normalized_ratio=True)
    GLOBAL_LOGGER.info(f"Comparing distribution fit: exponential vs lognormal: {R:.2f} vs {p:.2f}")
    R, p = fit.distribution_compare('power_law', 'truncated_power_law')
    GLOBAL_LOGGER.info(f"Comparing distribution fit: power_law vs truncated_power_law: {R:.2f} vs {p:.2f}")
    R, p = fit.distribution_compare('lognormal', 'truncated_power_law')
    GLOBAL_LOGGER.info(f"Comparing distribution fit: lognormal vs truncated_power_law: {R:.2f} vs {p:.2f}")
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
        show: bool = False
):
    if plot_by not in USER_META.keys():
        raise ValueError("Key used for plotting must be found from Dict 'USER_META'")
    # Take 5 first characters, hide rest
    if use_name:
        names = [k[:5] + "*" for k in users.keys()]
    else:
        names = range(0, len(users.keys()))
    num = [users.get(k).get(plot_by) for k in users.keys()]
    plot(names, num, x_label, y_label, description, plot_type, show)


def plot_sentiments_with_ternary(tweets: List[TweetData], show: bool = False, group: str = ""):
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
    name = group if group else "all"
    path = f"article/figures/sentiment_ternary_{name}.png"
    GLOBAL_LOGGER.info(f"Writing ternary plot of sentiment analysis into path {path}")
    fig.write_image(path)
    if show:
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


def get_graph_info(G: nx.Graph):
    """Calculate core information of graph"""
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
    GLOBAL_LOGGER.info(f"The maximum degree is {degree_max}")
    GLOBAL_LOGGER.info(f"The minimum degree is {degree_min}")
    GLOBAL_LOGGER.info(f"The average degree of the nodes is {np.mean(degrees):.2f}")
    GLOBAL_LOGGER.info(f"The number of connected components is {nx.number_connected_components(G)}")
    GLOBAL_LOGGER.info(f"The size of the largest component is {components[0]}")
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        GLOBAL_LOGGER.info(f"The diameter is {diameter:.2f}")
    else:
        S = max(nx.connected_components(G), key=len)
        diameter = nx.diameter(G.subgraph(S).copy())
        GLOBAL_LOGGER.info(
            f"All nodes are not connected, estimated length based on the subgraph"
            f" created from the biggest component: {diameter}")

    GLOBAL_LOGGER.info(f"The average clustering coefficient for the network is {average_c_coeff:.2f}")


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
            # "Amount of tweets fitted into CCDF, compared to different distribution candidates",
            plot_type="log",
            show=args.show
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
            show=args.show
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
                "Username",
                desc,
                f"{desc} per user, sorted as descending",
                use_name=True,
                show=args.show
            )

    GLOBAL_LOGGER.info("Plotting sentiments of all tweets...")
    if args.sentiment and args.all:
        plot_sentiments_with_ternary(data, args.show)
    elif args.sentiment:
        for k in list(USER_META.keys())[1:4]:
            users_combined_10 = []
            users_sorted_10 = get_slice(sort_users(users, k))
            GLOBAL_LOGGER.info(f"Plotting ternary by {k}")
            for j in users_sorted_10.keys():
                users_combined_10 += users_sorted_10.get(j).get("tweets")
            plot_sentiments_with_ternary(users_combined_10, args.show, k)
    G = nx.Graph()

    create_edge_network = True
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
        # Reduce network node amount by hashtag occurrences in tweets
        if args.k:
            # Remove isolated notes at first
            isolated = nx.isolates(G)
            GLOBAL_LOGGER.info("Removing isolates...")
            G.remove_nodes_from(list(isolated))
            # Remove nodes based on amount of connections: amount of appearances of tweets with hashtag pairs
            filter_by = [node for (node, v) in G.degree() if v < int(args.k)]
            G.remove_nodes_from(filter_by)
        # Create secondary graph of isolated components, later used for removing isolated components from original graph
        GLOBAL_LOGGER.debug("All nodes and edges added for hashtag graph")
        get_graph_info(G)
        if args.show:
            Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
            pos = nx.spring_layout(Gcc)
            nx.draw_networkx_nodes(Gcc, pos, node_size=10)
            nx.draw_networkx_edges(Gcc, pos, alpha=0.4)
            plt.show()

        if args.page_rank:
            GLOBAL_LOGGER.info("Creating plot for PageRank.")
            pr = nx.pagerank_numpy(G, alpha=0.9)
            # Sort rank of every key-value pair
            pr_rank = sorted(pr.items(), key=lambda x: x[1], reverse=True)
            values = [k[1] for k in pr_rank]
            amount = range(1, len(values) + 1)
            plt.scatter(amount, values)
            plt.xlabel("Rank of hashtag")
            plt.ylabel("PageRank value")
            plt.title("PageRank distribution of hashtag popularity")
            for index, item in enumerate(pr_rank[:10]):
                GLOBAL_LOGGER.info(f"Rank: {index + 1}, hashtag: {item[0]}")
            plt.savefig(f'article/figures/pagerank_distribution.png', bbox_inches='tight')
            if args.show:
                plt.show()
            else:
                plt.close()

        if args.lcc:
            clusters = nx.clustering(G)
            plt.hist(clusters.values(), bins=10)
            plt.title("Distribution of LCC")
            plt.xlabel('Clustering')
            plt.ylabel('Frequency')
            plt.savefig(f'article/figures/lcc_distribution.png', bbox_inches='tight')
            if args.show:
                plt.show()
            else:
                plt.close()

        if args.girvann:
            GLOBAL_LOGGER.info(f"Starting Girvan-Newmann analysis.")
            k = 1
            comp = nx.algorithms.community.centrality.girvan_newman(G)
            test = 1
            for communities in islice(comp, k):
                data = tuple(sorted(c) for c in communities)
                print(len(data))
                # with open(f"file{test}.txt", "w") as f:
                #     f.write(str(data))
                # test += 1
            # with open("file1.new.txt") as f:
            #     data = json.load(f).get("data")
            #     GLOBAL_LOGGER.info(f"Amount of communities: {len(data)}")
            # print(data[0])
            # Calculate pagerank

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
        "-s",
        "--show",
        dest="show",
        help="Additionally show plots on GUI instead only saving them as file.",
        action="store_true"
    )
    parser.add_argument(
        "-k",
        dest="k",
        help="Reduced graph size by requiring k amount of ",
        nargs='?',
        default=0
    )
    parser.add_argument(
        "--plot-tweets",
        dest="plot_tweets",
        help="Plot tweets",
        action="store_true"
    )
    parser.add_argument(
        "--sentiment",
        dest="sentiment",
        help="Make sentiment analysis for data.",
        action="store_true"
    )
    parser.add_argument(
        "--lcc",
        dest="lcc",
        help="Make Local Clustering Coefficient distribution plot for hashtag network",
        action="store_true"
    )
    parser.add_argument(
        "--page-rank",
        dest="page_rank",
        help="Make and plot PageRank distribution.",
        action="store_true"
    )
    parser.add_argument(
        "--girvann",
        dest="girvann",
        help="Make Girvan-Newman analysis to find communities from the network.",
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
