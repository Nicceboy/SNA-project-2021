#!/usr/bin/env python3
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
import plotly.graph_objects as go
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


def get_graph_info(G: nx.Graph, k: int = 0):
    """Calculate core information of the graph"""
    GLOBAL_LOGGER.info(f"Number of isolates in the graph: {nx.number_of_isolates(G)}")
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    GLOBAL_LOGGER.info(f"There are {nodes} nodes (unique hashtags) and {edges}"
                       f" edges in the Graph")
    # degrees = [v for (node, v) in G.degree()]
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    degree_max = degrees[0][1]
    degree_min = degrees[-1][1]
    deg_mean = round(np.mean([v for (node, v) in degrees]), 2)
    # diameter = nx.diameter(G)
    average_c_coeff = round(nx.average_clustering(G), 2)
    average_closeness_cent = round(np.mean(list(map(lambda x: x[1], nx.closeness_centrality(G).items()))), 2)
    average_betweenness_cent = round(np.mean(list(map(lambda x: x[1], nx.betweenness_centrality(G).items()))), 2)
    # average_betweenness_cent = np.mean(nx.betweenness_centrality(G))
    num_conn_comp = nx.number_connected_components(G)
    # Generate sorted list of connected components, the largest at first
    components = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    GLOBAL_LOGGER.info(f"Top 10 nodes in degree centrality by name:")
    pathlib.Path("article/misc/").mkdir(parents=True, exist_ok=True)
    h_file = pathlib.Path(f"article/misc/top_hashtags_k_{k}")
    with h_file.open("w") as f:
        start = """\\begin{table}[ht]
\\begin{tabular} { | c | c | }
\\hline
Rank & Hashtag \\\\
\\hline"""
        rank = 1
        print(start, file=f)
        for node, _ in degrees:
            print(f"{rank} & {node} \\\\", file=f)
            print("\\hline", file=f)
            rank += 1
        end = f"""
\\end{{tabular}}
\\caption {{Ranked hashtags by Degree Centrality when k = {k}}}
\\label{{tab:hashtag-ranks-k-{k}}}
\\end{{table}}"""
        print(end, file=f)
    GLOBAL_LOGGER.info(f"Ranked nodes in degree centrality by name into file {str(h_file)}")
    GLOBAL_LOGGER.info(f"The maximum degree is {degree_max}")
    GLOBAL_LOGGER.info(f"The minimum degree is {degree_min}")
    GLOBAL_LOGGER.info(f"The average degree of the nodes is {deg_mean:.2f}")
    GLOBAL_LOGGER.info(f"The number of connected components is {num_conn_comp}")
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
    GLOBAL_LOGGER.info(f"The average closeness centrality for the network is {average_closeness_cent:.2f}")
    GLOBAL_LOGGER.info(f"The average betweenness centrality for the network is {average_betweenness_cent:.2f}")
    # Return LaTeX formatted table row
    return nodes, edges, components, diameter, average_c_coeff, deg_mean, average_closeness_cent, average_betweenness_cent


def create_network(data: List[TweetData], k: int = 0) -> nx.Graph:
    """Create network, limit size by occurrences of tweet hashtag pairs
    Graph will be undirected - order of hashtag usage does not matter"""
    pair_occurences = {}
    edges = []
    G = nx.Graph()
    for tweet in data:
        hashtags = tweet.get_hashtags()
        if hashtags:
            # Remove possible duplicates
            hashtags = set(hashtags)
            # for tag in hashtags:
            #     G.add_node(tag)
            if len(hashtags) > 1:
                # Make unique pairs from hashtags, marking connection between hashtag
                combs = list(combinations(hashtags, 2))
                for comb in combs:
                    if not pair_occurences.get(comb) and not pair_occurences.get((comb[1], comb[0])):
                        pair_occurences[comb] = 1
                    else:
                        # Swap tuple if it exist already - direction in graph does not matter here
                        if pair_occurences.get((comb[1], comb[0])):
                            comb = (comb[1], comb[0])
                        pair_occurences[comb] += 1
    # occur_sorted = {str(k): v for k, v in sorted(pair_occurences.items(), key=lambda x: x[1], reverse=True)}
    # with open("occurrences.json", "w") as f:
    #     json.dump(occur_sorted, f)
    # exit()
    for key, value in pair_occurences.items():
        if value < k:
            continue
        else:
            edges.append(key)
    # finally add edges
    G.add_edges_from(edges)
    return G


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

    if args.user_hashtags:
        user_h = users_sorted.get(args.user_hashtags)
        rankings = {}

        for tweet in user_h.get("tweets"):
            hashtags = tweet.get_hashtags()
            for h in hashtags:
                if not rankings.get(h):
                    rankings[h] = 1
                else:
                    rankings[h] += 1
        # Top 10 hashtags by selected user
        sorted_ranks = {k: v for (k, v) in sorted(rankings.items(), key=lambda x: x[1], reverse=True)[:10]}
        from pprint import pprint
        pprint(sorted_ranks, sort_dicts=False)

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

    if args.sentiment:
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

    # Reduce network node amount by hashtag occurrences in tweets
    if args.k and args.create_network:
        # Collect k and degree centrality value pairs
        # plotted as heatmap later
        k_degree = []
        _table = """\\begin{table*}[ht]
        \\begin{tabular} { | c | c | c | c | c | c | c | c | c | }
        \\hline
        k & Nodes \\# & Edges \\# & Largest component & Diameter & Avg. Clustering Coef. & Avg. Degree & Avg. Closeness & Avg. Betweenness \\\\
        \\hline"""
        for _k in args.k:
            G = create_network(data, int(_k))
            nodes, edges, components, diameter, average_c_coeff, deg_mean, average_closeness_cent, average_betweenness_cent = get_graph_info(
                G, _k)
            k_degree.append((_k, [v for v in G.degree()]))
            tab_row = f"\t\t{_k} & {nodes} & {edges} & {components[0]} & {diameter} & " \
                      f"{average_c_coeff} & {deg_mean} & {average_closeness_cent} & {average_betweenness_cent} \\\\"
            _table += "\n"
            _table += tab_row + "\n\t\t\\hline"

        _table += """\n
        \\end{tabular}
        \\caption {Core measurements of hashtag graph by minimal hashtag pair occurrences (k)}
        \\label {tab:graphstats-by-k}
        \\end{table*}"""
        GLOBAL_LOGGER.info("Following LaTeX table autogenerated:")
        print(_table)
        # Example command
        # python -m analyse -k  1 2 3 4 5 10 15 --show -lDEBUG
        if args.heatmap:
            GLOBAL_LOGGER.info("Plotting heatmap of degree centrality based on k values")
            # degrees = [sorted(map(lambda x: x[1], v), reverse=True) for (k, v) in k_degree]
            degrees = [list(map(lambda x: x[1], v)) for (k, v) in k_degree]
            # hashtags = [list(map(lambda x: x[0], v)) for (k, v) in k_degree]
            fig = go.Figure(data=go.Heatmap(
                z=degrees,
                zmax=100,
                zmin=1,
                y=[k[0] for k in k_degree],
                colorscale='Inferno'))
            path = "article/figures/hashtag_heatmap_by_k.png"
            fig.write_image(path)
            if args.show:
                fig.show()

    elif args.create_network:
        G = create_network(data)
        get_graph_info(G)
    GLOBAL_LOGGER.debug("All nodes and edges added for hashtag graph")
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
        k = 7
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
        help="Reduce graph size by requiring k amount of tweets with hashtag pairs. Give list of values, e.g. -k 1 2 3 5",
        nargs='+',
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
        "--create-network",
        dest="create_network",
        help="Create hashtag network",
        action="store_true"
    )
    parser.add_argument(
        "--heatmap",
        dest="heatmap",
        help="Plot heatmap by occurrences of tweet hashtag pairs.",
        action="store_true"
    )
    parser.add_argument(
        "--user-hashtags",
        dest="user_hashtags",
        help="Show amount of hashtag use by user",
        nargs='?',
        default=""
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
