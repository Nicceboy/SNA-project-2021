# SNA-project-2021

The project work for Social Network Analysis, iteration for year 2021

## Initial assignment 

### Project 7. Analysis of ISIS Twitter dataset

This project aims to investigate the social network of ISIS Twitter network published in dark web website. It corresponds to 17,000 tweets from 100+ pro-ISIS fanboys from all over the world since the November2015 Paris Attacks. The dataset includes the following: Name, Username, Description, Location, Number of followers (at the time the tweet was downloaded), Number of statuses by the user (when the tweet was downloaded), Date and timestamp of the tweet, the tweet itself. The dataset is made available through the following link https://1drv.ms/x/s!AtcJs3OTsMZuiRSaq7O2ZdVysiXd

  1. Use appropriate NetworkX functions to plot the distribution of the number of tweet per user id. Discuss whether a power law can be fitted to this plot.

  2. Now we want to explore the content of tweet, and seek whether a mention (@), Retweet (RT) or hashtag (#) is present. Provide a plot showing the 10 most active users in terms of retweets, use of mentions and use of hashtags in their tweet messages(a graph for each of thesethree categories). 
  
  3. Use VADER tool (https://github.com/cjhutto/vaderSentiment), which output sentiment in terms of POSITIVE, NEGATIVE and NEUTRAL to determine the sentiment of each tweet of the dataset. Then represent the distribution of each tweet as a point in the ternary plot.
  4. Repeat the above process for the tweets associated to the ten most active users in Retweets, use of mentions, and use of hashtags. Should display three different ternary plot.
  
  5. Now we want to build a social graph where each node corresponds to a hashtag and an edge between hashtag A and hashtag B indicates that there is at least one tweet which contains both hashtag A and hashtag B. Implement a small python program that allows you to generate the above social network graph.
  
  6. Summarize in a table the main global properties of the above graph: Number of nodes, Number of edges, diameter,clustering coefficient, size of largest component using appropriate NetworkX functions. Comment on the obtained results highlighting any inherent limitation or characteristic of the data collection process.
  
  7. Plot the page-rank degree distribution and the local clustering coefficient distribution.
  
  8. Use Girvan-Newman algorithm to find communities in the above network through appropriate use of NetworkX functions. Compare the size of the generated communities in a table.Using your understanding of the name of the hashtag, speculate whether each community can be assigned some interpretation. 
  
  9. Using the information about the amount of support as a function of number of followers and number statuses of each tweet (you can suggest an expression of support of your own), rank the various hashtags according to the amount support they received accordingly. Note: the amount of support for a hashtag should be the sum of the amount of support of all tweets where this hashtag is mention.
  
  10. We want to reduce the size of the graph in 5) by requiring that an edge between nodes is established only if there are k number of tweets mentioning the two hashtags(nodes). Set different values fork, say, 2, 3, 4, 5, 10 for instance, and draw the corresponding graph using appropriate tool.Provide in a table global attributes of each graph in terms of size of network, diameter, clustering coefficient, average degree centrality, average closeness centrality, average inbetweeness centrality.
  
  11. Draw heat map of degree centrality distribution for each value of the threshold . [see examples of heat maps in https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0220061
  
  12. Comment on the results using identified literature in security and terrorism of your choice.


### Download Excel data (Linux)

Data is in `data` folder and excluded from version control.

```console
mkdir -p data && \
curl -o data/isis_twitter_data.xlsx -L "https://onedrive.live.com/download?resid=6EC6B09373B309D7\!1172&
ithint=file%2cxlsx&authkey=AJqrs7Zl1XKyJd0"
```

### Init Python environment (Linux)

Python 3.9.3 version initially used, lower versions might be applicable.

```console
python -m venv venv && \
source venv/bin/activate && \
pip install -r requirements.txt
```

### Inline conversion from .xlsx to .csv with column names

```console
python -c 'import pandas;data_xlsx = pandas.read_excel("data/isis_twitter_data.xlsx", "Sheet1", dtype=str, index_col=None, engine="openpyxl"); data_xlsx.to_csv("data/isis_twitter_data.csv", encoding="utf-8", index=True)'
```

### Usage

Currently main function code can be run as:

```console
python -m analyse
```

## Tests

Some tests are applied to ensure that data is extracted correctly.

To execute tests:

```console
pytest -v tests/test_tweetdata.py
```