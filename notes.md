## Possible sources for article

Twitter and power laws: https://www.sciencedirect.com/science/article/pii/S1877050914005006

 The authors propose that the frequency distribution of retweets follows a power law distribution. Preferential attachment “means a user is more likely to retweet a tweet that has already been retweeted many times”.
 
 
 ### Other
 
 When presentation exported to pdf, pages 1-4, 11-12, 14-18, 27-30, 41-45, 52-53, 55, 56-63 for less duplicated content



## Yuml.me diagram generation


[TweetData|+row_id: integer;+name: string;+username: string;+description: string; +location:string;+followers: ingeter;+number_status:integer;+time: datetime;+tweet: string;+sentiment;+sentiment_value;-_sentiment:Sentiment;-_sentiment_value: Dict|__format_str_time(time);get_hastags();get_mentions();is_retweet();set_sentiment_value(analyzer: SentimentIntensityAnalyzer)]
[Sentiment|POSITIVE;NEUTRAL;NEGATIVE|_missing_(value)]
[TweetData]<>-has enum[Sentiment]

## Graphs, useful

https://github.com/ugis22/analysing_twitter/blob/master/Jupyter%20Notebook%20files/Interaction%20Network.ipynb

## SNA

Paper: https://journals.sagepub.com/doi/10.1177/016555150202800601


## Powerlaw package

Powerlaw paper: https://epubs.siam.org/doi/pdf/10.1137/070710111

https://nbviewer.jupyter.org/github/jeffalstott/powerlaw/blob/master/manuscript/Manuscript_Code.ipynb
https://medium.com/future-vision/visualizing-twitter-interactions-with-networkx-a391da239af5
https://arxiv.org/pdf/1305.0215.pdf
https://github.com/jeffalstott/powerlaw

## ISIS online exrimists

Terrorism: https://ourworldindata.org/terrorism
Severity: https://onlinelibrary.wiley.com/doi/full/10.1111/ssqu.12721

https://www.hindawi.com/journals/complexity/2019/1238780/

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181405#pone.0181405.ref001


Similar than this work:
https://ieeexplore.ieee.org/document/8078846

https://www.researchgate.net/publication/335136503_Linguistic_analysis_of_pro-ISIS_users_on_Twitter

## Tweets

https://www.sciencedirect.com/science/article/pii/S1877050914005006