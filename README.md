# AskReddit Clustering

An exploration of clustering algorithms used to group responses to AskReddit posts.
Created for CS4320 - Information Storage and Retrieval at Utah State Univeristy. Source code available on
[GitHub](https://github.com/WhiteAbeLincoln/askreddit-clustering).

## Goal

Many questions on r/AskReddit recieve simple or direct responses, where the first or only sentence answers the question.

I wanted to identify and group these responses by subject/similarity. I could then determine the number of unique responses,
the number of responses in each category, and search for responses by subject.
We can view this problem as a limited version of search or categorization, where the top-level comments are documents,
and identified topics are phrase terms. In order to categorize, we must tokenize and parse the documents, identify a topic,
and normalize the responses with case-folding, stop word removal, and query expansion (normalize or link known synonyms).
We can store those results for search using an inverted index, where there is a dictionary of terms (the topics), with associated
postings lists (the responses).

To do this, we can either use a natural language method with the [Natural Language Toolkit](https://www.nltk.org/), or use clustering
on the responses and use common phrases in those clusers as the topics. There are some problems with these methods though. Not all responses
to the questions are simple, so the algorithm might choke on unexpected data. Categorization isn't at all useful if all responses are unique, since
they would then each have different topics. We also need to strip markup, such as italics, bold, and bullet-points, but this may be useful
semantic information that is lost. Finally, the different approaches for topic identification will be balances between
[precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall). I expect that the NLP method will have high precision
and low recall, while clustering may have high recall and low precision.

## Method

1. Fetch comments for a question using PRAW (the Python Reddit API Wrapper)
2. Preprocess comments by case-folding, removing [stop words](https://en.wikipedia.org/wiki/Stop_word), and [lemmatizing](https://en.wikipedia.org/wiki/Lemmatisation).
3. Use [k-means clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering) or [LDA topic modeling algorithm](https://towardsdatascience.com/lda-topic-modeling-an-explanation-e184c90aadcd) to cluster processed comments.

## Training Data

- [e4g2nm](https://old.reddit.com/r/AskReddit/comments/e4g2nm): You're a burglar, but you only steal things to slightly inconvenience your victims. What are you stealing?
- [e467d7](https://old.reddit.com/r/AskReddit/comments/e467d7): You're given a time machine fully equipped with bedroom, bathroom, cloaking device, matter replicator, and the entirety of Wikipedia, Where/When do you go first?

## Results

<figure style="text-align: center; margin-bottom: 1.5em;">
  <img
    style="width: 500px;"
    src="https://raw.githubusercontent.com/WhiteAbeLincoln/askreddit-clustering/master/results/kmeans/e467d7.png"
    alt="A visualization of the k-means algorithm">
  <figcaption>
      A plot of k-means using multi-dimensional scaling and the cosine-similarity measure.
      The data doesn't seem to be linearly seperable but the actual results were decent. It could be that this
      plotting method doesn't actually represent the behavior of the k-means algorithm, or the majority of the
      clusters do overlap, and any patterns seen in the data were incidental.
  </figcaption>
</figure>

You can find the raw results for the two algorithms [on GitHub](https://github.com/WhiteAbeLincoln/askreddit-clustering/tree/master/results).

K-means performed well on two sample threads (e467d7 and e4g2nm). The comments in discovered clusters seem to all be related,
however, as the number of comments in a cluster increased, the cohesiveness decreased. Most of the clusters with high cohesiveness
(determined by manual examination) consisted of less than 20 comments.

LDA didn't seem to be as successful as k-means in creating meaningful clusters. In one case (e467d7, topics: sleep, bedroom) it did
perform well, but most topic terms had equal probibility, leading to indistinct clusters.

## Future Ideas
- k-means tended to cluster comments which included text from the title; this isn't interesting. Perhaps filter n-grams that exist
  in the title from the documents.
- Use a quantitative evaluation mesure, either external evaluation with labeled data, or, considering the data set size, internal
  evaluation.
- Remove uninteresting/common words. Theoretically [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) should account for this,
  but it didn't seem to work in practice.
- Add a heirarchial algorithm to compare.
- Attempt proposed NLP approach, and explore the precision and recall differences.
