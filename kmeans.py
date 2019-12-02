# Much of the implementation code was derived from
# http://brandonrose.org/clustering

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.utils import simple_preprocess
from util import preprocess, preprocess_filter, print_verbose
from config import cfg

COLORS = [
  "#FFFFB3",
  "#FF803E",
  "#FFFF68",
  "#FFA6BD",
  "#FFC100",
  "#FFCEA2",
  "#FF8170",
  "#FF007D",
  "#FFF676",
  "#FF0053",
  "#FFFF7A",
  "#FF5337",
  "#FFFF8E",
  "#FFB328",
  "#FFF4C8",
  "#FF7F18",
  "#FF93AA",
  "#FF5933",
  "#FFF13A",
  "#FF232C"
]

def vectorize(comment_doc, preprocessor=preprocess):
  vectorizer = TfidfVectorizer(use_idf=True,
                               ngram_range=(1,3),
                               tokenizer=preprocessor,
                              )
  return (vectorizer.fit_transform(comment_doc), vectorizer)

def add_elem(d, key, val):
  arr = d.get(key, [])
  arr.append(val)
  d[key] = arr

def plot(comment_doc_mat, kmeans, order_centroids, terms):
  # method from: http://brandonrose.org/clustering
  from sklearn.metrics.pairwise import cosine_similarity
  from sklearn.manifold import MDS
  import pandas as pd
  import matplotlib as mpl
  mpl.use('Qt5Cairo')
  import matplotlib.pyplot as plt
  clusters = kmeans.labels_.tolist()
  dist = 1 - cosine_similarity(comment_doc_mat)
  MDS()

  # convert two components as we're plotting points in a two-dimensional plane
  # "precomputed" because we provide a distance matrix
  # we will also specify `random_state` so the plot is reproducible.
  mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
  pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

  xs, ys = (pos[:, 0], pos[:, 1])
  df = pd.DataFrame(dict(x=xs,y=ys,label=clusters))
  groups = df.groupby('label')
  _,ax = plt.subplots(figsize=(17,9))
  ax.margins(0.05)

  for name, group in groups:
    ax.plot(group.x,
            group.y,
            marker='o',
            linestyle='',
            ms=12,
            color=COLORS[name],
            label=",".join(map(lambda j: terms[j], order_centroids[name, :3])),
            mec='none'
          )
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
  ax.legend(numpoints=1)
  plt.show()

def filter_preprocessor(title):
  title_set = set(preprocess(title))
  def preprocessor(comment):
    pre = preprocess(comment)
    return [k for k in pre if k not in title_set]
  return preprocessor

def execute(comments, submission):
  prepper = preprocess_filter(submission.title) if cfg['--filter-title'] else preprocess
  # all the comments as single document
  (comment_doc_mat,vectorizer) = vectorize(map(lambda c: c.body, comments), preprocessor=prepper)

  terms = vectorizer.get_feature_names()

  num_clusters = cfg['--clusters']
  model = KMeans(n_clusters=num_clusters, n_jobs=-1)
  model.fit(comment_doc_mat)

  # sort cluster centers by proximity to centroid
  order_centroids = model.cluster_centers_.argsort()[:, ::-1]
  for i in range(num_clusters):
    print(f"Cluster {i}")
    for j in order_centroids[i, :15]:
      print(f" {terms[j]}")

  print('\n\n')

  cluster_dict = {}

  for comment in comments:
    body = comment.body
    X = vectorizer.transform([body])
    predicted = model.predict(X)
    add_elem(cluster_dict, str(predicted), comment)

  for key in cluster_dict.keys():
    clustered = map(lambda c: c.body, cluster_dict[key])
    print(key)
    for c in clustered:
      print(f">  {c}\n")
    print('\n')

  if cfg['--plot']:
    print_verbose('Plotting clusters')
    plot(comment_doc_mat, model, order_centroids, terms)

  return cluster_dict
