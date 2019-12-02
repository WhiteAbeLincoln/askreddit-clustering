import gensim
from config import cfg
from util import preprocess, preprocess_filter
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, TfidfModel
import numpy as np
import os

def add_elem(d, key, val):
  arr = d.get(key, [])
  arr.append(val)
  d[key] = arr

def bag_of_words_corpus(docs):
  dictionary = Dictionary(docs)
  # filter those words that only appear in 4 documents or those
  # in over half of the documents
  dictionary.filter_extremes(no_below=4, no_above=0.5)
  # make bag-of-words representation of the documents
  return (dictionary, [dictionary.doc2bow(doc) for doc in docs])

def tfidf(corpus):
  tfidf_model = TfidfModel(corpus)
  return (tfidf_model, tfidf_model[corpus])

chunk_size = 2000
passes = 100
iterations = 400
eval_every = None

def execute(comments, submission):
  prepper = preprocess_filter(submission.title) if cfg['--filter-title'] else preprocess

  docs = list(map(prepper, comments))
  dictionary, corpus_bow = bag_of_words_corpus(docs)
  _, corpus_tfidf = tfidf(corpus_bow)

  # Make a index to word dictionary.
  _ = dictionary[0]  # This is only to "load" the dictionary.
  model = LdaMulticore(
    corpus=corpus_tfidf,
    num_topics=cfg['--clusters'],
    id2word=dictionary,
    workers=len(os.sched_getaffinity(0)),
    chunksize=chunk_size,
    passes=passes,
    iterations=iterations,
    eval_every=eval_every
  )

  topics_matrix = model.show_topics(formatted=False, num_words=15)

  for i, word_pairs in topics_matrix:
    print(f'Topic {i}')
    for word in word_pairs:
      print(f' {word}')

  print('\n\n')

  results = {}

  for i, comment in enumerate(corpus_bow):
    # we only want to add the most probable topic if it exists
    # since comments will belong to multiple topics with differing probability
    for index,_ in sorted(model[comment], key=lambda tup: -1*tup[1]):
      add_elem(results, index, comments[i])
      break

  for key in results.keys():
    clustered = map(lambda c: c.body, results[key])
    print(f'[{key}]')
    for c in clustered:
      print(f">  {c}\n")
    print('\n')
