from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import sys

stops = set(stopwords.words('english'))

def get_lemmer():
  from nltk.stem.wordnet import WordNetLemmatizer
  lemmer = WordNetLemmatizer()
  return lambda tok: lemmer.lemmatize(tok)

def get_stemmer():
  from nltk.stem.porter import PorterStemmer
  stemmer = PorterStemmer()
  return lambda tok: stemmer.stem(tok)

def preprocess_filter(title):
  title_set = set(preprocess(title))
  return lambda comment: preprocess(comment, title_set=title_set)

# step 1. Preprocess the comments by tokenizing, removing stop words, and lemmatizing
def preprocess(comment, **kwargs):
  """
  Args:
    comment (str): The comment to parse

  Kwargs:
    stemmer (bool): Use stemmer over lemmer
  """
  title_set = kwargs.get('title_set', set([]))

  stemmer = get_stemmer() if (kwargs.get('stemmer') == True) else get_lemmer()
  body = comment if isinstance(comment, str) else comment.body
  tokens = [stemmer(tok) for tok in simple_preprocess(body) if tok not in stops and tok not in title_set]
  return tokens

def print_err(*args, **kwargs):
  print(*args, file=sys.stderr, **kwargs)

def print_verbose(*args, **kwargs):
  # we want to load this on every call
  # since cfg gets modified
  from config import cfg
  if cfg['--verbose']:
    print_err('VERBOSE: ', end='')
    print_err(*args, **kwargs)
