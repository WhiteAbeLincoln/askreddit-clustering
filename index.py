"""AskReddit Classifier

Usage:
  arc.py [-hqvnpf] [-k K] [-l N] [-s STRAT] [THREAD...]

Options:
  -h --help                     Show this help
  -q --quiet                    Quiet output
  -v --verbose                  Verbose output
  -s STRAT, --strategy=STRAT    Choose between 'hierarchial', 'k-means', and 'lda'. [default: k-means]
  -k K, --clusters=K            Number of clusters to use. [default: 8]
  -l N, --comment-limit=N       Maximum number of extra comments to fetch. Use -1 to get all. [default: 0]
  -n --no-cache                 Disable comment cache
  -p --plot                     Plot result
  -f --filter-title                Filter terms present in the title
"""
from docopt import docopt
from dotenv import load_dotenv
import os
import praw
import config
from xdg import XDG_CACHE_HOME
from os.path import join, exists
import pickle
from util import print_err, print_verbose

CACHE_NAME = 'askreddit_classifier'
CACHE_BASEDIR = join(XDG_CACHE_HOME, CACHE_NAME)

def ready_cfg(args):
    if (not args['--strategy']):
        args['--strategy'] = 'topic_lda'

    if (args['--clusters']):
        args['--clusters'] = int(args['--clusters'])

    if (args['--comment-limit']):
        args['--comment-limit'] = int(args['--comment-limit'])

    config.cfg = args

def init_reddit(cid, secret):
    reddit = praw.Reddit(
        client_id=cid,
        client_secret=secret,
        user_agent='AskReddit Categorization script'
    )
    reddit.read_only = True

    return reddit

def submission_getter(reddit):
    def getter(thread):
        if thread.startswith('http'):
            return reddit.submission(url=thread)
        else:
            return reddit.submission(id=thread)

    return getter

def read_from_cache(id, limit):
    try:
        print_verbose(f'Reading {id} from cache')
        with open(join(CACHE_BASEDIR, f'{limit}_{id}'), 'rb') as f:
            return pickle.load(f)
    except:
        return (-1,None)

def write_to_cache(id, limit, comments):
    try:
        print_verbose(f'Writing {id} to cache')
        with open(join(CACHE_BASEDIR, f'{limit}_{id}'), 'wb') as f:
            return pickle.dump((limit, comments), f)
    except:
        print_err(f'Failed to cache {id}')

def get_top_comments(s):
    limit = None if config.cfg['--comment-limit'] == -1 else config.cfg['--comment-limit']
    (c_limit, c_comments) = (-1,None) if config.cfg['--no-cache'] else read_from_cache(s.id, limit)

    comments = None if limit != c_limit else c_comments

    if comments is None:
        print_verbose(f'Getting comments for {s}')
        comments = []
        s.comments.replace_more(limit=limit)
        for c in s.comments:
            comments += [c]

    if not config.cfg['--no-cache']:
        write_to_cache(s.id, limit, comments)

    return comments

def process(submission):
    comments = get_top_comments(submission)
    method = config.cfg['--strategy']
    exec_fn = None
    if method == 'lda':
        from lda import execute
        exec_fn = execute
    elif method == 'k-means':
        from kmeans import execute
        exec_fn = execute
    elif method == 'hierarchial':
        from heirarchial import execute
        exec_fn = execute
    else:
        raise ValueError(f'Invalid strategy: {method}')

    print_verbose(f'Executing method {method} for {submission.id}')
    exec_fn(comments, submission)

def interactive(getter):
    print('Enter a thread url or id. Type exit to quit')
    try:
        while True:
            thread = input('> ')
            if thread.lower() == 'exit' or thread == 'q':
                break
            process(getter(thread))
    except (EOFError,KeyboardInterrupt):
        exit()

def main(args):
    ready_cfg(args)

    if not exists(CACHE_BASEDIR):
        print_verbose('Creating cache directory')
        from os import makedirs
        makedirs(CACHE_BASEDIR)
        # only download on first run
        print_verbose('Downloading nltk corpus')
        import nltk
        nltk.download('wordnet')
        nltk.download('stopwords')

    load_dotenv()

    reddit = init_reddit(os.getenv('CLIENT_ID'), os.getenv('CLIENT_SECRET'))
    getter = submission_getter(reddit)

    if (not args['THREAD']):
        interactive(getter)
    else:
        for t in args['THREAD']:
            print(t,'\n')
            process(getter(t))

if __name__ == '__main__':
    main(docopt(__doc__, version='0.1'))
