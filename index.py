"""AskReddit Classifier

Usage:
  arc.py [-hqv] [-s STRAT] [THREAD...]

Options:
  -h --help                     Show this help
  -q --quiet                    Quiet output
  -v --verbose                  Verbose output
  -s STRAT, --strategy=STRAT    Choose between 'nlp', 'clustering', and 'topic_lda'
"""
from docopt import docopt
from dotenv import load_dotenv
import os
import praw

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

def get_top_comments(s):
    from praw.models import MoreComments
    comments = []
    s.comments.replace_more(limit=0)
    for c in s.comments:
        comments += [c]

    return comments

def process(submission, args=None):
    print(get_top_comments(submission))

def interactive(getter, args=None):
    print('Enter a thread url or id. Type exit to quit')
    try:
        while True:
            thread = input('> ')
            if thread.lower() == 'exit' or thread == 'q':
                break
            process(getter(thread), args)
    except (EOFError,KeyboardInterrupt) as err:
        exit()

def main(args):
    load_dotenv()
    reddit = init_reddit(os.getenv('CLIENT_ID'), os.getenv('CLIENT_SECRET'))
    getter = submission_getter(reddit)

    if (not args['THREAD']):
        interactive(getter, args)
    else:
        for t in args['THREAD']:
            submission = getter(t)
            process(t, args)

if __name__ == '__main__':
    main(docopt(__doc__, version='0.1'))

