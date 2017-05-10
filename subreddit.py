# -*- encoding: utf-8 -*-

import sys
import pdb
import requests
import json
import code
import codecs

from classifier import DataLoader

def get_desc(subreddit):
    headers = {'User-agent': 'cmsc773 colink@umd.edu'}
    r = requests.get('https://www.reddit.com/r/%s/about.json' % subreddit, headers=headers)

    if r.status_code == 200:
        return json.loads(r.text)["data"]["public_description"]
    elif r.status_code == 429:
        # Rate-limiting
        print "Error: %s (%s)" % (json.loads(r.text)["message"], json.loads(r.text)["error"])
        print r.headers
        # assert False # just fail
        return '<RATE_LIMITED>'
    elif r.status_code == 404:
        # No description
        return ''
    elif r.status_code == 403:
        # No description
        return '<FORBIDDEN>'
    else:
        pdb.set_trace()

def clean_desc(desc):
    return desc.replace('\n', ' ')

def generate_subreddit_set():
    data = DataLoader()
    data.loadIdDivisions()
    data.readAllSamples(sredditFilter=False)
    subreddits = set()
    for posts in [data.posPosts,data.negPosts]:
        for post in posts:
            subreddits.add(post.subReddit)
    return subreddits

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == "load":
        """
        Usage:
        $ python subreddit.py load subreddits.txt

        Reads all posts from the training dataset and writes
        the set of subreddits into `subreddits.txt`
        """
        subreddits = generate_subreddit_set()
        with open(sys.argv[2], 'w') as sr_f:
            for subreddit in subreddits:
                sr_f.write("%s\n" % subreddit)
    elif len(sys.argv) == 4 and sys.argv[1] == "scrape":
        """
        Usage:
        $ python subreddit.py scrape subreddits.txt descriptions.txt

        Scrapes the public description of every subreddit (one-per-line) in
        `subreddits.txt`. Outputs each description (again, one-per-line in the same
        order) into `descriptions.txt`.
        """
        with open(sys.argv[2], "r") as sr_f:
            with codecs.open(sys.argv[3],"w",'utf-8') as dsc_f:
                for subreddit in sr_f:
                    desc = get_desc(subreddit.strip())
                    cleaned_desc = clean_desc(desc)
                    dsc_f.write("%s\n" % cleaned_desc)
