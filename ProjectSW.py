# -*- coding: utf-8 -*-
import pdb
import nltk
from nltk.corpus import stopwords
import string
import collections
import math
import operator
from collections import namedtuple, defaultdict
import random
from nltk.stem import WordNetLemmatizer
from nltk.collocations import *
from os.path import join,isfile
import pickle
from itertools import chain

controlsPath = "controls/"
positivesPath = "positives/"
liwcPath = "other_materials/liwc/"
picklePath = 'tmp/'

controlsIDs = {}
positivesIDs = {}
controlsIDs["Train"] = []
controlsIDs["Dev"] = []
controlsIDs["Test"] = []
positivesIDs["Train"] = []
positivesIDs["Dev"] = []
positivesIDs["Test"] = []

LIWC_Classes = {}
LIWC_words = {}

PostsStruct = namedtuple("PostsStruct", "postID userID timeStamp subReddit postTitle postBody")

stopws = stopwords.words('english')
# TODO: does string.punctuation need to be converted to unicode?
stopws.extend(string.punctuation)
stopws.extend(u"'s 're n't 'm 've 'd '' 't --".split())

###
# File I/O
###

def loadIdDivisions():
    """
    Loading DEV.txt TRAIN.txt TEST.txt which contain user ids
    """
    for dataset in ['Train', 'Dev', 'Test']:
        filename = dataset.upper() + '.txt'
        with open(join(controlsPath,filename), 'r') as fc:
            controlsIDs[dataset] = [int(line) for line in fc]
        with open(join(positivesPath,filename), 'r') as fc:
            positivesIDs[dataset] = [int(line) for line in fc]

def readPosts(filename, users=None):
    """
    Returns the list of PostStructs from a given file
    Optionally filters using a set of user ids
    """
    posts = []
    with open(filename, 'r') as f:
        for line in f:
            # Split on only the first 5 tabs (sometimes post bodies have tabs)
            segs = line.strip().split('\t', 5)
            # Add empty post body, if necessary (image posts, f.e.)
            if len(segs) == 5: segs.append('')
            ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), segs[3], segs[4], segs[5])
            if users is None or ps.userID in users:
                posts.append(ps)
    return posts

def getPostFilename(userID):
    return str(abs(math.floor(userID / 2000.0))).split('.')[0]

def loadPosts(posUsers, negUsers):
    """
    Loads all posts from a set of positive users and negative users
    Caches posts in a pickle object
    """
    # Check pickle cache
    pickle_filename = join(picklePath, str(hash(tuple(posUsers + negUsers))) + ".pickle")
    if isfile(pickle_filename):
        print 'Loading posts from cache'
        with open(pickle_filename, 'rb') as f:
            return pickle.load(f)

    posPosts = []
    negPosts = []

    postFilenames = []

    # Read positive posts (filenames: 0.txt...25.txt)
    for puid in posUsers:
        tmp = getPostFilename(puid)
        if tmp not in postFilenames:
            postFilenames += [tmp]
    for filename in postFilenames:
        posFilename = join(positivesPath, filename + '.posts')
        posPosts += readPosts(posFilename, posUsers)

    #for i in xrange(26):
    #    posFilename = join(positivesPath, str(i) + '.posts')
    #    posPosts += readPosts(posFilename, posUsers)

    # Read control posts (filenames: 1.txt...31.txt)
    postFilenames = []

    for nuid in negUsers:
        tmp = getPostFilename(nuid)
        if tmp not in postFilenames:
            postFilenames += [tmp]
    for filename in postFilenames:
        negFilename = join(controlsPath, filename + '.posts')
        negPosts += readPosts(negFilename, negUsers)

    #for i in xrange(1,32):
    #    negFilename = join(controlsPath, str(i) + '.posts')
    #    negPosts += readPosts(negFilename, negUsers)

    # Write to pickle cache
    with open(pickle_filename, 'wb') as f:
        pickle.dump((posPosts, negPosts), f, protocol=pickle.HIGHEST_PROTOCOL)

    return posPosts, negPosts

###
# Data pre-processing
###

def concatPosts(posts):
    """Concatenates all of the post bodies"""
    return ' '.join((p.postBody for p in posts))

def concatPostsByUser(posts):
    """Concatenates all of the post bodies by user id"""
    conRes = defaultdict(str)
    for p in posts:
        conRes[p.userID] += p.postBody + ' '
    return conRes

def tokenizePosts(posts):
    tokenTable = {}
    for uid in posts:
        tokenTable[uid] = nltk.word_tokenize(posts[uid].decode('utf-8'))
    return tokenTable

###
# Functions for reading and interpreting LIWC
###

def loadLIWC():
    with open(join(liwcPath, 'LIWC2007.dic'), 'r') as f:
        percCounter = 0
        for line in f:
            if (line[0] == '%'):
                percCounter += 1
            else:
                if (percCounter < 2):
                    segs = line.replace('\n','').split('\t')
                    LIWC_Classes[int(segs[0])] = segs[1]
                else:
                    segs = line.replace('\n','').split('\t')
                    if segs[0] == 'like' or segs[0] == 'kind':
                        continue
                    cli = []
                    for i in range(1,len(segs)):
                        if(segs[i] != ''):
                            cli += [int(segs[i])]
                    LIWC_words[segs[0]] = cli

def getLIWCclass(word):
    if word in LIWC_words:
        return LIWC_words[word]
    else:
        for liwcW in LIWC_words:
            if (len(liwcW) - 1 > len(word)):
                continue
            if liwcW[-1] == '*':
                tempWord = liwcW[:-1]
                contained = True
                for i in range(0,len(tempWord)):
                    if tempWord[i] != word[i]:
                        contained = False
                        break
                if (contained):
                    return LIWC_words[liwcW]
    return None

def translateLIWCClass(classID):
    return LIWC_Classes[classID] if classID in LIWC_Classes else None

def translateAllLIWCClasses(cid_list):
    return [translateLIWCClass(cid) for cid in cid_list]

def calcD(p, q, vocab):
    """Calculates D(P||Q)"""
    res = 0
    vlen = len(vocab)
    for w in vocab:
        cp = 0
        cq = 0
        for i in range(0, len(p)):
            if (w in p[i]):
                pb, pv = p[i]
                cp = pv
                break
        for i in range(0, len(q)):
            if (w in q[i]):
                qb, qv = q[i]
                cq = qv
                break

        probP = float(cp + (1.0 / vlen)) / float(vlen + 1.0)
        probQ = float(cq + (1.0 / vlen)) / float(vlen + 1.0)
        res += probP * math.log(probP / probQ, 2)
    return res

def calcAverageD(p, q, v):
    av1 = calcD(p, q, v)
    av2 = calcD(q, p, v)
    return ((av1 + av2) / 2.0)

###
# Collocation Feature Exploration
###

def collocations(ngram, tokenizedPosPosts, tokenizedNegPosts):
    """
    Generates all bigram collocations and computes their KL-divergence
    """
    print "Constructing collocations"
    allBigrams = {}

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    wordnet_lemmatizer = WordNetLemmatizer()
    vocab = {}

    # Count n-grams
    for uid in tokenizedPosPosts:
        vocab[uid] = []
        if (ngram == 1):
            tokens = (t for t in tokenizedPosPosts[uid] if t not in stopws)
            gram = nltk.FreqDist(tokens).items()
        else:
            finder = BigramCollocationFinder.from_words(tokenizedPosPosts[uid])
            finder.apply_word_filter(lambda w: w in stopws)
            gram = finder.ngram_fd.items()
        for (w, v) in gram:
            if (w not in vocab[uid]):
                vocab[uid] += [w]
        allBigrams[("pos", uid)] = gram

    for uid in tokenizedNegPosts:
        vocab[uid] = []
        if (ngram == 1):
            tokens = (t for t in tokenizedNegPosts[uid] if t not in stopws)
            gram = nltk.FreqDist(tokens).items()
        else:
            finder = BigramCollocationFinder.from_words(tokenizedNegPosts[uid])
            finder.apply_word_filter(lambda w: w in stopws)
            gram = finder.ngram_fd.items()
        for (w, v) in gram:
            if (w not in vocab[uid]):
                vocab[uid] += [w]
        allBigrams[("neg", uid)] = gram

    # Build`token vocabulary
    #vocab = {}
    #for bigrams in allBigrams.values():
    #    for token, count in bigrams:
    #        if (token not in vocab):
    #            vocab.append(token)

    print "Calculating D values."
    Mt = {}
    for (class1, uid1) in allBigrams:
        for (class2, uid2) in allBigrams:
            if (uid1, uid2) in Mt:
                ansD = Mt[(uid1,uid2)]
            elif (uid2, uid1) in Mt:
                ansD = Mt[(uid2, uid1)]
            else:
                Mt[(uid1, uid2)] = calcAverageD(allBigrams[(class1,uid1)], allBigrams[(class2,uid2)], vocab[uid1] + vocab[uid2])
                ansD = Mt[(uid1,uid2)]
            print "D (" + str(uid1) + " || " + str(uid2) + " ) = " + str(ansD)
    return Mt

def saveCollocations(filename, tableM):
    with open(filename, "w") as f:
        for (uid1,uid2) in tableM:
            line = str(uid1) + '\t' + str(uid2) + '\t' + str(tableM[(uid1, uid2)]) + '\n'
            f.write(line)

if __name__ == "__main__":
    random.seed(773)

    loadIdDivisions()

    posSamples = random.sample(positivesIDs["Train"], 5)
    negSamples = random.sample(controlsIDs["Train"], 5)
    print 'Positive Samples: %s' % posSamples
    print 'Control Samples: %s' % negSamples

    print "Reading posts."

    posPosts,negPosts = loadPosts(posSamples, negSamples)

    tpPosts = tokenizePosts(concatPostsByUser(posPosts))
    tnPosts = tokenizePosts(concatPostsByUser(negPosts))

    saveCollocations("results/dlconv.txt", collocations(2, tpPosts, tnPosts))


    """temp = readPost(positivesPath+"0.posts")
    for i in range(0,2):
        print temp[i].postBody
    loadLIWC()
    print LIWC_words['ace']
    print getLIWCclass('kissed')
    print translateAllLIWCClasses(getLIWCclass('kissing'))
    print translateAllLIWCClasses(getLIWCclass('a'))
    """