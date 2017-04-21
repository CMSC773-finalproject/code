import nltk
from nltk.corpus import stopwords
import string
import collections
import math
import operator
from collections import namedtuple
import random
from nltk.stem import WordNetLemmatizer
from nltk.collocations import *


controlsPath = "controls/"
positivesPath = "positives/"
liwcPath = "other_materials/liwc/"

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

#
#           Loading DEV.txt TRAIN.txt TEST.txt for user ids
#

def loadDivisions():
    with open(controlsPath+"TRAIN.txt", "r") as fc:
        for line in fc:
            controlsIDs["Train"] += [int(line)]
    with open(controlsPath+"DEV.txt", "r") as fc:
        for line in fc:
            controlsIDs["Dev"] += [int(line)]
    with open(controlsPath+"TEST.txt", "r") as fc:
        for line in fc:
            controlsIDs["Test"] += [int(line)]
    with open(positivesPath+"TEST.txt", "r") as fc:
        for line in fc:
            positivesIDs["Test"] += [int(line)]
    with open(positivesPath+"DEV.txt", "r") as fc:
        for line in fc:
            positivesIDs["Dev"] += [int(line)]
    with open(positivesPath+"TRAIN.txt", "r") as fc:
        for line in fc:
            positivesIDs["Train"] += [int(line)]

#
#       Reads all posts in the given filename and returns an array of PostsStruct s
#
def readPost(filename):
    posts = []
    with open(filename, "r") as f:
        for line in f:
            segs = line.replace('\t\n','').replace('\n','').split('\t')
            if (len(segs) < 6):
                ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), ' ', segs[3], segs[4])
            else:
                ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), segs[3], segs[4], segs[5])
            posts += [ps]
    return posts

def readPostWithID(filename, user):
    posts = []
    with open(filename, "r") as f:
        for line in f:
            segs = line.replace('\t\n','').replace('\n','').split('\t')
            if (len(segs) < 6):
                ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), ' ', segs[3], segs[4])
            else:
                ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), segs[3], segs[4], segs[5])
            if (ps.userID == user):
                posts += [ps]
    return posts

def readPostWithGroupID(filename, users):
    posts = []
    with open(filename, "r") as f:
        for line in f:
            segs = line.replace('\t\n','').replace('\n','').split('\t')
            if (len(segs) < 6):
                ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), ' ', segs[3], segs[4])
            else:
                ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), segs[3], segs[4], segs[5])
            if (ps.userID in users):
                posts += [ps]
    return posts

def concatPosts(posts):
    conRes = ""
    for p in posts:
        conRes += (p.postBody + ' ')
    return conRes

def concatGroupPosts(posts):
    conRes = {}
    for p in posts:
        if (p.userID not in conRes):
            conRes[p.userID] = p.postBody + ' '
        else:
            conRes[p.userID] += (p.postBody + ' ')
    return conRes

def tokenizeGroupPost(posts):
    tokenTable = {}
    for p in posts:
        tokenTable[p] = nltk.word_tokenize(posts[p].decode('utf-8'))
    return tokenTable

#
#           All functions required to read and interpret LIWC
#
def loadLIWC():
    with open(liwcPath+"LIWC2007.dic", "r") as f:
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
    if classID in LIWC_Classes:
        return LIWC_Classes[classID]
    return None

def translateAllLIWCClasses(cid_list):
    translist = []
    for cid in cid_list:
        translist += [translateLIWCClass(cid)]
    return translist
#
#       Calculates D(P||Q)
#

def calcD(p, q, vocab):
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

#
#               Main
#

if __name__ == "__main__":
    stopws = stopwords.words('english')

    loadDivisions()

    posSamples = random.sample(positivesIDs["Train"], 5)
    negSamples = random.sample(controlsIDs["Train"], 5)

    print posSamples
    print negSamples

    posPosts = []
    negPosts = []

    print "Reading posts."

    for i in range(0,32):
        posFilename = positivesPath + str(i) + ".posts"
        negFilename = controlsPath + str(i) + ".posts"
        if (i <= 25):
            posPosts += readPostWithGroupID(posFilename, posSamples)
        if (i >= 1):
            negPosts += readPostWithGroupID(negFilename, negSamples)

    tpPosts = tokenizeGroupPost(concatGroupPosts(posPosts))
    tnPosts = tokenizeGroupPost(concatGroupPosts(negPosts))

    print "Constructing collocations"

    allBigrams = {}
    vocab = []

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    wordnet_lemmatizer = WordNetLemmatizer()
    for p in tpPosts:
        #finder = BigramCollocationFinder.from_words(tpPosts[p])
        #finder.apply_word_filter(lambda w: w in stopws or w in string.punctuation or w in "'s 're n't 'm 've --")
        tokens = (t for t in tpPosts[p] if t not  in stopws and t not in string.punctuation and t not in "'s 're n't 'm 've --")
        monogram = nltk.FreqDist(tokens).items()
        print monogram
        #collections.Counter(tpPosts[p])

        allBigrams[("pos", p)] = monogram #finder.ngram_fd.items()
    for n in tnPosts:
        #finder = BigramCollocationFinder.from_words(tnPosts[n])
        #finder.apply_word_filter(lambda w: w in stopws or w in string.punctuation or w in "'s 're n't 'm 've --")
        tokens = (t for t in tnPosts[n] if t not  in stopws and t not in string.punctuation and t not in "'s 're n't 'm 've --")
        monogram = nltk.FreqDist(tokens).items()
        allBigrams[("neg", n)] = monogram #finder.ngram_fd.items()

    for (s, p) in allBigrams:
        temp = allBigrams[(s, p)]
        for j in range(0, len(temp)):
            tb, tv = temp[j]
            if (tb not in vocab):
                vocab += [tb]

    print "Calculating D values."
    Mt = {}
    for (s1, p1) in allBigrams:
        for (s2, p2) in allBigrams:
            if (p1, p2) in Mt:
                ansD = Mt[(p1,p2)]
            elif (p2, p1) in Mt:
                ansD = Mt[(p2, p1)]
            else:
                Mt[(p1, p2)] = calcAverageD(allBigrams[(s1,p1)], allBigrams[(s2,p2)], vocab)
                ansD = Mt[(p1,p2)]
            print "D (" + str(p1) + " || " + str(p2) + " ) = " + str(ansD)




    """temp = readPost(positivesPath+"0.posts")
    for i in range(0,2):
        print temp[i].postBody
    loadLIWC()
    print LIWC_words['ace']
    print getLIWCclass('kissed')
    print translateAllLIWCClasses(getLIWCclass('kissing'))
    print translateAllLIWCClasses(getLIWCclass('a'))
    """


