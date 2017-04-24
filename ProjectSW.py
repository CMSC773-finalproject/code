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
mpqaPath = "papers/subjectivity_clues_hltemnlp05/subjectivity_clues_hltemnlp05/"
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

filterSubreddit = ["Anger", "BPD", "EatingDisorders", "MMFB", "StopSelfHarm", "SuicideWatch", "addiction", "alcoholism",
                   "depression", "feelgood", "getting over it", "hardshipmates", "mentalhealth", "psychoticreddit",
                   "ptsd", "rapecounseling", "socialanxiety", "survivorsofabuse", "traumatoolbox"]

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

def readPosts(filename, users=None, sredditFilter=True):
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
                if (sredditFilter == True and ps.subReddit not in filterSubreddit):
                    posts.append(ps)
    return posts

def getPostFilename(userID):
    return str(abs(math.floor(userID / 2000.0))).split('.')[0]

def loadPosts(posUsers, negUsers, subRedditFilter = True):
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
        posPosts += readPosts(posFilename, posUsers, subRedditFilter)

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
        negPosts += readPosts(negFilename, negUsers, subRedditFilter)

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

def concatPostTitlesByUser(posts):
    """Concatenates all of the post titles by user id"""
    conRes = defaultdict(str)
    for p in posts:
        conRes[p.userID] += p.postTitle + ' '
    return conRes

def samplePostsFromUser(posts, numSample):
    # Assumption: All Posts are from the same user
    tempPosts = (p for p in posts if p.postBody != '')
    conRes = defaultdict(str)
    tmp = random.sample(tempPosts, numSample)
    counter = 1
    for p in tmp:
        conRes[str(p.userID) + '_' + str(counter)] = p.postBody
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

def sentToLIWC(tokenizedSent, filterStopWords = True):
    liwcRepresentation = {}
    for w in tokenizedSent:
        if (filterStopWords and w in stopws):
            continue
        wToliwc = getLIWCclass(w)
        if (wToliwc != None):
            for wl in wToliwc:
                if wl not in liwcRepresentation:
                    liwcRepresentation[wl] = 1
                else:
                    liwcRepresentation[wl] += 1
    retTable = []
    for lr in liwcRepresentation:
        retTable += [(lr, liwcRepresentation[lr])]
    return retTable

def postsToLIWC(posts, filterStopWords = True):
    liwcTable = {}
    for uid in posts:
        liwcTable[uid] = sentToLIWC(posts[uid], filterStopWords)
    return liwcTable

def calcD(p, q, vocab, lenP, lenQ):
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

        probP = float(cp + 1.0) / float(vlen + lenP)
        probQ = float(cq + 1.0) / float(vlen + lenQ)
        res += probP * math.log(probP / probQ, 2)
    return res

def calcAverageD(p, q, v, lenP, lenQ):
    av1 = calcD(p, q, v, lenP, lenQ)
    av2 = calcD(q, p, v, lenQ, lenP)
    return ((av1 + av2) / 2.0)

###
# Collocation Feature Exploration
###
def getVocab(ngrams):
    vocab = []
    for ngram in ngrams.values():
        for (word, value) in ngram:
            if word not in vocab:
                vocab += [word]
    return vocab

def collocations(ngram, tokenizedPosPosts, tokenizedNegPosts, vocab = None):
    """
    Generates all bigram collocations and computes their KL-divergence
    """
    print "Constructing collocations"
    allBigrams = {}
    allLengths = {}

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    wordnet_lemmatizer = WordNetLemmatizer()

    # Count n-grams
    for uid in tokenizedPosPosts:
        if (vocab == None):
            if (ngram == 1):
                tokens = (t for t in tokenizedPosPosts[uid] if t not in stopws)
                gram = nltk.FreqDist(tokens).items()
            else:
                finder = BigramCollocationFinder.from_words(tokenizedPosPosts[uid])
                finder.apply_word_filter(lambda w: w in stopws)
                gram = finder.ngram_fd.items()
        else:
            gram = tokenizedPosPosts[uid]

        allLengths[uid] = math.fsum(v for (g, v) in gram)
        allBigrams[("pos", uid)] = gram

    for uid in tokenizedNegPosts:
        if (vocab == None):
            if (ngram == 1):
                tokens = (t for t in tokenizedNegPosts[uid] if t not in stopws)
                gram = nltk.FreqDist(tokens).items()
            else:
                finder = BigramCollocationFinder.from_words(tokenizedNegPosts[uid])
                finder.apply_word_filter(lambda w: w in stopws)
                gram = finder.ngram_fd.items()
        else:
            gram = tokenizedNegPosts[uid]

        allLengths[uid] = math.fsum(v for (g, v) in gram)
        allBigrams[("neg", uid)] = gram

    # Build`token vocabulary
    if (vocab == None):
        vocab = getVocab(allBigrams)

    print "Calculating D values."
    Mt = {}
    for (class1, uid1) in allBigrams:
        for (class2, uid2) in allBigrams:
            if (uid1, uid2) in Mt:
                ansD = Mt[(uid1,uid2)]
            elif (uid2, uid1) in Mt:
                ansD = Mt[(uid2, uid1)]
            else:
                Mt[(uid1, uid2)] = calcAverageD(allBigrams[(class1,uid1)], allBigrams[(class2,uid2)], vocab, allLengths[uid1], allLengths[uid2])
                ansD = Mt[(uid1,uid2)]
            print "D (" + str(uid1) + " || " + str(uid2) + " ) = " + str(ansD)
    return Mt

def saveCollocations(filename, tableM):
    with open(filename, "w") as f:
        for (uid1,uid2) in tableM:
            line = str(uid1) + '\t' + str(uid2) + '\t' + str(tableM[(uid1, uid2)]) + '\n'
            f.write(line)

#
#   Emotion Score based on LIWC: 127 = Neg emo, 126 = Pos emo
#
def scoreEmotion(liwcConvertedPost):
    normalFact = 0
    posFact = 0
    negFact = 0
    for (w, v) in liwcConvertedPost:
        if (w == 126):
            posFact = v
        if (w == 127):
            negFact = v
        normalFact += v
    if (normalFact < 1):
        return None
    return float((- negFact)) / float(normalFact)

def calculateEmotionScoreForAllPosts(posts, filename = None):
    convertedPosts = postsToLIWC(posts)
    emoscore = {}
    for uid in convertedPosts:
        emoscore[uid] = scoreEmotion(convertedPosts[uid])
        print ("UserID: " + str(uid) + '\t' + str(emoscore[uid]))

    if (filename != None):
        with open(filename, "w") as f:
            for uid in emoscore:
                line = str(uid) + '\t' + str(emoscore[uid]) + '\n'
                f.write(line)

    return emoscore

#
#   MPQA Subjectivity and polarity
#

def loadMPQA():
    MPQADict = {}
    with open(mpqaPath + "subjclueslen1-HLTEMNLP05.tff", "r") as f:
        for line in f:
            segs = line.replace('\n','').split(' ')
            mpqa_type = segs[0].replace('type=','')
            mpqa_word = segs[2].replace('word1=','')
            mpqa_stem = segs[4]
            mpqa_stem = mpqa_stem[-1]
            mpqa_polr = segs[5].replace('priorpolarity=','')
            if (mpqa_stem == 'y'):
                if (mpqa_word[-1] == 'e'):
                    mpqa_word = mpqa_word[:-1] + '*'
                else:
                    mpqa_word += '*'
            mpqa_type = mpqa_type[0]
            if (mpqa_polr == "negative"):
                plr = -1
            elif (mpqa_polr == "positive"):
                plr = 1
            elif (mpqa_polr == "neutral"):
                plr = 0
            else:
                plr = 0
            MPQADict[mpqa_word] = (mpqa_type, plr)
    return MPQADict

def getMPQAofWord(word, mpqaDict):
    if (word in mpqaDict):
        return mpqaDict[word]
    else:
        for mpqa in mpqaDict:
            if (mpqa[-1] != '*' or len(mpqa) - 1 > len(word)):
                continue
            mpqaWord = mpqa[:-1]
            match = True
            for i in range(0,len(mpqaWord)):
                if (mpqaWord[i] != word[i]):
                    match = False
                    break
            if (match):
                return mpqaDict[mpqa]
    return None

def getSentenceMPQA(sent, mpqaDict, type = 'P'):
    totalCount = 0
    res = 0
    for word in sent:
        tp = getMPQAofWord(word, mpqaDict)
        if (tp == None):
            totalCount += 1
            continue
        sbj, polr = tp
        if (type == 'P'):
            res += int(polr)
        elif (type == 'B'):
            if(sbj == 's'):
                res += float(polr)
            else:
                res += 0.1 * float(polr)
        else:
            if(sbj == 's'):
                res += 1
            else:
                res -= 1
        totalCount += 1
    if (totalCount < 1):
        return None
    return float(res) / float(totalCount)

def getPostsMPQA(posts, mpqaDict, type = 'P', filename = None):
    mpqaTable = {}
    for uid in posts:
        mpqaTable[uid] = getSentenceMPQA(posts[uid], mpqaDict, type)

    if (filename != None):
        with open(filename, "w") as f:
            for uid in mpqaTable:
                line = str(uid) + '\t' + str(mpqaTable[uid]) + '\n'
                f.write(line)

    return mpqaTable

if __name__ == "__main__":
    #random.seed(773)

    loadIdDivisions()
    loadLIWC()

    MPQADict = loadMPQA()


    liwcVocab = []
    for c in LIWC_Classes:
        liwcVocab += [c]

    posSamples = random.sample(positivesIDs["Train"], 200)
    negSamples = random.sample(controlsIDs["Train"], 200)
    print 'Positive Samples: %s' % posSamples
    print 'Control Samples: %s' % negSamples

    print "Reading posts."

    posPosts,negPosts = loadPosts(posSamples, negSamples)

    tpPosts = tokenizePosts(concatPostsByUser(posPosts))
    tnPosts = tokenizePosts(concatPostsByUser(negPosts))

    topicPosPosts = tokenizePosts(concatPostTitlesByUser(posPosts))
    topicNegPosts = tokenizePosts(concatPostTitlesByUser(negPosts))
    #print posPosts
    #tpPosts = tokenizePosts(samplePostsFromUser(posPosts, 5))
    #tnPosts = tokenizePosts(samplePostsFromUser(negPosts, 5))

    print "Computing MPQA scores."
    getPostsMPQA(topicPosPosts, MPQADict, 'P', "results/PTmpqaPos.txt")
    getPostsMPQA(topicNegPosts, MPQADict, 'P', "results/PTmpqaNeg.txt")
    getPostsMPQA(tpPosts, MPQADict, 'P', "results/PBmpqaPos.txt")
    getPostsMPQA(tnPosts, MPQADict, 'P', "results/PBmpqaNeg.txt")
    getPostsMPQA(topicPosPosts, MPQADict, 'S', "results/STmpqaPos.txt")
    getPostsMPQA(topicNegPosts, MPQADict, 'S', "results/STmpqaNeg.txt")
    getPostsMPQA(tpPosts, MPQADict, 'S', "results/SBmpqaPos.txt")
    getPostsMPQA(tnPosts, MPQADict, 'S', "results/SBmpqaNeg.txt")

    #print "Computing LIWC representation."
    #liwcPPosts = postsToLIWC(tpPosts)
    #liwcNPosts = postsToLIWC(tnPosts)
    #calculateEmotionScoreForAllPosts(tpPosts, "results/posScore.txt")
    #calculateEmotionScoreForAllPosts(tnPosts, "results/negScore.txt")

    #saveCollocations("results/dlconv.txt", collocations(1, liwcPPosts, liwcNPosts, liwcVocab))
    #saveCollocations("results/dlconv.txt", collocations(1, tpPosts, tnPosts))


    """temp = readPost(positivesPath+"0.posts")
    for i in range(0,2):
        print temp[i].postBody
    loadLIWC()
    print LIWC_words['ace']
    print getLIWCclass('kissed')
    print translateAllLIWCClasses(getLIWCclass('kissing'))
    print translateAllLIWCClasses(getLIWCclass('a'))
    """