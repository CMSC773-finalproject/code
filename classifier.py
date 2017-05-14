# -*- coding: utf-8 -*-
import pdb
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.collocations import *
import nltk.classify
from nltk.stem.porter import PorterStemmer
import string
import collections
import math
import operator
from collections import namedtuple, defaultdict
import random

from os.path import join,isfile
import pickle
from itertools import chain
from operator import itemgetter
from sklearn.svm import LinearSVC
from gensim import corpora, models
import gensim

controlsPath = "controls/"
positivesPath = "positives/"
liwcPath = "other_materials/liwc/"
mpqaPath = "papers/subjectivity_clues_hltemnlp05/subjectivity_clues_hltemnlp05/"
picklePath = 'tmp/'


PostsStruct = namedtuple("PostsStruct", "postID userID timeStamp subReddit postTitle postBody")

stopws = stopwords.words('english')
# TODO: does string.punctuation need to be converted to unicode?
stopws.extend(string.punctuation)
stopws.extend(u"'s 're n't 'm 've 'd '' 't -- `` ... .. ** +_ __".split())

filterSubreddit = ["Anger", "BPD", "EatingDisorders", "MMFB", "StopSelfHarm", "SuicideWatch", "addiction", "alcoholism",
                   "depression", "feelgood", "getting over it", "hardshipmates", "mentalhealth", "psychoticreddit",
                   "ptsd", "rapecounseling", "socialanxiety", "survivorsofabuse", "traumatoolbox"]


similarPositives = [50583, 46515, 40386, 36662, 36229, 35064, 26069, 26015, 28419, 18947, 16451, 16360, 15334, 14883, 13231,
                    10317, 9889, 9768, 8877, 7971, 5047, 4447, 1936, 1104, 741]

filterPostsAfterSW = False

firstPersonPronouns = ['i', 'me', 'myself', 'mine', 'my', 'we', 'us', 'our', 'ours', 'ourselves']
allProunouns = firstPersonPronouns + ['you', 'yours', 'your', 'yourself', 'yourselves', 'he', 'she', 'his', 'her',
                                      'hers', 'himself', 'herself', 'they', 'them', 'their', 'theirs', 'it', 'its',
                                      'themselves', 'itself']
###
# File I/O
###

class DataLoader:
    def __init__(self):
        self.controlsIDs = {}
        self.positivesIDs = {}
        self.controlsIDs["Train"] = []
        self.controlsIDs["Dev"] = []
        self.controlsIDs["Test"] = []
        self.positivesIDs["Train"] = []
        self.positivesIDs["Dev"] = []
        self.positivesIDs["Test"] = []
        self.loadIdDivisions()
        self.posPosts = []
        self.negPosts = []
        self.posLIWCPosts = []
        self.negLIWCPosts = []

    def loadIdDivisions(self):
        """
        Loading DEV.txt TRAIN.txt TEST.txt which contain user ids
        """
        for dataset in ['Train', 'Dev', 'Test']:
            filename = dataset.upper() + '.txt'
            with open(join(controlsPath,filename), 'r') as fc:
                self.controlsIDs[dataset] = [int(line) for line in fc]
            with open(join(positivesPath,filename), 'r') as fc:
                self.positivesIDs[dataset] = [int(line) for line in fc]

    def getPostFilename(self, userID):
        return str(abs(math.floor(userID / 2000.0))).split('.')[0]

    def clearPosts(self):
        self.posPosts = []
        self.negPosts = []

    def getRandomSample(self, number, setType = "Train", sredditFilter=True, fileType = "Both"):
        posSamples = random.sample(self.positivesIDs[setType], number)
        negSamples = random.sample(self.controlsIDs[setType], number)
        if (fileType != "Word"):
            self.loadLIWCConvertedPosts(posSamples, negSamples, sredditFilter)
        if (fileType != "LIWC"):
            self.loadPosts(posSamples, negSamples, sredditFilter)

    def readSimilarPositives(self, sredditFilter=True, fileType = "Both"):
        if (fileType != "Word"):
            self.loadLIWCConvertedPosts(similarPositives, None, sredditFilter)
        if (fileType != "LIWC"):
            self.loadPosts(similarPositives, None, sredditFilter)

    def readAllSamples(self, setType = "Train", sredditFilter=True, fileType = "Both"):
        if (fileType != "Word"):
            self.loadLIWCConvertedPosts(self.positivesIDs[setType], self.controlsIDs[setType], sredditFilter)
        if (fileType != "LIWC"):
            self.loadPosts(self.positivesIDs[setType], self.controlsIDs[setType], sredditFilter)

    def readPosts(self, filename, users=None, sredditFilter=True):
        """
        Returns the list of PostStructs from a given file
        Optionally filters using a set of user ids
        """
        posts = []
        skipusers = []
        with open(filename, 'r') as f:
            for line in f:
                # Split on only the first 5 tabs (sometimes post bodies have tabs)
                segs = line.strip().split('\t', 5)
                # Add empty post body, if necessary (image posts, f.e.)
                if len(segs) == 5: segs.append('')
                ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), segs[3], segs[4], segs[5])
                if users is None or ps.userID in users:
                    if (ps.subReddit == "SuicideWatch" and filterPostsAfterSW == True):
                        skipusers += [ps.userID]
                    if (ps.userID in skipusers):
                        continue
                    if (sredditFilter == True and ps.subReddit not in filterSubreddit):
                        posts.append(ps)
        return posts

    def loadPosts(self, posUsers, negUsers, subRedditFilter = True):
        """
        Loads all posts from a set of positive users and negative users
        Caches posts in a pickle object
        """
        # Check pickle cache
        #pickle_filename = join(picklePath, str(hash(tuple(posUsers + negUsers))) + ".pickle")
        #if isfile(pickle_filename):
        #    print 'Loading posts from cache'
        #    with open(pickle_filename, 'rb') as f:
        #        return pickle.load(f)

        if posUsers != None:
            postFilenames = []
            # Read positive posts (filenames: 0.txt...25.txt)
            for puid in posUsers:
                tmp = self.getPostFilename(puid)
                if tmp not in postFilenames:
                    postFilenames += [tmp]
            for filename in postFilenames:
                posFilename = join(positivesPath, filename + '.posts')
                self.posPosts += self.readPosts(posFilename, posUsers, subRedditFilter)

        # Read control posts (filenames: 1.txt...31.txt)
        if negUsers != None:
            postFilenames = []

            for nuid in negUsers:
                tmp = self.getPostFilename(nuid)
                if tmp not in postFilenames:
                    postFilenames += [tmp]
            for filename in postFilenames:
                negFilename = join(controlsPath, filename + '.posts')
                self.negPosts += self.readPosts(negFilename, negUsers, subRedditFilter)

        # Write to pickle cache
        #with open(pickle_filename, 'wb') as f:
        #    pickle.dump((self.posPosts, self.negPosts), f, protocol=pickle.HIGHEST_PROTOCOL)

    def getPositivePosts(self):
        return self.posPosts

    def getControlsPosts(self):
        return self.negPosts

    def getLIWCPositivePosts(self):
        return self.posLIWCPosts

    def getLIWCControlsPosts(self):
        return self.negLIWCPosts

    def loadLIWCConvertedPosts(self, posUsers, negUsers, subRedditFilter = True):
        if posUsers != None:
            postFilenames = []
            # Read positive posts (filenames: 0.txt...25.txt)
            for puid in posUsers:
                tmp = self.getPostFilename(puid)
                if tmp not in postFilenames:
                    postFilenames += [tmp]
            for filename in postFilenames:
                posFilename = join(positivesPath, filename + '.pliwc')
                self.posLIWCPosts += self.readLIWCConvertedPosts(posFilename, posUsers, subRedditFilter)

        if negUsers != None:
            # Read control posts (filenames: 1.txt...31.txt)
            postFilenames = []
            for nuid in negUsers:
                tmp = self.getPostFilename(nuid)
                if tmp not in postFilenames:
                    postFilenames += [tmp]
            for filename in postFilenames:
                negFilename = join(controlsPath, filename + '.pliwc')
                self.negLIWCPosts += self.readLIWCConvertedPosts(negFilename, negUsers, subRedditFilter)


    def readLIWCConvertedPosts(self, filename, users=None, sredditFilter=True):
        posts = []
        skipusers = []
        with open(filename, 'r') as f:
            for line in f:
                postTitle = []
                postBody = []
                # Split on only the first 5 tabs (sometimes post bodies have tabs)
                segs = line.strip().split('\t', 5)
                # Add empty post body, if necessary (image posts, f.e.)
                if len(segs) == 5: segs.append('')
                elif len(segs) < 5:
                    continue
                pTitle = segs[4].split(' ')
                for pt in pTitle:
                    liwcTitle = pt.split(',')
                    if (len(liwcTitle) < 2):
                        continue
                    postTitle += [(int(liwcTitle[0]), int(liwcTitle[1]))]
                pBody = segs[5].split(' ')
                for pb in pBody:
                    liwcBody = pb.split(',')
                    if (len(liwcBody) < 2):
                        continue
                    postBody += [(int(liwcBody[0]), int(liwcBody[1]))]
                ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), segs[3], postTitle, postBody)
                if users is None or ps.userID in users:
                    if (ps.subReddit == "SuicideWatch" and filterPostsAfterSW == True):
                        skipusers += [ps.userID]
                    if (ps.userID in skipusers):
                        continue
                    if (sredditFilter == True and ps.subReddit not in filterSubreddit):
                        posts.append(ps)
                    elif (sredditFilter == False):
                        posts.append(ps)
        return posts

###
# Data pre-processing
###
class PostProcessing:
    def __init__(self, posts):
        self.rawPosts = posts

    def concatPosts(self):
        """Concatenates all of the post bodies"""
        self.concatinatedPosts = ' '.join((p.postBody for p in self.rawPosts))

    def concatLIWCPosts(self):
        conRes = {}
        conRes[1] = []
        for p in self.rawPosts:
            conRes[1] += p.postBody
        self.concatinatedPosts = conRes

    def concatPostsByUser(self):
        """Concatenates all of the post bodies by user id"""
        conRes = defaultdict(str)
        for p in self.rawPosts:
            conRes[p.userID] += p.postBody + ' '
        self.concatinatedPosts = conRes

    def concatLIWCPostsByUser(self):
        """Concatenates all of the post bodies by user id"""
        conRes = {}
        for p in self.rawPosts:
            if p.userID not in conRes:
                conRes[p.userID] = p.postBody
            else:
                conRes[p.userID] += p.postBody
        self.concatinatedPosts = conRes

    def concatPostTitlesByUser(self):
        """Concatenates all of the post titles by user id"""
        conRes = defaultdict(str)
        for p in self.rawPosts:
            conRes[p.userID] += p.postTitle + ' '
        self.concatinatedTitles = conRes

    def concatLIWCPostTitlesByUser(self):
        """Concatenates all of the post titles by user id"""
        conRes = {}
        for p in self.rawPosts:
            if p.userID not in conRes:
                conRes[p.userID] = p.postTitle
            else:
                conRes[p.userID] += p.postTitle
        self.concatinatedTitles = conRes

    def getConcatPostBodies(self):
        return self.concatinatedPosts

    def getConcatPostTitles(self):
        return self.concatinatedTitles

    def separatePostCorpusWrtSW(self):
        """Concatenates all of the post bodies by user id  and returns 2 series, post before and after SuicideWatch"""
        beforeCorpus = defaultdict(str)
        afterCorpus = defaultdict(str)
        swPostSeen = {}
        for i in range(0, len(self.rawPosts)):
            p = self.rawPosts[i]
            if p.userID not in swPostSeen:
                swPostSeen[p.userID] = False

            if (swPostSeen[p.userID]):
                if p.subReddit not in filterSubreddit:
                    afterCorpus[p.userID] += p.postBody + ' '
            else:
                if p.subReddit == "SuicideWatch":
                    swPostSeen[p.userID] = True
                if p.subReddit not in filterSubreddit:
                    beforeCorpus[p.userID] += p.postBody + ' '
        return (beforeCorpus, afterCorpus)

    def separateLIWCPostCorpusWrtSW(self):
        """Concatenates all of the post bodies by user id  and returns 2 series, post before and after SuicideWatch"""
        beforeCorpus = {}
        afterCorpus = {}
        swPostSeen = {}
        for i in range(0, len(self.rawPosts)):
            p = self.rawPosts[i]
            if p.userID not in swPostSeen:
                swPostSeen[p.userID] = False
            if p.userID not in beforeCorpus:
                beforeCorpus[p.userID] = []
                afterCorpus[p.userID] = []

            if (swPostSeen[p.userID]):
                if p.subReddit not in filterSubreddit:
                    afterCorpus[p.userID] += p.postBody
            else:
                if p.subReddit == "SuicideWatch":
                    swPostSeen[p.userID] = True
                if p.subReddit not in filterSubreddit:
                    beforeCorpus[p.userID] += p.postBody
        return (beforeCorpus, afterCorpus)

class PostProcessingHelpers:

    def getVocabulary(self, tokenizedPosts):
        vocab = set()
        for uid in tokenizedPosts:
            vocab.union(set(tokenizedPosts[uid]))
        return vocab

    def tokenizePosts(self, posts, filterStopWords = True):
        tokenTable = {}
        for uid in posts:
            tokenTable[uid] = nltk.word_tokenize(posts[uid].lower().decode('utf-8'))
            if (filterStopWords):
                tokenTable[uid] = [t for t in tokenTable[uid] if t not in stopws and t != '']
        return tokenTable

    def stemmSinglePost(self, tokenizedPost):
        p_stemmer = PorterStemmer()
        return [p_stemmer.stem(w) for w in tokenizedPost]

    def stemmPosts(self, tokenizedPosts):
        stemmedTable = {}
        for uid in tokenizedPosts:
            stemmedTable[uid] = self.stemmSinglePost(tokenizedPosts[uid])
        return stemmedTable

    def getVocabularyFromPosts(self, posPosts, negPosts, numVocab):
        vcb = []
        for uid in posPosts:
            vcb += posPosts[uid]
        for uid in negPosts:
            vcb += negPosts[uid]
        vocab = list(nltk.FreqDist(vcb))
        if (len(vocab) <= numVocab or numVocab < 0):
            return vocab
        return vocab[:numVocab]

    def unwrapLIWCPost(self, concatedPosts, filterStopWs = True):
        cpRes = {}
        for uid in concatedPosts:
            cpRes[uid] = []
            for (lc, lv) in concatedPosts[uid]:
                if (lc <= 10 ):
                    continue
                for i in range(0,lv):
                    cpRes[uid] += [lc]
        return cpRes

    def getEmoScoreForLIWCConvertedPosts(self, concatedPosts, liwcDict, emoscoreType = "Neg"):
        emsRes = {}
        for uid in concatedPosts:
            emsRes[uid] = liwcDict.scoreEmotionUnrwappedSent(concatedPosts[uid], emoscoreType)
        return emsRes

    def getEmoScoreForLIWCConvertedPost(self, concatedPost, liwcDict, emoscoreType = "Neg", bucketing = True):
        rtval = liwcDict.scoreEmotionUnrwappedSent(concatedPost, emoscoreType)
        if rtval == None:
            rtval = 0

        if (bucketing):
            emscore = math.fabs(rtval)
            if (emoscoreType != "Both"):
                rtval = int(emscore * 100.0)
                if (rtval > 10):
                    rtval = 10
            else:
                rtval = int ((emscore + 0.1) * 50.0)
                if (rtval < 0):
                    rtval = 0
                if (rtval > 10):
                    rtval = 10
        return rtval


###
# Functions for reading and interpreting LIWC
###

class LIWCProcessor:
    def __init__(self):
        self.LIWC_Classes = {}
        self.LIWC_words = {}
        self.loadLIWC()

    def loadLIWC(self):
        with open(join(liwcPath, 'LIWC2007.dic'), 'r') as f:
            percCounter = 0
            for line in f:
                if (line[0] == '%'):
                    percCounter += 1
                else:
                    if (percCounter < 2):
                        segs = line.replace('\n','').split('\t')
                        self.LIWC_Classes[int(segs[0])] = segs[1]
                    else:
                        segs = line.replace('\n','').split('\t')
                        if segs[0] == 'like' or segs[0] == 'kind':
                            continue
                        cli = []
                        for i in range(1,len(segs)):
                            if(segs[i] != ''):
                                cli += [int(segs[i])]
                        self.LIWC_words[segs[0]] = cli

    def getLIWCclass(self, word):
        if word in self.LIWC_words:
            return self.LIWC_words[word]
        else:
            for liwcW in self.LIWC_words:
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
                        return self.LIWC_words[liwcW]
        return None

    def translateLIWCClass(self, classID):
        return self.LIWC_Classes[classID] if classID in self.LIWC_Classes else None

    def translateAllLIWCClasses(self, cid_list):
        return [self.translateLIWCClass(cid) for cid in cid_list]

    def sentToLIWC(self, tokenizedSent, filterStopWords = True):
        liwcRepresentation = {}
        for w in tokenizedSent:
            if (filterStopWords and w in stopws):
                continue
            wToliwc = self.getLIWCclass(w)
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

    def postsToLIWC(self, posts, filterStopWords = True):
        liwcTable = {}
        for uid in posts:
            liwcTable[uid] = self.sentToLIWC(posts[uid], filterStopWords)
        return liwcTable

    def getLIWCVocab(self):
        liwcVocab = []
        for c in self.LIWC_Classes:
            liwcVocab += [c]
        return liwcVocab

    #
    #   Emotion Score based on LIWC: 127 = Neg emo, 126 = Pos emo
    #
    def scoreEmotion(self, liwcConvertedPost, scoreType = "Neg"):
        normalFact = 0
        posFact = 0
        negFact = 0
        for (w, v) in liwcConvertedPost:
            if (w == 126):
                posFact += v
            if (w == 127):
                negFact += v
            normalFact += v
        if (normalFact < 1):
            return None
        numerator = -negFact
        if (scoreType == "Pos"):
            numerator = posFact
        elif (scoreType == "Both"):
            numerator = posFact - negFact
        return float(numerator) / float(normalFact)

    def scoreEmotionUnrwappedSent(self, liwcUnwrappedPost, scoreType = "Neg"):
        normalFact = 0
        posFact = 0
        negFact = 0
        for w in liwcUnwrappedPost:
            if (w == 126):
                posFact += 1
            if (w == 127):
                negFact += 1
            normalFact += 1
        if (normalFact < 1):
            return None
        numerator = -negFact
        if (scoreType == "Pos"):
            numerator = posFact
        elif (scoreType == "Both"):
            numerator = posFact - negFact
        return float(numerator) / float(normalFact)

    def calculateEmotionScoreForAllPosts(self, posts, filename = None):
        convertedPosts = self.postsToLIWC(posts)
        emoscore = {}
        for uid in convertedPosts:
            emoscore[uid] = self.scoreEmotion(convertedPosts[uid])
            print ("UserID: " + str(uid) + '\t' + str(emoscore[uid]))

        if (filename != None):
            with open(filename, "w") as f:
                for uid in emoscore:
                    line = str(uid) + '\t' + str(emoscore[uid]) + '\n'
                    f.write(line)
        return emoscore


class SupervisedClassifier:
    def __init__(self, liwc, vocab, lda):
        self.liwcProcesses = liwc
        self.vocab = vocab
        self.helpers = PostProcessingHelpers()
        self.ldaModel = lda
        self.truePositives = None

    def addTruPositive(self, tp):
        self.truePositives = tp

    # Feature fromat:
    # ({'Feature name' : Feature value}, class)
    def unigramFeatureSet(self, tokenizedSentence, liwcFeature = False):
        usingvocab = self.vocab
        if liwcFeature:
            usingvocab = self.liwcProcesses.getLIWCVocab()
        ngram_features = {}
        for word in usingvocab:
            ngram_features[word] = (word in tokenizedSentence)
        return ngram_features

    def bigramCollocationFeatureSet(self, tokenizedSentece):
        ngram_features = {}
        bigramSent = self.getBigramCollocation(tokenizedSentece)
        for bigram in self.bigramVocab:
            ngram_features[bigram] = self.isBigramIncluded(bigram, bigramSent)
        return ngram_features

    def isBigramIncluded(self, bigram, bigramSent):
        for b, n in bigramSent:
            if bigram == b:
                return True
        return False

    def emotionScoreFeatureSet(self, liwcUnwarpedSentence):
        negScore = self.helpers.getEmoScoreForLIWCConvertedPost(liwcUnwarpedSentence, self.liwcProcesses, "Neg")
        posScore = self.helpers.getEmoScoreForLIWCConvertedPost(liwcUnwarpedSentence, self.liwcProcesses, "Pos")
        avgScore = self.helpers.getEmoScoreForLIWCConvertedPost(liwcUnwarpedSentence, self.liwcProcesses, "Both")
        score_features = {}
        score_features['NEGEMO'] = negScore
        score_features['POSEMO'] = posScore
        score_features['AVGEMO'] = avgScore
        return score_features

    def ldaTopicModelFeatureSet(self, tokenizedSentence):
        feature_set = {}
        if tokenizedSentence == []:
            return feature_set
        ldatopicsN, ldatopicsP = self.ldaModel.getTopic(tokenizedSentence)
        if (ldatopicsP != None):
            for lt, lp in ldatopicsP:
                tag = "LDATOPIC" + str(lt)
                val = int(lp * 20.0)
                feature_set[tag] = val
        if (ldatopicsN != None):
            for lt, lp in ldatopicsN:
                tag = "LDATOPIC" + str(lt)
                val = int(lp * 20.0)
                feature_set[tag] = val
        return feature_set

    def similarityWithTPFeatureSet(self, liwcUnWrappedSentence):
        feature_set = {}
        res = self.collocationsFeature(self.truePositives, liwcUnWrappedSentence, vocab=self.vocab)
        if res > 1.0:
            res = 1.0
        feature_set["SimTP"] = int(round(res * 100.0))
        return feature_set

    def getFeatureSetForAllPosts(self, posTokenizedPosts, negTokenizedPosts, posLIWCPosts, negLIWCPosts):
        feature_set = []
        #for uid in posTokenizedPosts:
            #feature_set += [(self.unigramFeatureSet(posTokenizedPosts[uid]), 'pos')]
            #feature_set += [(self.ldaTopicModelFeatureSet(posTokenizedPosts[uid]), 'pos')]
            #feature_set += [(self.bigramCollocationFeatureSet(posTokenizedPosts[uid]), 'pos')]
        for uid in posLIWCPosts:
            #feature_set += [(self.unigramFeatureSet(posLIWCPosts[uid], liwcFeature=True), 'pos')]
            feature_set += [(self.similarityWithTPFeatureSet(posLIWCPosts[uid]), 'pos')]
        #    feature_set += [(self.emotionScoreFeatureSet(posLIWCPosts[uid]), 'pos')]
        #for uid in negTokenizedPosts:
            #feature_set += [(self.bigramCollocationFeatureSet(negTokenizedPosts[uid]), 'neg')]
            #feature_set += [(self.unigramFeatureSet(negTokenizedPosts[uid]), 'neg')]
            #feature_set += [(self.ldaTopicModelFeatureSet(negTokenizedPosts[uid]), 'neg')]
        for uid in negLIWCPosts:
            #feature_set += [(self.unigramFeatureSet(negLIWCPosts[uid], liwcFeature=True), 'neg')]
            feature_set += [(self.similarityWithTPFeatureSet(negLIWCPosts[uid]), 'neg')]
        #    feature_set += [(self.emotionScoreFeatureSet(negLIWCPosts[uid]), 'neg')]

        random.shuffle(feature_set)
        return feature_set

    #def getFeatureSetForAPost(self,tokenizedPost, classification):
    #    feature_set = [(self.unigramFeatureSet(tokenizedPost), classification)]
    #    feature_set += [(self.emotionScoreFeatureSet(tokenizedPost), classification)]
    #    return feature_set

    def trainClassifier(self, posTokenizedPosts, negTokenizedPosts, posLIWCPosts, negLIWCPosts, classifierType = "NB"):
        print "Training " + classifierType + "!"
        feature_set = self.getFeatureSetForAllPosts(posTokenizedPosts, negTokenizedPosts, posLIWCPosts, negLIWCPosts)
        if classifierType == "NB":
            self.classifier = nltk.NaiveBayesClassifier.train(feature_set)
            self.classifier.show_most_informative_features(50)
        elif classifierType == "SVM":
            self.classifier = nltk.classify.SklearnClassifier(LinearSVC())
            self.classifier.train(feature_set)
        elif classifierType == "Maxent":
            self.classifier = nltk.MaxentClassifier.train(feature_set)
        else:
            self.classifier = nltk.DecisionTreeClassifier.train(feature_set)
            print self.classifier.pseudocode(depth=10)


    def classifierAccuracy(self, posTokenizedPostsTest, negTokenizedPostsTest, posLIWCPosts, negLIWCPosts):
        feature_set = self.getFeatureSetForAllPosts(posTokenizedPostsTest, negTokenizedPostsTest, posLIWCPosts, negLIWCPosts)
        return nltk.classify.accuracy(self.classifier, feature_set)

    def classifierConfusionMatrix(self, posTokenizedPostsTest, negTokenizedPostsTest, posLIWCPosts, negLIWCPosts):
        gold = self.getFeatureSetForAllPosts(posTokenizedPostsTest, negTokenizedPostsTest, posLIWCPosts, negLIWCPosts)
        feature_set = self.removeClassesFromFeatures(gold)
        predictions = self.classifier.classify_many(feature_set)
        gs = []
        for i in range(0, len(gold)):
            g, v = gold[i]
            gs += [v]
        return nltk.ConfusionMatrix(gs, predictions)

    # Computes Confusion Matrix and returns, Precision, Recall, F-Measure, Accuaracy, and Confusion Matrix
    def classifierPRF(self, posTokenizedPostsTest, negTokenizedPostsTest, posLIWCPosts, negLIWCPosts):
        cm = self.classifierConfusionMatrix(posTokenizedPostsTest, negTokenizedPostsTest, posLIWCPosts, negLIWCPosts)
        TP = cm['pos', 'pos']
        TN = cm['neg', 'neg']
        FP = cm['neg', 'pos']
        FN = cm['pos', 'neg']
        if (TP + FP) == 0:
            precision = None
        else:
            precision = float(TP) / float(TP + FP)
        if(TP + FN) == 0:
            recall = None
        else:
            recall = float(TP) / float(TP + FN)
        if (precision == None or recall == None or (precision + recall) == 0):
            FMeasure = None
        else:
            FMeasure = (2.0 * precision * recall) / (precision + recall)
        Accuracy = float(TP + TN) / float(TP + TN + FP + FN)
        return (precision, recall, FMeasure, Accuracy, cm)

    def removeClassesFromFeatures(self, featureSet):
        feature_set = []
        for feat, val in featureSet:
            feature_set.append(feat)
        return feature_set

    def calcD(self, p, q, vocab, lenP, lenQ):
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

    def calcAverageD(self, p, q, v, lenP, lenQ):
        av1 = self.calcD(p, q, v, lenP, lenQ)
        av2 = self.calcD(q, p, v, lenQ, lenP)
        return ((av1 + av2) / 2.0)

    ###
    # Collocation Feature Exploration
    ###
    def trainBigramCollocations(self, negPost, posPost):
        print "Constructing collocations"
        allBigrams = {}

        for uid in negPost:
            allBigrams[uid] = BigramCollocationFinder.from_words(negPost[uid]).ngram_fd.items()

        for uid in posPost:
            allBigrams[uid] = BigramCollocationFinder.from_words(posPost[uid]).ngram_fd.items()

        self.bigramVocab = self.getVocab(allBigrams)
        #self.bigrams = allBigrams

    def getBigramCollocation(self, post):
        return BigramCollocationFinder.from_words(post).ngram_fd.items()

    def getVocab(self, ngrams):
        vocab = []
        for ngram in ngrams.values():
            for (word, value) in ngram:
                if word not in vocab:
                    vocab += [word]
        return vocab

    def collocationsFeature(self, tokenizedPosPost, tokenizedNegPost, vocab = None):
        """
        Generates all bigram collocations and computes their KL-divergence
        """
        allBigrams = {}
        allLengths = {}

        # Count n-grams
        gram = nltk.FreqDist(tokenizedPosPost).items()

        allLengths[1] = math.fsum(v for (g, v) in gram)
        allBigrams[1] = gram

        gram = nltk.FreqDist(tokenizedNegPost).items()

        allLengths[2] = math.fsum(v for (g, v) in gram)
        allBigrams[2] = gram

        # Build`token vocabulary
        if (vocab == None):
            vocab = self.getVocab(allBigrams)
        return self.calcAverageD(allBigrams[1], allBigrams[2], vocab, allLengths[1], allLengths[2])


    def collocations(self, ngram, tokenizedPosPosts, tokenizedNegPosts, vocab = None, compareType = "AllvAll"):
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
            vocab = self.getVocab(allBigrams)

        print "Calculating D values."
        Mt = {}

        if (compareType == "AllvAll"):
            for (class1, uid1) in allBigrams:
                for (class2, uid2) in allBigrams:
                    if (uid1, uid2) in Mt:
                        ansD = Mt[(uid1,uid2)]
                    elif (uid2, uid1) in Mt:
                        ansD = Mt[(uid2, uid1)]
                    else:
                        Mt[(uid1, uid2)] = self.calcAverageD(allBigrams[(class1,uid1)], allBigrams[(class2,uid2)], vocab, allLengths[uid1], allLengths[uid2])
                        ansD = Mt[(uid1,uid2)]
                    print "D (" + str(uid1) + " || " + str(uid2) + " ) = " + str(ansD)
        elif (compareType == "1vAll"):
            theoneClass = 'pos'
            theoneID = 1
            for (class1, uid1) in allBigrams:
                if uid1 == 1:
                    theoneClass = class1
                    theoneID = uid1
                    break
            for (class2, uid2) in allBigrams:
                Mt[(theoneID, uid2)] = self.calcAverageD(allBigrams[(theoneClass, theoneID)], allBigrams[(class2,uid2)],
                                                         vocab, allLengths[theoneID], allLengths[uid2])
        else: # "SameID"
            for (class1, uid1) in allBigrams:
                for (class2, uid2) in allBigrams:
                    if (uid1 == uid2 and class1 != class2):
                        Mt[(uid1, uid2)] = self.calcAverageD(allBigrams[(class1, uid1)],
                                                                           allBigrams[(class2, uid2)], vocab,
                                                                           allLengths[uid1], allLengths[uid2])
        return Mt

    def saveCollocations(self, filename, tableM):
        with open(filename, "w") as f:
            for (uid1,uid2) in tableM:
                line = str(uid1) + '\t' + str(uid2) + '\t' + str(tableM[(uid1, uid2)]) + '\n'
                f.write(line)

    def firstPersonScore(self, tokenizedSentence):
        fpron = 0
        allpron = 0
        for word in tokenizedSentence:
            if word in firstPersonPronouns:
                fpron += 1
                allpron += 1
            elif word in allProunouns:
                allpron += 1
        if allpron == 0:
            return 0
        return (float(fpron) / float(allpron))

    def firstPersonSocreForAllPosts(self, posPost, negPost):
        scoreRes = {}
        for uid in posPost:
            scoreRes[uid] = self.firstPersonScore(posPost[uid])
        for uid in negPost:
            scoreRes[uid] = self.firstPersonScore(negPost[uid])
        return scoreRes

    def savefirstPersonScores(self, fScores, filename):
        with open(filename, "w") as f:
            for uid in fScores:
                if uid > 0:
                    line = str(uid) + '\t' + str(fScores[uid]) + '\n'
                    f.write(line)
            for uid in fScores:
                if uid < 0:
                    line = str(uid) + '\t' + str(fScores[uid]) + '\n'
                    f.write(line)

class LDAModeling:
    def __init__(self, numTopics, trainingMode = "Pos"):
        self.nTopic = numTopics
        self.trainMode = trainingMode

    def trainLDA(self, negPosts, posPosts, numPasses = 20):
        print "LDA Modeling started"
        negTexts = []
        posTexts = []
        for uid in negPosts:
            negTexts.append(negPosts[uid])
        for uid in posPosts:
            if (self.trainMode ==  "Cmb"):
                negTexts.append(posPosts[uid])
            else:
                posTexts.append(posPosts[uid])

        if (self.trainMode == "Sep"):
            self.negCorpus = [self.dictionary.doc2bow(text) for text in negTexts]
            self.posCorpus = [self.dictionary.doc2bow(text) for text in posTexts]

            self.negLdaModel = gensim.models.ldamodel.LdaModel(self.negCorpus, num_topics=self.nTopic, id2word=self.dictionary, passes=numPasses)
            self.posLdaModel = gensim.models.ldamodel.LdaModel(self.posCorpus, num_topics=self.nTopic, id2word=self.dictionary, passes=numPasses)
        elif (self.trainMode == "Pos"):
            self.posCorpus = [self.dictionary.doc2bow(text) for text in posTexts]
            self.posLdaModel = gensim.models.ldamodel.LdaModel(self.posCorpus, num_topics=self.nTopic,
                                                               id2word=self.dictionary, passes=numPasses)
        else:
            self.negCorpus = [self.dictionary.doc2bow(text) for text in negTexts]
            self.negLdaModel = gensim.models.ldamodel.LdaModel(self.negCorpus, num_topics=self.nTopic,
                                                               id2word=self.dictionary, passes=numPasses)


    def makeDictionary(self, trainPosPost, trainNegPost, devPosPost, devNegPost):
        texts = []
        for uid in trainNegPost:
            texts.append(trainNegPost[uid])
        for uid in trainPosPost:
            texts.append(trainPosPost[uid])
        for uid in devNegPost:
            texts.append(devNegPost[uid])
        for uid in devPosPost:
            texts.append(devPosPost[uid])
        self.dictionary = corpora.Dictionary(texts)

    def getLDAResults(self, numWords):
        if (self.trainMode == "Sep"):
            negRes = self.negLdaModel.print_topics(num_topics=self.nTopic, num_words=numWords)
            posRes = self.posLdaModel.print_topics(num_topics=self.nTopic, num_words=numWords)
            return (negRes, posRes)
        elif (self.trainMode == "Pos"):
            posRes = self.posLdaModel.print_topics(num_topics=self.nTopic, num_words=numWords)
            return (None, posRes)
        else:
            negRes = self.negLdaModel.print_topics(num_topics=self.nTopic, num_words=numWords)
            return (negRes, None)

    def getTopic(self, post):
        postbow = self.dictionary.doc2bow(post)
        if (self.trainMode == "Pos"):
            res = self.posLdaModel.get_document_topics(postbow)
            res.sort(key=itemgetter(1), reverse=True)
            return (None, res)
        elif (self.trainMode == "Sep"):
            resN = self.negLdaModel.get_document_topics(postbow)
            resN.sort(key=itemgetter(1), reverse=True)
            resP = self.posLdaModel.get_document_topics(postbow)
            resP.sort(key=itemgetter(1), reverse=True)
            return (resN, resP)
        else:
            res = self.negLdaModel.get_document_topics(postbow)
            res.sort(key=itemgetter(1), reverse=True)
            return (res, None)

    def getNumTopics(self):
        return self.nTopic


#
#   MPQA Subjectivity and polarity
#
class MPQALoader:
    def __init__(self):
        self.MPQADict = {}
        self.loadMPQA()

    def loadMPQA(self):
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
                self.MPQADict[mpqa_word] = (mpqa_type, plr)

    def getMPQAofWord(self, word):
        if (word in self.MPQADict):
            return self.MPQADict[word]
        else:
            for mpqa in self.MPQADict:
                if (mpqa[-1] != '*' or len(mpqa) - 1 > len(word)):
                    continue
                mpqaWord = mpqa[:-1]
                match = True
                for i in range(0,len(mpqaWord)):
                    if (mpqaWord[i] != word[i]):
                        match = False
                        break
                if (match):
                    return self.MPQADict[mpqa]
        return None

    def getSentenceMPQA(self, sent, type = 'P'):
        totalCount = 0
        res = 0
        for word in sent:
            tp = self.getMPQAofWord(word)
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

    def getPostsMPQA(self, posts, type = 'P', filename = None):
        mpqaTable = {}
        for uid in posts:
            mpqaTable[uid] = self.getSentenceMPQA(posts[uid], type)

        if (filename != None):
            with open(filename, "w") as f:
                for uid in mpqaTable:
                    line = str(uid) + '\t' + str(mpqaTable[uid]) + '\n'
                    f.write(line)

        return mpqaTable

if __name__ == "__main__":
    #random.seed(773)

    data = DataLoader()
    data.getRandomSample(1000, fileType="Word", sredditFilter=True)
    liwcLoader = LIWCProcessor()
    postHelperFuncs = PostProcessingHelpers()
    ldamodels = LDAModeling(20, "Cmb")

    posPostProcess = PostProcessing(data.getPositivePosts())
    negPostProcess = PostProcessing(data.getControlsPosts())
    posPostProcess.concatPostsByUser()
    negPostProcess.concatPostsByUser()

    tpPostsTrain = postHelperFuncs.tokenizePosts(posPostProcess.getConcatPostBodies(), filterStopWords=False)
    tnPostsTrain = postHelperFuncs.tokenizePosts(negPostProcess.getConcatPostBodies(), filterStopWords=False)

    supervised_classifier = SupervisedClassifier(liwcLoader, vocab=liwcLoader.LIWC_Classes, lda=ldamodels)
    fscores = supervised_classifier.firstPersonSocreForAllPosts(tpPostsTrain, tnPostsTrain)
    supervised_classifier.savefirstPersonScores(fscores, "results/fperson.txt")

    """
    data.readSimilarPositives(fileType="LIWC")
    knownpositive = PostProcessing(data.getLIWCPositivePosts())
    knownpositive.concatLIWCPosts()
    knownPositivePost = postHelperFuncs.unwrapLIWCPost(knownpositive.getConcatPostBodies())

    data.clearPosts()
    data.getRandomSample(1000, fileType="LIWC", sredditFilter=True)

    posLIWCPostProcess = PostProcessing(data.getLIWCPositivePosts())
    posLIWCPostProcess.concatLIWCPostsByUser()

    negLIWCPostProcess = PostProcessing(data.getLIWCControlsPosts())
    negLIWCPostProcess.concatLIWCPostsByUser()

    pLIWCPosts = postHelperFuncs.unwrapLIWCPost(posLIWCPostProcess.getConcatPostBodies())
    nLIWCPosts = postHelperFuncs.unwrapLIWCPost(negLIWCPostProcess.getConcatPostBodies())

    data.clearPosts()
    data.getRandomSample(500, setType='Test', fileType="LIWC")

    posDevLIWCPostProcess = PostProcessing(data.getLIWCPositivePosts())
    posDevLIWCPostProcess.concatLIWCPostsByUser()
    negDevLIWCPostProcess = PostProcessing(data.getLIWCControlsPosts())
    negDevLIWCPostProcess.concatLIWCPostsByUser()

    pDevLIWCPosts = postHelperFuncs.unwrapLIWCPost(posDevLIWCPostProcess.getConcatPostBodies())
    nDevLIWCPosts = postHelperFuncs.unwrapLIWCPost(negDevLIWCPostProcess.getConcatPostBodies())

    """

    #beforeCorpus, afterCorpus = posLIWCPostProcess.separateLIWCPostCorpusWrtSW()

    #beforeLIWCPosts = postHelperFuncs.unwrapLIWCPost(beforeCorpus)
    #afterLIWCPosts = postHelperFuncs.unwrapLIWCPost(afterCorpus)

    #res = supervised_classifier.collocations(1, knownPositivePost, pLIWCPosts, compareType="1vAll")
    #res = supervised_classifier.collocations(1, beforeCorpus, afterCorpus, compareType="SameID")
    #supervised_classifier.saveCollocations("results/simneg.txt", res)
    """
    supervised_classifier = SupervisedClassifier(liwcLoader, vocab=liwcLoader.LIWC_Classes, lda=ldamodels)
    supervised_classifier.addTruPositive(knownPositivePost[1])

    supervised_classifier.trainClassifier(None, None, pLIWCPosts, nLIWCPosts, classifierType="DTree")

    pr, rc, fm, ac, cm = supervised_classifier.classifierPRF(None, None, pDevLIWCPosts, nDevLIWCPosts)
    print (cm.pretty_format(sort_by_count=True, show_percents=True))
    print "Accuracy = " + str(ac) + ", Precision = " + str(pr) + ", Recall = " + str(rc) + ", F-Measure = " + str(fm)

    print "Classifier SVM:"
    supervised_classifier_SVM = SupervisedClassifier(liwcLoader, vocab=liwcLoader.LIWC_Classes, lda=ldamodels)
    supervised_classifier_SVM.addTruPositive(knownPositivePost[1])

    supervised_classifier_SVM.trainClassifier(None, None, pLIWCPosts, nLIWCPosts, "SVM")

    pr, rc, fm, ac, cm = supervised_classifier_SVM.classifierPRF(None, None, pDevLIWCPosts, nDevLIWCPosts)
    print (cm.pretty_format(sort_by_count=True, show_percents=True))
    print "Accuracy = " + str(ac) + ", Precision = " + str(pr) + ", Recall = " + str(rc) + ", F-Measure = " + str(fm)

    print "Classifier NB:"
    supervised_classifier_NB = SupervisedClassifier(liwcLoader, vocab=liwcLoader.LIWC_Classes, lda=ldamodels)
    supervised_classifier_NB.addTruPositive(knownPositivePost[1])
    supervised_classifier_NB.trainClassifier(None, None, pLIWCPosts, nLIWCPosts, "NB")

    pr, rc, fm, ac, cm = supervised_classifier_NB.classifierPRF(None, None, pDevLIWCPosts, nDevLIWCPosts)
    print (cm.pretty_format(sort_by_count=True, show_percents=True))
    print "Accuracy = " + str(ac) + ", Precision = " + str(pr) + ", Recall = " + str(rc) + ", F-Measure = " + str(fm)

    print "Classifier Maxent:"
    supervised_classifier_ME = SupervisedClassifier(liwcLoader, vocab=liwcLoader.LIWC_Classes, lda=ldamodels)
    supervised_classifier_ME.addTruPositive(knownPositivePost[1])
    supervised_classifier_ME.trainClassifier(None, None, pLIWCPosts, nLIWCPosts, "Maxent")

    pr, rc, fm, ac, cm = supervised_classifier_ME.classifierPRF(None, None, pDevLIWCPosts, nDevLIWCPosts)
    print (cm.pretty_format(sort_by_count=True, show_percents=True))
    print "Accuracy = " + str(ac) + ", Precision = " + str(pr) + ", Recall = " + str(rc) + ", F-Measure = " + str(fm)
    """

    # -------------------------------------------------------------------------------------------------------------
    # ------------------------------------ Word Class -------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    """

    posLIWCPostProcess = PostProcessing(data.getLIWCPositivePosts())
    posLIWCPostProcess.concatLIWCPostsByUser()
    posLIWCPostProcess.concatLIWCPostTitlesByUser()

    negLIWCPostProcess = PostProcessing(data.getLIWCControlsPosts())
    negLIWCPostProcess.concatLIWCPostsByUser()


    pLIWCPosts = postHelperFuncs.unwrapLIWCPost(posLIWCPostProcess.getConcatPostBodies())
    nLIWCPosts = postHelperFuncs.unwrapLIWCPost(negLIWCPostProcess.getConcatPostBodies())

    data.clearPosts()
    data.getRandomSample(500, setType='Test', liwcConverted=True)
    posDevLIWCPostProcess = PostProcessing(data.getLIWCPositivePosts())
    posDevLIWCPostProcess.concatLIWCPostsByUser()

    negDevLIWCPostProcess = PostProcessing(data.getLIWCControlsPosts())
    negDevLIWCPostProcess.concatLIWCPostsByUser()

    pDevLIWCPosts = postHelperFuncs.unwrapLIWCPost(posDevLIWCPostProcess.getConcatPostBodies())
    nDevLIWCPosts = postHelperFuncs.unwrapLIWCPost(negDevLIWCPostProcess.getConcatPostBodies())

    vocabulary = liwcLoader.getLIWCVocab()

    supervised_classifier = SupervisedClassifier(liwcLoader, vocabulary)
    supervised_classifier.trainClassifier(pLIWCPosts, nLIWCPosts, "Maxent")

    pr, rc, fm, ac, cm = supervised_classifier.classifierPRF(pDevLIWCPosts, nDevLIWCPosts)
    print (cm.pretty_format(sort_by_count=True, show_percents=True))
    print "Accuracy = " + str(ac) + ", Precision = " + str(pr) + ", Recall = " + str(rc) + ", F-Measure = " + str(fm)
    """

    # -------------------------------------------------------------------------------------------------------------
    # ------------------------------------ Unigram ---------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------
    """
    posPostProcess = PostProcessing(data.getPositivePosts())
    negPostProcess = PostProcessing(data.getControlsPosts())
    posPostProcess.concatPostsByUser()
    negPostProcess.concatPostsByUser()

    posLIWCPostProcess = PostProcessing(data.getLIWCPositivePosts())
    posLIWCPostProcess.concatLIWCPostsByUser()
    negLIWCPostProcess = PostProcessing(data.getLIWCControlsPosts())
    negLIWCPostProcess.concatLIWCPostsByUser()

    pLIWCPosts = postHelperFuncs.unwrapLIWCPost(posLIWCPostProcess.getConcatPostBodies())
    nLIWCPosts = postHelperFuncs.unwrapLIWCPost(negLIWCPostProcess.getConcatPostBodies())

    data.clearPosts()
    data.getRandomSample(50, setType='Test', fileType="LIWC")

    posDevLIWCPostProcess = PostProcessing(data.getLIWCPositivePosts())
    posDevLIWCPostProcess.concatLIWCPostsByUser()
    negDevLIWCPostProcess = PostProcessing(data.getLIWCControlsPosts())
    negDevLIWCPostProcess.concatLIWCPostsByUser()

    pDevLIWCPosts = postHelperFuncs.unwrapLIWCPost(posDevLIWCPostProcess.getConcatPostBodies())
    nDevLIWCPosts = postHelperFuncs.unwrapLIWCPost(negDevLIWCPostProcess.getConcatPostBodies())

    posPostProcessDev = PostProcessing(data.getPositivePosts())
    negPostProcessDev = PostProcessing(data.getControlsPosts())
    posPostProcessDev.concatPostsByUser()
    negPostProcessDev.concatPostsByUser()

    tpPostsTrain = postHelperFuncs.stemmPosts(postHelperFuncs.tokenizePosts(posPostProcess.getConcatPostBodies()))
    tnPostsTrain = postHelperFuncs.stemmPosts(postHelperFuncs.tokenizePosts(negPostProcess.getConcatPostBodies()))

    tpPostsDev = postHelperFuncs.stemmPosts(postHelperFuncs.tokenizePosts(posPostProcessDev.getConcatPostBodies()))
    tnPostsDev = postHelperFuncs.stemmPosts(postHelperFuncs.tokenizePosts(negPostProcessDev.getConcatPostBodies()))

    """

    #ldamodels.makeDictionary(tpPostsTrain, tnPostsTrain, tpPostsDev, tnPostsDev)
    #ldamodels.trainLDA(tnPostsTrain, tpPostsTrain)
    #neglda, poslda = ldamodels.getLDAResults(20)
    #print "Negative posts LDA:"
    #for t, r in neglda:
    #    print "Topic " + str(t) + " = " + r
    #print "Positive posts LDA:"
    #for t, r in poslda:
    #    print "Topic " + str(t) + " = " + r

    #print "Predictions:"
    #print "Pos Posts"
    #for uid in tpPostsDev:
    #    if (tpPostsDev[uid] == []):
    #        continue
    #    print ldamodels.getTopic(tpPostsDev[uid])
    #print "Neg Posts"
    #for uid in tnPostsDev:
    #    if (tnPostsDev[uid] == []):
    #        continue
    #    print ldamodels.getTopic(tnPostsDev[uid])


    """

    vocabulary = postHelperFuncs.getVocabularyFromPosts(tpPostsTrain, tnPostsTrain, 1000)

    print "Classifier DTree:"

    supervised_classifier = SupervisedClassifier(liwcLoader, vocabulary, ldamodels)
    #supervised_classifier.trainBigramCollocations(tnPostsTrain, tpPostsTrain)
    supervised_classifier.trainClassifier(tpPostsTrain, tnPostsTrain, pLIWCPosts, nLIWCPosts, classifierType="DTree")

    pr, rc, fm, ac, cm = supervised_classifier.classifierPRF(tpPostsDev, tnPostsDev, pDevLIWCPosts, nDevLIWCPosts)
    print (cm.pretty_format(sort_by_count=True, show_percents=True))
    print "Accuracy = " + str(ac) + ", Precision = " + str(pr) + ", Recall = " + str(rc) + ", F-Measure = " + str(fm)

    print "Classifier SVM:"
    #supervised_classifier_SVM = SupervisedClassifier(liwcLoader, vocabulary, ldamodels)
    supervised_classifier.trainClassifier(tpPostsTrain, tnPostsTrain, pLIWCPosts, nLIWCPosts, "SVM")

    pr, rc, fm, ac, cm = supervised_classifier.classifierPRF(tpPostsDev, tnPostsDev, pDevLIWCPosts, nDevLIWCPosts)
    print (cm.pretty_format(sort_by_count=True, show_percents=True))
    print "Accuracy = " + str(ac) + ", Precision = " + str(pr) + ", Recall = " + str(rc) + ", F-Measure = " + str(fm)

    print "Classifier NB:"
    #supervised_classifier_NB = SupervisedClassifier(liwcLoader, vocabulary, ldamodels)
    supervised_classifier.trainClassifier(tpPostsTrain, tnPostsTrain, pLIWCPosts, nLIWCPosts, "NB")

    pr, rc, fm, ac, cm = supervised_classifier.classifierPRF(tpPostsDev, tnPostsDev, pDevLIWCPosts, nDevLIWCPosts)
    print (cm.pretty_format(sort_by_count=True, show_percents=True))
    print "Accuracy = " + str(ac) + ", Precision = " + str(pr) + ", Recall = " + str(rc) + ", F-Measure = " + str(fm)

    print "Classifier Maxent:"
    #supervised_classifier_ME = SupervisedClassifier(liwcLoader, vocabulary, ldamodels)
    supervised_classifier.trainClassifier(tpPostsTrain, tnPostsTrain, pLIWCPosts, nLIWCPosts, "Maxent")

    pr, rc, fm, ac, cm = supervised_classifier.classifierPRF(tpPostsDev, tnPostsDev, pDevLIWCPosts, nDevLIWCPosts)
    print (cm.pretty_format(sort_by_count=True, show_percents=True))
    print "Accuracy = " + str(ac) + ", Precision = " + str(pr) + ", Recall = " + str(rc) + ", F-Measure = " + str(fm)

    """

    # -------------------------------------------------------------------------------------------------------------
    # ------------------------------------ Junk -------------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    #print 'Positive Samples: %s' % posSamples
    #print 'Control Samples: %s' % negSamples

    #print "Reading posts."

    #posPosts,negPosts = loadPosts(posSamples, negSamples)

    #tpPosts = tokenizePosts(concatPostsByUser(posPosts))
    #tnPosts = tokenizePosts(concatPostsByUser(negPosts))

    #topicPosPosts = tokenizePosts(concatPostTitlesByUser(posPosts))
    #topicNegPosts = tokenizePosts(concatPostTitlesByUser(negPosts))

    #print posPosts
    #tpPosts = tokenizePosts(samplePostsFromUser(posPosts, 5))
    #tnPosts = tokenizePosts(samplePostsFromUser(negPosts, 5))

    #print "Computing MPQA scores."
    #getPostsMPQA(topicPosPosts, MPQADict, 'P', "results/PTmpqaPos.txt")
    #getPostsMPQA(topicNegPosts, MPQADict, 'P', "results/PTmpqaNeg.txt")
    #getPostsMPQA(tpPosts, MPQADict, 'P', "results/PBmpqaPos.txt")
    #getPostsMPQA(tnPosts, MPQADict, 'P', "results/PBmpqaNeg.txt")
    #getPostsMPQA(topicPosPosts, MPQADict, 'S', "results/STmpqaPos.txt")
    #getPostsMPQA(topicNegPosts, MPQADict, 'S', "results/STmpqaNeg.txt")
    #getPostsMPQA(tpPosts, MPQADict, 'S', "results/SBmpqaPos.txt")
    #getPostsMPQA(tnPosts, MPQADict, 'S', "results/SBmpqaNeg.txt")

    #print "Computing LIWC representation."
    #liwcPPosts = postsToLIWC(tpPosts)
    #liwcNPosts = postsToLIWC(tnPosts)
    #calculateEmotionScoreForAllPosts(tpPosts, "results/posScore.txt")
    #calculateEmotionScoreForAllPosts(tnPosts, "results/negScore.txt")

    #saveCollocations("results/dlconv.txt", collocations(1, liwcPPosts, liwcNPosts, liwcVocab))
    #saveCollocations("results/dlconv.txt", collocations(1, tpPosts, tnPosts))
