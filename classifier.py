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

    def getRandomSample(self, number, setType = "Train", sredditFilter=True):
        posSamples = random.sample(self.positivesIDs[setType], number)
        negSamples = random.sample(self.controlsIDs[setType], number)
        self.loadPosts(posSamples, negSamples, sredditFilter)

    def readAllSamples(self, setType = "Train", sredditFilter=True):
        self.loadPosts(self.positivesIDs[setType], self.controlsIDs[setType], sredditFilter)

    def readPosts(self, filename, users=None, sredditFilter=True):
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
                    if (not sredditFilter or ps.subReddit not in filterSubreddit):
                        posts.append(ps)
        return posts

    def loadPosts(self, posUsers, negUsers, subRedditFilter = True):
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
        postFilenames = []

        for nuid in negUsers:
            tmp = self.getPostFilename(nuid)
            if tmp not in postFilenames:
                postFilenames += [tmp]
        for filename in postFilenames:
            negFilename = join(controlsPath, filename + '.posts')
            self.negPosts += self.readPosts(negFilename, negUsers, subRedditFilter)

        # Write to pickle cache
        with open(pickle_filename, 'wb') as f:
            pickle.dump((self.posPosts, self.negPosts), f, protocol=pickle.HIGHEST_PROTOCOL)

    def getPositivePosts(self):
        return self.posPosts

    def getControlsPosts(self):
        return self.negPosts

    def readLIWCConvertedPosts(self, filename, users=None, sredditFilter=True):
        posts = []
        with open(filename, 'r') as f:
            for line in f:
                postTitle = []
                postBody = []
                # Split on only the first 5 tabs (sometimes post bodies have tabs)
                segs = line.strip().split('\t', 5)
                # Add empty post body, if necessary (image posts, f.e.)
                if len(segs) == 5: segs.append('')
                pTitle = segs[4].split(' ')
                for pt in pTitle:
                    liwcTitle = pt.split(',')
                    postTitle += [(int(liwcTitle[0]), int(liwcTitle[1]))]
                pBody = segs[5].split(' ')
                for pb in pBody:
                    liwcBody = pb.split(',')
                    postBody += [(int(liwcBody[0]), int(liwcBody[1]))]
                ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), segs[3], postTitle, postBody)
                if users is None or ps.userID in users:
                    if (sredditFilter == True and ps.subReddit not in filterSubreddit):
                        posts.append(ps)
        return posts

###
# Data pre-processing
###
class PostProcessing:
    def __init__(self, posts):
        self.rawPosts = posts

    def concatPosts(self, rawPosts):
        """Concatenates all of the post bodies"""
        self.concatinatedPosts = ' '.join((p.postBody for p in self.rawPosts))

    def concatPostsByUser(self):
        """Concatenates all of the post bodies by user id"""
        conRes = defaultdict(str)
        for p in self.rawPosts:
            conRes[p.userID] += p.postBody + ' '
        self.concatinatedPosts = conRes

    def concatPostTitlesByUser(self):
        """Concatenates all of the post titles by user id"""
        conRes = defaultdict(str)
        for p in self.rawPosts:
            conRes[p.userID] += p.postTitle + ' '
        self.concatinatedTitles = conRes

    def getConcatPostBodies(self):
        return self.concatinatedPosts

    def getConcatPostTitles(self):
        return self.concatinatedTitles


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
    def scoreEmotion(self, liwcConvertedPost):
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
    def __init__(self, liwc, vocab):
        self.liwcProcesses = liwc
        self.vocab = vocab

    # Feature fromat:
    # ({'Feature name' : Feature value}, class)
    def unigramFeatureSet(self, tokenizedSentence):
        ngram_features = {}
        for word in self.vocab:
            ngram_features[word] = (word in tokenizedSentence)
        return ngram_features

    def getFeatureSetForAllPosts(self,posTokenizedPosts, negTokenizedPosts):
        feature_set = []
        for uid in posTokenizedPosts:
            feature_set += [(self.unigramFeatureSet(posTokenizedPosts[uid]), 'pos')]
        for uid in negTokenizedPosts:
            feature_set += [(self.unigramFeatureSet(negTokenizedPosts[uid]), 'neg')]
        random.shuffle(feature_set)
        return feature_set

    def getFeatureSetForAPost(self,tokenizedPost, classification):
        feature_set = [(self.unigramFeatureSet(tokenizedPost), classification)]
        return feature_set

    def trainClassifier(self, posTokenizedPosts, negTokenizedPosts):
        print "Training!"
        feature_set = self.getFeatureSetForAllPosts(posTokenizedPosts, negTokenizedPosts)
        self.classifier = nltk.NaiveBayesClassifier.train(feature_set)

    def classifierAccuracy(self, posTokenizedPostsTest, negTokenizedPostsTest):
        feature_set = self.getFeatureSetForAllPosts(posTokenizedPostsTest, negTokenizedPostsTest)
        return nltk.classify.accuracy(self.classifier, feature_set)

    def classifierConfusionMatrix(self, posTokenizedPostsTest, negTokenizedPostsTest):
        gold = self.getFeatureSetForAllPosts(posTokenizedPostsTest, negTokenizedPostsTest)
        feature_set = self.removeClassesFromFeatures(gold)
        predictions = self.classifier.classify_many(feature_set)
        gs = []
        for i in range(0, len(gold)):
            g, v = gold[i]
            gs += [v]
        return nltk.ConfusionMatrix(gs, predictions)

    def classifierPRF(self, posTokenizedPostsTest, negTokenizedPostsTest):
        cm = self.classifierConfusionMatrix(posTokenizedPostsTest, negTokenizedPostsTest)
        TP = cm['pos', 'pos']
        TN = cm['neg', 'neg']
        FP = cm['neg', 'pos']
        FN = cm['pos', 'neg']
        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        FMeasure = (2.0 * precision * recall) / (precision + recall)
        return (precision, recall, FMeasure)

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
    def getVocab(self, ngrams):
        vocab = []
        for ngram in ngrams.values():
            for (word, value) in ngram:
                if word not in vocab:
                    vocab += [word]
        return vocab

    def collocations(self, ngram, tokenizedPosPosts, tokenizedNegPosts, vocab = None):
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
        for (class1, uid1) in allBigrams:
            for (class2, uid2) in allBigrams:
                if (uid1, uid2) in Mt:
                    ansD = Mt[(uid1,uid2)]
                elif (uid2, uid1) in Mt:
                    ansD = Mt[(uid2, uid1)]
                else:
                    Mt[(uid1, uid2)] = self.liwcProcesses.calcAverageD(allBigrams[(class1,uid1)], allBigrams[(class2,uid2)], vocab, allLengths[uid1], allLengths[uid2])
                    ansD = Mt[(uid1,uid2)]
                print "D (" + str(uid1) + " || " + str(uid2) + " ) = " + str(ansD)
        return Mt

    def saveCollocations(self, filename, tableM):
        with open(filename, "w") as f:
            for (uid1,uid2) in tableM:
                line = str(uid1) + '\t' + str(uid2) + '\t' + str(tableM[(uid1, uid2)]) + '\n'
                f.write(line)


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
    data.getRandomSample(100)
    liwcLoader = LIWCProcessor()
    postHelperFuncs = PostProcessingHelpers()

    posPostProcess = PostProcessing(data.getPositivePosts())
    negPostProcess = PostProcessing(data.getControlsPosts())
    posPostProcess.concatPostsByUser()
    negPostProcess.concatPostsByUser()

    data.clearPosts()
    data.getRandomSample(20, 'Dev')

    posPostProcessDev = PostProcessing(data.getPositivePosts())
    negPostProcessDev = PostProcessing(data.getControlsPosts())
    posPostProcessDev.concatPostsByUser()
    negPostProcessDev.concatPostsByUser()


    tpPostsTrain = postHelperFuncs.tokenizePosts(posPostProcess.getConcatPostBodies())
    tnPostsTrain = postHelperFuncs.tokenizePosts(negPostProcess.getConcatPostBodies())

    tpPostsDev = postHelperFuncs.tokenizePosts(posPostProcessDev.getConcatPostBodies())
    tnPostsDev = postHelperFuncs.tokenizePosts(negPostProcessDev.getConcatPostBodies())

    #vocabulary = postHelperFuncs.getVocabulary(tpPostsTrain)
    #vocabulary.union(postHelperFuncs.getVocabulary(tnPostsTrain))
    vocabulary = postHelperFuncs.getVocabularyFromPosts(tpPostsTrain, tnPostsTrain, 200)


    supervised_classifier = SupervisedClassifier(liwcLoader, vocabulary)
    supervised_classifier.trainClassifier(tpPostsTrain, tnPostsTrain)


    print supervised_classifier.classifierAccuracy(tpPostsDev, tnPostsDev)
    cm = supervised_classifier.classifierConfusionMatrix(tpPostsDev, tnPostsDev)
    print (cm.pretty_format(sort_by_count=True, show_percents=True))
    pr, rc, fm = supervised_classifier.classifierPRF(tpPostsDev, tnPostsDev)
    print "Precision = " + str(pr) + ", Recall = " + str(rc) + ", F-Measure = " + str(fm)


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
