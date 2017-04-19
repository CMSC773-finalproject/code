import nltk
from nltk.corpus import stopwords
import string
import collections
import math
import operator
from collections import namedtuple

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
#               Main
#

if __name__ == "__main__":

    stopws = stopwords.words('english')
    temp = readPost(positivesPath+"0.posts")
    for i in range(0,2):
        print temp[i].postBody
    loadLIWC()
    print LIWC_words['ace']
    print getLIWCclass('kissed')
    print translateAllLIWCClasses(getLIWCclass('kissing'))
    print translateAllLIWCClasses(getLIWCclass('a'))



