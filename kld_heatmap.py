import nltk
from nltk.corpus import stopwords
import string
import collections
import math
import operator
from nltk.collocations import *
from nltk.stem import WordNetLemmatizer
import re
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

def readfile(filename):
    tableData = {}
    with open(filename, "r") as f:
        for line in f:
            segs = line.replace('\n', '').split('\t')
            uid1 = int(segs[0])
            uid2 = int(segs[1])
            klVal = float(segs[2])
            tableData[(uid1, uid2)] = klVal
            tableData[(uid2, uid1)] = klVal
    return collections.OrderedDict(sorted(tableData.items()))

if __name__ == "__main__":
    tdata = readfile("results/dlconv.txt")
    plotly.tools.set_credentials_file(username='pmotakef', api_key='FnOhmwb5Gx66G9PnDvk8')
    tb = []
    tx = []
    ty = []

    sampleNum = int(math.sqrt(float(len(tdata))))
    print len(tdata)

    for i in range(0,sampleNum):
        tb2 = []
        for j in range(0,sampleNum):
            tb2 += [0]
        tb += [tb2]
        tx += [" "]
        ty += [" "]

    xcounter = 0
    ycounter = 0
    print tdata
    for uids in tdata.iteritems():
        uidss, val = uids
        uid1, uid2 = uidss
        x = xcounter
        y = ycounter

        tb[x][y] = val
        if (ycounter < 1):
            tx[x] = str(uid2) + 'id'


        print uid1
        xcounter += 1
        if (xcounter >= sampleNum):
            xcounter = 0
            ycounter += 1
            ty[y] = str(uid1) + 'id'

    data = [
        go.Heatmap(
            z=tb,
            x=tx,
            y=ty
        )
    ]
    py.iplot(data, filename="klheatmap")
