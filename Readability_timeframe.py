#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 22:28:39 2017

@author: Yash
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 21:35:35 2017

@author: Yash
"""
#import sys
#reload(sys)
#sys.setdefaultencoding("utf-8")
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
    from scipy import stats, integrate
    from itertools import chain
    import pandas as pd
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import cmudict
    from pandas import ExcelWriter
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    from sklearn.utils import shuffle
    from sklearn import linear_model 
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import StandardScaler 
    from sklearn.decomposition import PCA
    import time
    from textstat.textstat import textstat
    import seaborn as sns
    from sklearn import metrics
    import os
    from sklearn import tree
    
    not_punctuation = lambda w: not (len(w)==1 and (not w.isalpha()))
    get_word_count = lambda text: len(filter(not_punctuation, word_tokenize(text)))
    get_sent_count = lambda text: len(sent_tokenize(text))
    
    
    prondict = cmudict.dict()
    
    numsyllables_pronlist = lambda l: len(filter(lambda s:(s.encode('ascii', 'ignore').lower()[-1]).isdigit(), l))
    def numsyllables(word):
      try:
        return list(set(map(numsyllables_pronlist, prondict[word.lower()])))
      except KeyError:
        return [0]
        
    def text_statistics(text):
      word_count = get_word_count(text)
      sent_count = get_sent_count(text)
      syllable_count = sum(map(lambda w: max(numsyllables(w)), word_tokenize(text)))
      return word_count, sent_count, syllable_count
      

    PostsStruct = namedtuple("PostsStruct", "postID userID timeStamp subReddit postTitle postBody")
    
    stopws = stopwords.words('english')
    # TODO: does string.punctuation need to be converted to unicode?
    stopws.extend(string.punctuation)
    stopws.extend(u"'s 're n't 'm 've 'd '' 't --".split())
    
    filterSubreddit = ["Anger", "BPD", "EatingDisorders", "MMFB", "StopSelfHarm", "SuicideWatch", "addiction", "alcoholism",
                       "depression", "feelgood", "getting over it", "hardshipmates", "mentalhealth", "psychoticreddit",
                       "ptsd", "rapecounseling", "socialanxiety", "survivorsofabuse", "traumatoolbox"]
              
    #
    fk_formula = lambda word_count, sent_count, syllable_count : 0.39 * word_count / (sent_count+1) + 11.8 * syllable_count / (word_count+1) - 15.59
    def flesch_kincaid(text):
      word_count, sent_count, syllable_count = text_statistics(text)
      if (sent_count != 0.0) or (word_count != 0.0)or (syllable_count != 0.0):
    #      sent_count = 1
    #      word_count = 1
          return fk_formula(word_count, sent_count, syllable_count)
      else:
          return 0.0;
    def flesch_reading_ease(self,text):
         word_count, sent_count, syllable_count = text_statistics(text)
         if (sent_count != 0.0) or (word_count != 0.0)or (syllable_count != 0.0):
             return self.flesch_kincaid_grade(text)
         else: 
             return 0.0;
         
    def convert_time(number):
         return time.gmtime(number)[3]
         
    
    def readPosts(filename, users=None, sredditFilter=True):
        """
        Returns the list of PostStructs from a given file
        Optionally filters using a set of user ids
        """
        posts = []
        with open(filename,'r') as f:
            for line in f:
                # Split on only the first 5 tabs (sometimes post bodies have tabs) encoding = 'utf8' decode('utf-8')
                segs = line.decode('utf-8').strip().split('\t', 5)
                # Add empty post body, if necessary (image posts, f.e.)
                if len(segs) == 5: segs.append('')
                ps = PostsStruct(segs[0], int(segs[1]), int(segs[2]), segs[3], segs[4], segs[5])
                if users is None or ps.userID in users:
                    if (sredditFilter == True and ps.subReddit not in filterSubreddit):
                        posts.append(ps)
        return posts
    #only positives with the filter one (so only posts from positive set with no suicide posts)   
    
    #Bucketing
    def group_readability(score):
        if score >= 15:
            return 'Confusing'
        if score < 15 and score >= 12:
            return 'Difficult'
        elif score < 12 and score >= 9:
            return 'Fairly Difficult'
        elif score < 9 and score >=5:
            return 'Standard'
        elif score < 5 and score >= 3:
            return 'Fairly Easy'
        elif score < 3 and score >= 1:
            return 'Easy'
        else:
            return 'Outlier'
    
    def group_time(time):
        if time >= 20:
            return 'Night'
        if time < 20 and time >= 17:
            return 'Evening'
        elif time < 17 and time >= 12:
            return 'afternoon'
        elif time < 12 and time >=8:
            return 'Morning'
        elif time < 8 and time >= 5:
            return 'Early Morning'
        elif time < 5 and time >= 0:
            return 'Late Night'
        else:
            return 'Outlier'
    
#Post level    
    posts = readPosts('positives/suicidewatch_positives/merged-file.posts')
    data = pd.DataFrame.from_dict(posts)
    data['postBody'].replace('', np.nan, inplace=True)
    data['postTitle'].replace('', np.nan, inplace=True)
    data.dropna(subset=['postBody'], inplace=True)
    data.dropna(subset=['postTitle'], inplace=True)
    data['fleschkin_score'] = data['postBody'].apply(flesch_kincaid)
    sns.distplot(data['fleschkin_score'])
    #range shown in histogram
    filtered = data['fleschkin_score'][(data['fleschkin_score'] >= 0) & (data['fleschkin_score'] < 15)]
    sns.distplot(filtered, kde=True, hist=True, hist_kws={"range": [0,15]})
    plt.show()     

    data['fleschkin_score'].describe() #describes the data of the particular field
    data.corr()
    data['class']='Positive'


# User Level for positives
#/Users/Yash/Desktop/UMD Courses/Sem4/Linguistics2/Project/
positive_train_df = pd.read_csv('positives/TRAIN.txt')
positve_users_train = pd.DataFrame.from_dict(positive_train_df) 
positve_users_train.columns = ['userID']
positive_test_df = pd.read_csv('positives/TEST.txt')
positve_users_test = pd.DataFrame.from_dict(positive_test_df)
positive_dev_df = pd.read_csv('positives/DEV.txt')
positve_users_dev = pd.DataFrame.from_dict(positive_dev_df)

positve_users_train.columns = ['userID']

s1 = pd.merge(positve_users_train, data, how='inner', on=['userID']) #s1 has all positive users used to train the data
s2 = s1.groupby('userID', as_index=False)['fleschkin_score'].mean() #this result will have userId and 
#we get the distribution of these positive users
filtered_avg_score = s2['fleschkin_score'][(s2['fleschkin_score'] >= 0) & (s2['fleschkin_score'] < 15)]
sns.distplot(filtered_avg_score, kde=True, hist=True, hist_kws={"range": [0,15]})
plt.show() 



#we do the same with the control users
controls_train_df = pd.read_csv('controls/suicidewatch_controls/TRAIN.txt')
controls_users_train = pd.DataFrame.from_dict(controls_train_df) 
controls_test_df = pd.read_csv('controls/suicidewatch_controls/TEST.txt')
controls_users_test = pd.DataFrame.from_dict(controls_test_df)
controls_dev_df = pd.read_csv('controls/suicidewatch_controls/DEV.txt')
controls_users_dev = pd.DataFrame.from_dict(controls_dev_df)

controls_users_train.columns = ['userID']

s3 = pd.merge(controls_users_train, data1, how='inner', on=['userID']) #s1 has all positive users used to train the data
s4 = s3.groupby('userID', as_index=False)['fleschkin_score'].mean() #this result will have userId and 
#we get the distribution of these control users
filtered_avg_score_1 = s4['fleschkin_score'][(s4['fleschkin_score'] >= 0) & (s4['fleschkin_score'] < 15)]
sns.distplot(filtered_avg_score_1, kde=True, hist=True, hist_kws={"range": [0,15]})
plt.show()

stats.ttest_ind(s4['fleschkin_score'], s2['fleschkin_score'])

#User level over

s1 = pd.merge(positve_users_train, data, how='inner', on=['postID', 'userID']) #s1 has all positive users used to train the data
s1.groupby(['userID']).mean() #this result will have userId and 
#we get the distribution of these positive users
filtered_avg_score = s1['fleschkin_score'][(data['fleschkin_score'] >= 0) & (s1['fleschkin_score'] < 15)]
sns.distplot(filtered_avg_score, kde=True, hist=True, hist_kws={"range": [0,15]})
plt.show() 
#plot them


#conclude with the observation

   
    #all control filter on (just to avoid outliers)
    posts1 = readPosts('controls/suicidewatch_controls/merged-file_control.posts')
    data1 = pd.DataFrame.from_dict(posts1)
    data1['postBody'].replace('', np.nan, inplace=True)
    data1['postTitle'].replace('', np.nan, inplace=True)
    data1.dropna(subset=['postBody'], inplace=True)
    data1.dropna(subset=['postTitle'], inplace=True)
    data1['fleschkin_score'] = data1['postBody'].apply(flesch_kincaid)
    
    filtered1 = data1['fleschkin_score'][(data1['fleschkin_score'] >= 0) & (data1['fleschkin_score'] < 15)]
    sns.distplot(filtered1, kde=True, hist=True, hist_kws={"range": [0,15]})
    plt.show()
    
    sns.distplot(filtered, label='x')
    sns.distplot(filtered1,label='y')
    plt.legend(loc='upper right')
    plt.show()
    
    data1['fleschkin_score'].describe() #describes the data of the particular field
    data1.corr()
    data1['class']='control'
    
    data_con = data
    data_new = data_con.append(data1, ignore_index=True)
    data_new['time_hrs']= 0
    data_new['time_hrs'] = data_new['timeStamp'].apply(convert_time)
    data_new['time_hrs'].describe()
    data_new['fleschkin_score_bin']=data_new['fleschkin_score'].map(group_readability)
    data_new['time_hrs_bin']=data_new['time_hrs'].map(group_time)

    data_new_score = data_new.groupby(['userID']).mean().reset_index()
    data_new_score.drop('timeStamp', axis=1, inplace=True)
    data_new_score.drop('time_hrs', axis=1, inplace=True)
    data_temp_1 = data_new_score[(data_new_score['userID']>0)]
    data_temp_2 = data_new_score[(data_new_score['userID']<0)]
    data_temp_1['class']='positive'
    data_temp_2['class']='control'
    frames = [data_temp_1,data_temp_2]
    data_new_score = pd.concat(frames)
 
    # for significnce of Readability score
    dist1 = data_temp_1['fleschkin_score'][(data_temp_1['fleschkin_score'] >= 0) & (data_temp_1['fleschkin_score'] < 15)]
    sns.distplot(dist1, kde=True, hist=True, hist_kws={"range": [0,15]})
    plt.show()
    dist2 = pd.DataFrame.from_dict(dist1)
    dist2['fleschkin_score'].describe()
    
    dist3 = data_temp_2['fleschkin_score'][(data_temp_2['fleschkin_score'] >= 0) & (data_temp_2['fleschkin_score'] < 15)]
    sns.distplot(dist3, kde=True, hist=True, hist_kws={"range": [0,15]})
    plt.show()
    dist4 = pd.DataFrame.from_dict(dist3)
    dist4['fleschkin_score'].describe()
    
    sns.distplot(dist1, label='x')
    sns.distplot(dist3,label='y')
    plt.legend(loc='upper right')
    plt.show()

stats.ttest_ind(dist2['fleschkin_score'], dist4['fleschkin_score'])

sns.distplot(y1['num'], kde=True, hist=True, hist_kws={"range": [0,7]})
plt.show() 

data_new_score['fleschkin_score_bin']=data_new_score['fleschkin_score'].map(group_readability)
data_new_score.drop('fleschkin_score', axis=1, inplace=True)
pd.crosstab(data_new_score['fleschkin_score_bin'], data_new_score['class'], margins=True)
df_score=pd.get_dummies(data_new_score[['class','fleschkin_score_bin']], prefix='dummy', drop_first=True)
df_score.columns
predictor_names=df_score.drop('dummy_positive', axis=1).columns
predictor_names
predictors=df_score[predictor_names]
target= df_score['dummy_positive']


filtered_new_pos = data_new_score['fleschkin_score'][(data_new_score['fleschkin_score'] >= 0) & (data_new_score['fleschkin_score'] < 15) & (data_new_score['userID'] > 0)]
sns.distplot(filtered_new_pos, kde=True, hist=True, hist_kws={"range": [0,15]})
plt.show()

filtered_new_con = data_new_score['fleschkin_score'][(data_new_score['fleschkin_score'] >= 0) & (data_new_score['fleschkin_score'] < 15) & (data_new_score['userID'] < 0)]
sns.distplot(filtered_new_con, kde=True, hist=True, hist_kws={"range": [0,15]})
plt.show()

stats.ttest_ind(filtered_new_pos, filtered_new_con, equal_var = True)

s1 = pd.merge(positve_users_train, data, how='inner', on=['postID', 'userID'])

user_time_ferq = data_new.groupby(["userID", "time_hrs_bin"]).size()

x = data_new.groupby(['userID','time_hrs_bin']).size().to_frame('occurences').reset_index()
#writer = pd.ExcelWriter('positive_negetive_time_occurance.xlsx')
#x.to_excel(writer,'Sheet1')

y = x.groupby(['userID'], sort=False)['occurences'].max().reset_index()
#Join x and y
y1 = pd.merge(x, y, how='inner', on=['userID','occurences'])
y1 = y1.drop_duplicates('userID', keep = 'last')
y1['class'] = "control"
y1.loc[y1['userID'] > 0, 'class'] = 'positive'
y1.time_hrs_bin.unique()
#y1['c'].cat.codes

y1.loc[y1['userID'] > 0, 'class'] = 'positive'
y1['num'] = 0
# Set Age_Group value for all row indexes which Age are greater than 40
y1['num'][y1['time_hrs_bin'] == "Early Morning"] = 1
# Set Age_Group value for all row indexes which Age are greater than 18 and < 40
y1['num'][y1['time_hrs_bin'] == "Morning"] = 2
# Set Age_Group value for all row indexes which Age are less than 18
y1['num'][y1['time_hrs_bin'] == "afternoon"] = 3
y1['num'][y1['time_hrs_bin'] == "Evening"] = 4
y1['num'][y1['time_hrs_bin'] == "Night"] = 5
y1['num'][y1['time_hrs_bin'] == "Late Night"] = 6

time_dist_pos = y1['num'][(y1['userID'] > 0)]
sns.distplot(time_dist_pos, kde=True, hist=True, hist_kws={"range": [0,7]})
plt.show()
time_dist_pos.describe()
#y1 = pd.DataFrame({'time_hrs_bin':[1,2,3,4,5]})
s,p = stats.ttest_ind(y1[y1['userID']>0]['num'], y1[y1['userID']<0] ['num'])


time_dist_con = y1['num'][(y1['userID'] < 0)]
sns.distplot(time_dist_con, kde=True, hist=True, hist_kws={"range": [0,7]})
plt.show()



#train and test with time buckets data as predictor

predictors = y1['time_hrs_bin']
target = y1['class']


#pd.crosstab(data_new['fleschkin_score_bin'], data_new['class'], margins=True)
pd.crosstab(y1['time_hrs_bin'], y1['class'], margins=True)


df_1=pd.get_dummies(y1[['class', 'time_hrs_bin']], prefix='dummy', drop_first=True)
df_1.columns

predictor_names=df_1.drop('dummy_positive', axis=1).columns
predictor_names
predictors=df_1[predictor_names]
target= df_1['dummy_positive']



X_train, X_test, y_train, y_test = train_test_split(predictors, target,
    test_size=0.20, random_state=1)

#logistic Regression model

logr_model = linear_model.LogisticRegression()
func=logr_model.fit(X_train, y_train) 

logr_model.score(X_train, y_train)
coeff_value_list=logr_model.coef_
coeff_value_list
predictions=logr_model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))

#Decision Tree Model
dt_model = tree.DecisionTreeClassifier(criterion='entropy') 
dt_model.fit(X_train, y_train)
predictions=dt_model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))

#SVM model
from sklearn import svm 
svm_model = svm.SVC() 
svm_model.fit(X_train, y_train)
predictions=svm_model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))

#y_ = pd.DataFrame.from_dict(y)
#writer = pd.ExcelWriter('positive_negetive_occurance.xlsx')
#y_.to_excel(writer,'Sheet1')
#idx = x.groupby(['Mt'])['count'].transform(max) == df['occurences']
#data_con = shuffle(data_con)
#
