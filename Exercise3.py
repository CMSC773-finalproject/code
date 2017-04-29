# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:52:10 2016

@author: Yash
"""

import csv
import math
from matplotlib import pyplot as plt
import pandas as pd
import textblob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from collections import Counter
import re
from collections import Counter
from collections import defaultdict
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree

#outcomes_df = pd.read_csv("outcomes.csv")
#projects_df = pd.read_csv("projects.csv")
#donations_df = pd.read_csv("donations.csv")
#essays_df = pd.read_csv("essays2.csv",encoding = 'iso-8859-1')
#essays_df_2=essays_df["short_description"]
#essays_df_2
#
#for row in essays_df_2:
#    x=row.split(' ')
#    for i in x:
#        i=i.replace(".","")
#        i=i.replace(",","")
#        i=i.replace("!","")
#        i=i.replace("?","")
#        print (x.count(i), i)


def clean_essay(string, lower=False):
    string = re.sub(r"\\t", " ", string)     #replace tab with a space
    string = re.sub(r"\\n", " ", string)     #replace next line with a space
    string = re.sub(r"\\r", " ", string)   
    string = re.sub(r"[^A-Za-z0-9\']", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)
    if lower:
        string = string.lower()
    return string.strip()
 
a = pd.read_csv("outcomes.csv")
#label_obj = preprocessing.LabelEncoder()
#a['is_exciting_level'] = label_obj.fit_transform(a['is_exciting'])
#b = pd.read_csv("essays.csv")
#c = pd.read_csv("projects.csv")
d = pd.read_csv("donations.csv")
#b = b.dropna(axis=1)
#a=a.dropna(axis=1)
#merged = a.merge(b, on='projectid')
#joining outcomes.csv and essays.csv
#merged = pd.merge(b,a,how='left',on='projectid')
#merged = pd.merge(b,a,how='inner',on='projectid')
#merged["split"] = "train"
#merged["split"][merged["date_posted"]<"2010-04-01"] = "none"
#merged["split"][merged["date_posted"]>="2013-01-01"] = "val"
#merged["split"][merged["date_posted"]>="2014-01-01"]= "test"
#merged = merged[merged["split"]!="none"]
#merged["y"] = 0
#merged["y"][merged["is_exciting"]=="t"] = 1
#
#text_vars=["title", "short_description", "need_statement", "essay"]
#for var in text_vars:
#    merged[var][pd.isnull(merged[var])] = ""
#    merged[var] = merged[var].apply(clean_essay)
#merged.to_csv("merge.csv", index=False)

#joining outcomes.csv and donations.csv
df = pd.merge(d, a, how='left', on='projectid')
df.columns

df.describe(include='all')
df.corr()
df.isnull().any()
#df.applymap(np.isreal).any()
# Review unique values in categorical fields
pd.unique(df[['donor_state', 'dollar_amount',  'payment_method', 'for_honoree']].values.ravel())
df['donor_state'].unique()
df['dollar_amount'].unique()
df['is_teacher_acct'].unique()
df['donation_included_optional_support'].unique()
df['payment_included_acct_credit'].unique()
df['payment_included_campaign_gift_card'].unique()
df['payment_included_web_purchased_gift_card'].unique()
df['via_giving_page'].unique()
df['for_honoree'].unique()

df=df.drop('donor_city',1)
df=df.drop('donation_timestamp',1)
df=df.drop('donation_message',1)
df=df.drop('donor_zip',1)
df=df.drop('donationid',1)
df=df.drop('donor_acctid',1)
df=df.drop('donation_to_project',1)

df.columns

# Remove missing values - there are very few of them
#dfx=df.replace({'?': np.nan}, regex=False)
df=df.dropna(axis=0)

# Review various attributes vs the outcome - and combine some attribute values
pd.crosstab(df['donor_state'], df['is_exciting'], margins=True)
pd.crosstab(df['is_teacher_acct'], df['is_exciting'], margins=True)
pd.crosstab(df['donation_total'], df['is_exciting'], margins=True)
pd.crosstab(df['donation_included_optional_support'], df['is_exciting'], margins=True)
pd.crosstab(df['payment_method'], df['is_exciting'], margins=True)
pd.crosstab(df['payment_included_acct_credit'], df['is_exciting'], margins=True)
pd.crosstab(df['payment_included_campaign_gift_card'], df['is_exciting'], margins=True)
pd.crosstab(df['payment_included_web_purchased_gift_card'], df['is_exciting'], margins=True)
pd.crosstab(df['payment_was_promo_matched'], df['is_exciting'], margins=True)
pd.crosstab(df['via_giving_page'], df['is_exciting'], margins=True)
pd.crosstab(df['for_honoree'], df['is_exciting'], margins=True)
pd.crosstab(df['dollar_amount'], df['is_exciting'], margins=True)

# Process State
def group_states(state):
    if state in ('AL','AR',
       'IN', 'IL', 'KY', 'KS', 'LA', 'MO', 'MN', 'MS', 'NE', 'ND', 'OK', 'SD', 'TX', 'TN', 'WI'):
        return 'CST'
    elif state in ('FL', 'NJ', 'MD',
                     'NY', 'IA', 'CT', 'DE', 'GA', 'MA', 'ME', 'MI', 'NH', 'NC', 'OH', 'PA', 'RI', 'SC', 'VA', 'VT', 'DC', 'WV'):
        return 'EST'
    elif state in ('CA', 'NV', 'OR', 'WA'):
        return 'PST'
    elif state in('AZ', 'CO', 'ID', 'MT', 'NM', 'UT', 'WY'):
        return 'MST'
    else:
        return 'other'
df['state_bin']=df['donor_state'].map(group_states)

# Process dollar amount
def group_amount_dollar(amount):
    if amount in('10_to_100'):
        return 'Between 10 and 100'
    elif amount in('under_10'):
        return 'Below 10'
    else:
        return 'above 100'    
df['amount_bin']=df['dollar_amount'].map(group_amount_dollar)

# Process payment method
def group_payment_method(payment):
    if payment in('paypal', 'creditcard', 'amazon', 'check'):
        return 'Online'
    elif payment in('promo_code_match', 'double_your_impact_match', 'almost_home_match'):
        return 'Discount'
    else:
        return 'no cash received'    
df['payment_bin']=df['payment_method'].map(group_payment_method)

# Process is exciting
#def group_is_exciting(value):
#    if value in('f'):
#        return 'false_is_exciting'
#    elif value in('t'):
#        return 'true_is_exciting'
#    else:
#        return value    
#df['exciting_bin']=df['is_exciting'].map(group_is_exciting)

#process for honoree
def group_for_honoree(value):
    if value=='f':
        return 'false_for_honoree'
    elif value=='t':
        return 'true_for_honoree'
    else:
        return value    
df['honoree_bin']=df['for_honoree'].map(group_for_honoree)

#label_obj = preprocessing.LabelEncoder()
#df['honoree_bin'] = label_obj.fit_transform(df['for_honoree'])
#df['honoree_bin'].unique()

#process via_giving_page
def group_via_giving_page(value):
    if value in('f'):
        return 'false_via_giving_page'
    elif value in('t'):
        return 'true_via_giving_page'
    else:
        return value    
df['via_giving_page_bin']=df['via_giving_page'].map(group_via_giving_page)

#process payment_was_promo_matched
def group_payment_was_promo_matched(value):
    if value in('f'):
        return 'false_payment_was_promo_matched'
    elif value in('t'):
        return 'true_payment_was_promo_matched'
    else:
        return value    
df['payment_was_promo_matched_bin']=df['payment_was_promo_matched'].map(group_payment_was_promo_matched)


#process payment_included_web_purchased_gift_card
def group_payment_included_web_purchased_gift_card(value):
    if value in('f'):
        return 'false_payment_included_web_purchased_gift_card'
    elif value in('t'):
        return 'true_payment_included_web_purchased_gift_card'
    else:
        return value    
df['payment_included_web_purchased_gift_card_bin']=df['payment_included_web_purchased_gift_card'].map(group_payment_included_web_purchased_gift_card)

#process payment_included_campaign_gift_card
def group_payment_included_campaign_gift_card(value):
    if value in('f'):
        return 'false_payment_included_campaign_gift_card'
    elif value in('t'):
        return 'true_payment_included_campaign_gift_card'
    else:
        return value    
df['payment_included_campaign_gift_card_bin']=df['payment_included_campaign_gift_card'].map(group_payment_included_campaign_gift_card)

#process payment_included_acct_credit
def group_payment_included_acct_credit(value):
    if value in('f'):
        return 'false_payment_included_acct_credit'
    elif value in('t'):
        return 'true_payment_included_acct_credit'
    else:
        return value    
df['payment_included_acct_credit_bin']=df['payment_included_acct_credit'].map(group_payment_included_acct_credit)

#process donation_included_optional_support
def group_donation_included_optional_support(value):
    if value in('f'):
        return 'false_donation_included_optional_support'
    elif value in('t'):
        return 'true_donation_included_optional_support'
    else:
        return value    
df['donation_included_optional_support_bin']=df['donation_included_optional_support'].map(group_donation_included_optional_support)

#process is_teacher_acct
def group_is_teacher_acct(value):
    if value in('f'):
        return 'false_is_teacher_acct'
    elif value in('t'):
        return 'true_is_teacher_acct'
    else:
        return value    
df['is_teacher_acct_bin']=df['is_teacher_acct'].map(group_is_teacher_acct)

#creating dummies
#df2=pd.get_dummies(df[['is_exciting','honoree_bin','via_giving_page_bin','payment_was_promo_matched_bin','payment_included_web_purchased_gift_card_bin', 'payment_included_campaign_gift_card_bin', 'payment_included_acct_credit_bin', 'donation_included_optional_support_bin', 'is_teacher_acct_bin', 'amount_bin', 'payment_bin','state_bin']], prefix='dummy', drop_first=True)
df2=pd.get_dummies(df[['is_exciting','payment_was_promo_matched_bin' ,'donation_included_optional_support_bin', 'is_teacher_acct_bin', 'payment_bin','amount_bin', 'state_bin']], prefix='dummy', drop_first=True)
df2.columns
#df2['dummy_true_for_honoree'].unique()
#df2['dummy_t'].unique()
x=df2['dummy_t']
x.count()
df2.columns
df3=pd.concat([df[['projectid']],df2], axis=1)

predictor_names=df2.drop('dummy_t', axis=1).columns
predictor_names
predictors=df2[predictor_names]
target= df2['dummy_t']
df3.columns
df2.count

#sorting correlation table in ascending order with respect to dummy_true_is_exciting 
c=df3.corr() 
s = c.unstack() 
so = s.sort_values() 
print(so.loc['dummy_t'])

#analysis
X_train, X_test, y_train, y_test = train_test_split(predictors, target,
    test_size=0.33, random_state=123)
    
logr_model = linear_model.LogisticRegression()
func=logr_model.fit(X_train, y_train)
logr_model.score(X_train, y_train)
coeff_value_list=logr_model.coef_
coeff_value_list
predictions=logr_model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))


dt_model = tree.DecisionTreeClassifier()
#(criterion='entropy')
dt_model.fit(X_train, y_train) 
dt_model
dt_model.feature_importances_
predictions=dt_model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))






a = pd.read_csv("outcomes.csv")
d = pd.read_csv("resources.csv")
#b = b.dropna(axis=1)
#a=a.dropna(axis=1)
#merged = a.merge(b, on='projectid')
#joining outcomes.csv and essays.csv
#merged = pd.merge(b,a,how='left',on='projectid')
#merged = pd.merge(b,a,how='inner',on='projectid')
#merged["split"] = "train"
#merged["split"][merged["date_posted"]<"2010-04-01"] = "none"
#merged["split"][merged["date_posted"]>="2013-01-01"] = "val"
#merged["split"][merged["date_posted"]>="2014-01-01"]= "test"
#merged = merged[merged["split"]!="none"]
#merged["y"] = 0
#merged["y"][merged["is_exciting"]=="t"] = 1
#

#joining outcomes.csv and donations.csv
df = pd.merge(d, a, how='left', on='projectid')
df.columns

df.describe(include='all')
df.corr()
df.isnull().any()
df=df.dropna(axis=0)
#df.applymap(np.isreal).any()
# Review unique values in categorical fields
df['project_resource_type'].unique()
df['item_quantity'].min()
df['item_unit_price'].max()
df['vendor_name'].unique()
df = df[df.item_unit_price > 0]

df=df.drop('resourceid',1)
df=df.drop('vendorid',1)
df=df.drop('item_name',1)
df=df.drop('item_number',1)

df.columns

# Remove missing values - there are very few of them
df=df.dropna(axis=0)

# Review various attributes vs the outcome - and combine some attribute values
pd.crosstab(df['vendor_name'], df['is_exciting'], margins=True)
pd.crosstab(df['project_resource_type'], df['is_exciting'], margins=True)
pd.crosstab(df['item_unit_price'], df['is_exciting'], margins=True)
pd.crosstab(df['item_quantity'], df['is_exciting'], margins=True)

# Process item_unit_price
def group_item_unit_price(amount):
    if amount>0 and amount<100:
        return 'Between 0 and 100'
    elif amount>=100 and amount<1000:
        return 'Between 100 and 1000'
    elif amount>=1000 and amount<10000:
        return 'Between 1000 and 10000'
    elif amount>=10001 and amount<20000:
        return 'Between 10000 and 20000'
    elif amount>=20001 and amount<35000:
        return 'Between 20000 and 35000'
    else:
        return 'above 35000'    
df['item_unit_price_bin']=df['item_unit_price'].map(group_item_unit_price)

# Process item_quantity
def group_item_quantity(amount):
    if amount>0 and amount<10:
        return 'Between 0 and 10'
    elif amount>=11 and amount<100:
        return 'Between 10 and 100'
    elif amount>=101 and amount<500:
        return 'Between 100 and 500'
    elif amount>=501 and amount<1000:
        return 'Between 500 and 1000'
    else:
        return 'above 1000'    
df['item_quantity_bin']=df['item_quantity'].map(group_item_quantity)

#creating dummies
df2=pd.get_dummies(df[['is_exciting','item_unit_price_bin','item_quantity_bin','vendor_name','project_resource_type']], prefix='dummy', drop_first=True)
#df2=pd.get_dummies(df[['is_exciting','payment_was_promo_matched_bin' ,'donation_included_optional_support_bin', 'is_teacher_acct_bin', 'payment_bin','amount_bin', 'state_bin']], prefix='dummy', drop_first=True)
df2.columns
#df2['dummy_true_for_honoree'].unique()
#df2['dummy_t'].unique()
x=df2['dummy_t']
x.count()
df2.columns
df3=pd.concat([df[['projectid']],df2], axis=1)

predictor_names=df2.drop('dummy_t', axis=1).columns
predictor_names
predictors=df2[predictor_names]
target= df2['dummy_t']
df3.columns
df2.count

#sorting correlation table in ascending order with respect to dummy_true_is_exciting 
c=df3.corr() 
s = c.unstack() 
so = s.sort_values() 
print(so.loc['dummy_t'])

#analysis
X_train, X_test, y_train, y_test = train_test_split(predictors, target,
    test_size=0.33, random_state=123)
    
logr_model = linear_model.LogisticRegression()
func=logr_model.fit(X_train, y_train)
logr_model.score(X_train, y_train)
coeff_value_list=logr_model.coef_
coeff_value_list
predictions=logr_model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))


dt_model = tree.DecisionTreeClassifier()
#(criterion='entropy')
dt_model.fit(X_train, y_train) 
dt_model
dt_model.feature_importances_
predictions=dt_model.predict(X_test)
print(metrics.classification_report(y_test,predictions))
print(metrics.confusion_matrix(y_test,predictions))