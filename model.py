import features_preprocessing as fp
import submission_preprocessing as sp
from configuration import CONFIG
import linex as ln
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
from sklearn import linear_model
import datetime as dt
from linex import loss_linex
from sklearn.feature_selection import RFE
  
    
    
preprocessing = fp.feature_preprocessing('train_2011_2012_2013.csv', ';')
submission = pd.read_csv('submission.txt', sep = '\t')
submission['ASS_ID'] = submission['ASS_ASSIGNMENT'].apply(lambda x: int(CONFIG.ass_assign[x]))

print('Ready for training...')

for id in [0]:
    
    print("Working on assignment : ",id)
    preprocessing_id = fp.feature_preprocessing()
    preprocessing_id.data = preprocessing.data.copy()
    preprocessing_id.full_preprocess(id)
    data = preprocessing_id.data
    print(id,' Data loaded')
    
    Y = data['CSPL_RECEIVED_CALLS']
    X = data.drop(['CSPL_RECEIVED_CALLS'], axis=1)
    data.info()
    
    
    print(id,' Data ready to be used')
    

    print('Loading submission')
    submission_id = sp.submission_preprocessing()
    submission_id.data = submission.copy()
    submission_id.full_preprocess(id) #Choose columns
    for date in submission_id.data.index:
        if (date - dt.timedelta(days = 7)) in data.index:
            tmp = data.loc[date - dt.timedelta(days = 7)]
            print(tmp)
            submission_id.data.loc[date,'before7'] = tmp['CSPL_RECEIVED_CALLS']
    X_test = submission_id.data.drop('prediction', axis = 1)
    X_test.info()
    print(X_test.head(n=10))
#    
#    #Implement cross validation (10 splits)
#    #if X_test.shape[0]!=0:
#    #   Y_pred = [0]*X_test.shape[0]
#    #   error=0
#    #   for i in range(10):
#    #       X_train, X_cv, Y_train, Y_cv = cross_validation.train_test_split(X, Y, test_size = 0.2, random_state=0)
#    #       model= learn_xgb(X_train, Y_train)
#    #       Y_pred_cv = model.predict(X_cv)
#    #       error+= loss_linex(Y_pred_cv,Y_cv).sum()
#    #       Y_pred += model.predict(X_test)
#    
#    if (X_test.shape[0] > 0):
#        cv_score = 0
#        Y_pred = [0]*X_test.shape[0]
#        X_train, X_cv, Y_train, Y_cv = cross_validation.train_test_split(X, Y, test_size = 0.2, random_state=0)
#        X_train = X_train.drop(['MAX'], axis=1)
#        X_cv_max = X_cv['MAX']
#        X_cv = X_cv.drop(['MAX'], axis=1)
#        
#        clf = RandomForestRegressor(n_estimators=50, oob_score=True)
#        selector = RFE(clf, 5, step=1)
#        selector = selector.fit(X_train, Y_train)
#        print (selector.support_)
#        print(selector.ranking_)
#        clf.fit(X_train,Y_train)
#        Y_pred_cv = clf.predict(X_cv)
#        Y_pred_cv = np.maximum(Y_pred_cv, X_cv_max)
#        cv_score = loss_linex(Y_cv,Y_pred_cv).mean()
#        print('Computing mean cv score')
#        print(id,cv_score)
#        print('Cross validation Done')
#
#        submission.loc[submission['ASS_ID'] == id,'prediction']=Y_pred
#
#
#
#submission = submission.drop(['ASS_ID'], axis = 1)
#submission.to_csv('test.txt', sep = '\t')
#print('Job done')
#
#
#


