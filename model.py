
import sys
import os
sys.path.append(os.path.abspath("/Users/benjaminpujol/Desktop/Projet-avec-le-mexicain"))
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
from learn_xgb import learn_xgb
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
    #data = preprocessing_id.data.reset_index(drop=True)
    print(id,' Data loaded')
    #data.to_csv('X.txt', sep='\t')
    
    Y = data['CSPL_RECEIVED_CALLS']
    X = data.drop(['CSPL_RECEIVED_CALLS'], axis=1)
    X.info()
    
    
    
    print(id,' Data ready to be used')
    

    print('Loading submission')
    submission_id = sp.submission_preprocessing()
    submission_id.data = submission.copy()
    submission_id.full_preprocess(id) #Choose columns
    X_test = submission_id.data.drop('prediction', axis = 1)
    X_test.info()
    
    #Implement cross validation (10 splits)
    #if X_test.shape[0]!=0:
    #   Y_pred = [0]*X_test.shape[0]
    #   error=0
    #   for i in range(10):
    #       X_train, X_cv, Y_train, Y_cv = cross_validation.train_test_split(X, Y, test_size = 0.2, random_state=0)
    #       model= learn_xgb(X_train, Y_train)
    #       Y_pred_cv = model.predict(X_cv)
    #       error+= loss_linex(Y_pred_cv,Y_cv).sum()
    #       Y_pred += model.predict(X_test)
    
    if (X_test.shape[0] > 0):
        #Implement cross validation (10 splits)
        cv_score = 0
        Y_pred = [0]*X_test.shape[0]
        X_train, X_cv, Y_train, Y_cv = cross_validation.train_test_split(X, Y, test_size = 0.2, random_state=0)
        X_train = X_train.drop(['MAX'], axis=1)
        X_cv_max = X_cv['MAX']
        X_cv = X_cv.drop(['MAX'], axis=1)
        
        clf = RandomForestRegressor(n_estimators=50, oob_score=True)
        selector = RFE(clf, 5, step=1)
        selector = selector.fit(X_train, Y_train)
        print (selector.support_)
        print(selector.ranking_)
        clf.fit(X_train,Y_train)
        Y_pred_cv = clf.predict(X_cv)
        Y_pred_cv = np.maximum(Y_pred_cv, X_cv_max)
        cv_score = loss_linex(Y_cv,Y_pred_cv).mean()
        print('Computing mean cv score')
        print(id,cv_score)
        print('Cross validation Done')
        #pred = pd.DataFrame()
        #pred['prediction']=Y_pred_cv
        #pred['true value']=Y_cv.as_matrix()
        #pred.to_csv('Caca1.txt', sep='\t', index=False)

        submission.loc[submission['ASS_ID'] == id,'prediction']=Y_pred



submission = submission.drop(['ASS_ID'], axis = 1)
submission.to_csv('test.txt', sep = '\t')
print('Job done')





#AdaboostRegression

#clf2 = AdaBoostRegressor(n_estimators=100)
#clf2.fit(X_train,Y_train)
#predict_2 = clf2.predict(X_test)
#print(clf2.score(X_test,Y_test))

