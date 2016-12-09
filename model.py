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
from tqdm import tqdm
  
    
#Loading training dataset and submission dataset    
preprocessing = fp.feature_preprocessing('train_2011_2012_2013.csv', ';')
submission = pd.read_csv('submission.txt', sep = '\t')
submission['ASS_ID'] = submission['ASS_ASSIGNMENT'].apply(lambda x: int(CONFIG.ass_assign[x]))

print('Ready for training...')

#Training model on 27 assignments
for id in range(28):
    
    print("Working on assignment : ",id)
    preprocessing_id = fp.feature_preprocessing()
    preprocessing_id.data = preprocessing.data.copy()
    preprocessing_id.full_preprocess(id)
    data = preprocessing_id.data
    print(id,' Data loaded')
    Y = data['CSPL_RECEIVED_CALLS']
    X = data.drop(['CSPL_RECEIVED_CALLS'], axis=1)
    
    
    print(id,' Data ready to be used')
    

    print('Loading submission')
    submission_id = sp.submission_preprocessing()
    submission_id.data = submission.copy()
    submission_id.full_preprocess(id) #Choose columns
    
    #Create time_series features 'before7', 'before14', ... corresponding to the number of calls 1 week, 2 weeks, etc before the given date
    #Succession of if statements ensure that a value is available for these features (exemple: if 'before14' does not exist use 'before7' value instead)
    for date in submission_id.data.index:
            if (date - dt.timedelta(days = 7)) in data.index:
                tmp = data.loc[date - dt.timedelta(days = 7)]
                submission_id.data.loc[date,'before7'] = tmp['CSPL_RECEIVED_CALLS']
            else:
                submission_id.data.loc[date,'before7'] = 0
            
            if (date - dt.timedelta(days = 14) in data.index):            
                submission_id.data.loc[date,'before14'] = data['CSPL_RECEIVED_CALLS'].loc[date - dt.timedelta(days = 14)]
            else:
                submission_id.data.loc[date,'before14'] = submission_id.data.loc[date,'before7']
            
            
            if (date - dt.timedelta(days = 21) in data.index):            
                submission_id.data.loc[date,'before21'] = data['CSPL_RECEIVED_CALLS'].loc[date - dt.timedelta(days = 21)]
            else:
                submission_id.data.loc[date,'before21'] = submission_id.data.loc[date,'before14']

            
            if (date - dt.timedelta(days = 28) in data.index):            
                submission_id.data.loc[date,'before28'] = data['CSPL_RECEIVED_CALLS'].loc[date - dt.timedelta(days = 28)]
            else:
                submission_id.data.loc[date,'before28'] = submission_id.data.loc[date,'before21']
            
            
            if (date - dt.timedelta(days = 56) in data.index):            
                submission_id.data.loc[date,'before56'] = data['CSPL_RECEIVED_CALLS'].loc[date - dt.timedelta(days = 56)]
            else:
                submission_id.data.loc[date,'before56'] = submission_id.data.loc[date,'before28']
            
            if date - dt.timedelta(days = 84) in data.index:            
                submission_id.data.loc[date,'before84'] = data['CSPL_RECEIVED_CALLS'].loc[date - dt.timedelta(days = 84)]
            else:
                submission_id.data.loc[date,'before84'] = submission_id.data.loc[date,'before56']
            
            if (date - dt.timedelta(days = 112) in data.index):            
                submission_id.data.loc[date,'before112'] = data.loc[date - dt.timedelta(days = 112)]['CSPL_RECEIVED_CALLS']
            else:
                submission_id.data.loc[date,'before112'] = submission_id.data.loc[date,'before84']            
                
            
            if (date - dt.timedelta(days = 140) in data.index):            
                submission_id.data.loc[date,'before140'] = data.loc[date - dt.timedelta(days = 140)]['CSPL_RECEIVED_CALLS']
            else:
                submission_id.data.loc[date,'before140'] = submission_id.data.loc[date,'before112']
            
            if (date - dt.timedelta(days = 168) in data.index):            
                submission_id.data.loc[date,'before168'] = data.loc[date - dt.timedelta(days = 168)]['CSPL_RECEIVED_CALLS']
            else:
                submission_id.data.loc[date,'before168'] = submission_id.data.loc[date,'before140']    
    
    #Creation of X_test['MAX']: maximum value for CSPL_RECEIVED_CALLS on the previous weeks.
    for week in submission_id.data['WEEK'].unique():    
        X_test = submission_id.data.loc[data['WEEK'] == week].drop(['prediction','DATE','', axis = 1)
        X_test['MAX'] = np.maximum.reduce([X_test['before7'], X_test['before14'], X_test['before21'], X_test['before28'], X_test['before56'], X_test['before112'], X_test['before140'], X_test['before168'] ] )
    
      
        if (X_test.shape[0] > 0):
            Y_pred = [0]*X_test.shape[0]        
            for k in tqdm(range(5)):        
                cv_score = 0
                Y_pred = [0]*X_test.shape[0]
                X_train, X_cv, Y_train, Y_cv = cross_validation.train_test_split(X, Y, test_size = 0.2, random_state=0)
                X_train = X_train.drop(['MAX'], axis=1)
                X_cv_max = X_cv['MAX']
                X_cv = X_cv.drop(['MAX'], axis=1)
                
                clf = RandomForestRegressor(n_estimators=1000, oob_score=True)
                clf.fit(X_train,Y_train)
               
            # Cross validation score                                                  
                Y_pred_cv = clf.predict(X_cv)
                Y_pred_cv = np.maximum(Y_pred_cv, X_cv_max)
                cv_score = loss_linex(Y_cv,Y_pred_cv).mean()
           
            #Prediction for X_test                                                     
                Y_pred_tmp = clf.predict(X_test.drop('MAX',axis = 1))
                Y_pred_tmp = np.maximum(Y_pred_tmp,X_test['MAX'])   #Take the maximum between the new prediction and the feature X_test['MAX']. Ensures that we never underestimate the walue of received calls.
                Y_pred = np.maximum.reduce([Y_pred, Y_pred_tmp] ) #Take the maximum value for Y_pred in the cross validation
                          
                print('Cross validation Done')

    submission.loc[submission['ASS_ID'] == id,'prediction']=Y_pred  #Fill column 'prediction' with predicted values for the Ass_ID



submission = submission.drop(['ASS_ID'], axis = 1)  # Drop column ASS_ID
submission.reset_index(inplace = True, drop = True)
submission.to_csv('submission_new.txt', sep = '\t', encoding = 'utf-8', index = False)
print('Job done')





