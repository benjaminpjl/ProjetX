import pandas as pd
import operator
from numpy import *
from datetime import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from features_preprocessing import extract_date
from configuration import CONFIG
#data = read_csv('train_2011_2012_2013.csv', sep = ';')
#print(read_csv('train_2011_2012_2013.csv', sep = ';').shape)#labels = read_csv('train_2011_2012_2013.csv', sep = ';', nrows = 2)
#data.columns = labels.columns
#data = data.ix[:,CONFIG.useful_columns]
#data.to_csv('train_sample.csv',sep = ";")

#Moyenne cumulative au cours du temps

def date_to_days(date):
    date_debut = datetime(2011,1,1)
    date_last = date_parser(date)
    return (date_last - date_debut).days 

def date_parser(date):
    return extract_date(date)

def row_iterator(path, useful_cols):
    for row in pd.read_csv(path, sep=';', chunksize = 100000, usecols = useful_cols):
        yield row

def data_variance(ass_id):
    print('Loading data...')
    data = pd.read_csv('train_2011_2012_2013.csv', sep = ';', usecols = ['DATE','CSPL_RECEIVED_CALLS','ASS_ASSIGNMENT'])        
    data['DATE'] = data['DATE'].apply(date_parser)
    
    print('Looking for ASS_ID ',ass_id)
    data['ASS_ID'] = data['ASS_ASSIGNMENT'].apply(lambda x : int(CONFIG.ass_assign[x]))
    data.drop(['ASS_ASSIGNMENT'],axis = 1)
    
    print('Reducing data...')
    data = data[data['ASS_ID'] == ass_id].groupby(['DATE','ASS_ID']).sum()
    
    print('Calculating data variance...')
    print('Variance: ', data['CSPL_RECEIVED_CALLS'].var())
    print(data)

def cumulative_mean(ass_id):
    mean = [0]
    nb_date = 0
    k = 0
    it = row_iterator('train_2011_2012_2013.csv',['DATE','CSPL_RECEIVED_CALLS','ASS_ASSIGNMENT'])
    for row in it:
        k += 1
        row_date = row.copy()
        row_date['ASS_ID'] = row_date['ASS_ASSIGNMENT'].apply(lambda x : int(CONFIG.ass_assign[x]))
        row_date.drop(['ASS_ASSIGNMENT'],axis = 1)
        row_date['DATE'].apply(date_parser)
        row_date = row_date[row_date['ASS_ID'] == ass_id].groupby(['DATE','ASS_ID']).sum()
        mean_before = mean[-1]*nb_date
        nb_date += row_date.shape[0]    
        mean.append((mean_before + row_date['CSPL_RECEIVED_CALLS'].as_matrix().sum())/nb_date)
        
        print(k, mean[-1])
    
    plt.figure()
    plt.plot(mean)
    plt.legend()
    plt.show()
    
def received_calls_day(path):
    
    it = row_iterator(path, ['DATE','CSPL_RECEIVED_CALLS','ASS_ASSIGNMENT'])
    data_processed = pd.DataFrame()  
    
    for chunk in it:
        tmp = chunk.copy()        
        tmp['ASS_ID'] = tmp['ASS_ASSIGNMENT'].apply(lambda x: int(CONFIG.ass_assign[x]))        
        tmp['DATE'] = tmp['DATE'].apply(date_parser)        
        tmp = tmp.groupby(['DATE','ASS_ID']).sum()
        tmp.reset_index(drop=True)
        data_processed = pd.concat([data_processed,tmp])
        print('OK')
    print('End for concatenation...')    
    data_processed.to_csv('received_calls_day.csv',sep = ';')
    print(data_processed.shape[0]," dates")    
    print('Job done ! ;)')


def plot_calls_per_day(ass_id, all_id = False):
    data = pd.read_csv('received_calls_day.csv', sep = ';')
    print('OK READ')
    if not all_id:
        data = data[data['ASS_ID'] == ass_id].groupby(['DATE','ASS_ID']).sum()
    else:
        data = data.groupby(['DATE']).sum()
    print('OK GROUPBY')
    data = data.reset_index()
    print('OK RESET')
    print(data)
    plt.figure()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    dates = [datetime.strptime(data['DATE'].ix[j], "%Y-%m-%d") for j in arange(data.shape[0])]
    plt.plot(dates,data['CSPL_RECEIVED_CALLS'])
    plt.gcf().autofmt_xdate()
    plt.show()

#data = pd.read_csv('train_2011_2012_2013.csv', sep = ';', usecols = ['DATE','ASS_ASSIGNMENT','CSPL_RECEIVED_CALLS'])


plot_calls_per_day(0,False)
#cumulative_mean(22)
#data_variance(22)




    
    
    
