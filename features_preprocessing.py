import pandas as pd
from configuration import CONFIG
import numpy as np
import datetime as dt
import logging
import time
logging.basicConfig(filename='log_27112016_2.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
dateparse = lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S.000')

def find_day(day):             #Transforme les jours de la semaine en int compris entre 0 et 1
    if (day == "Dimanche"):
        return 0
    if (day == "Lundi"):
        return 1
    if (day == "Mardi"):
        return 2
    if (day == "Mercredi"):
        return 3
    if (day == "Jeudi"):
        return 4
    if (day == "Vendredi"):
        return 5
    if (day == "Samedi"):
        return 6


class feature_preprocessing():
    
    def __init__(self, filename = None, sepa = None, usecols = CONFIG.useful_columns):
        if filename == None and sepa == None:
            self.data = pd.DataFrame()
        else:
            self.data = pd.read_csv(filename, sep=sepa , usecols=usecols)

    def ass_id_creation(self): # Create ASS_ID (int between 0 and 28) from ASS_ASSIGNMENT as defined in configuration.py
        self.data['ASS_ID'] = self.data['ASS_ASSIGNMENT'].apply(lambda x: int(CONFIG.ass_assign[x]))
    
    def select_assid(self, assid):  #Select all the rows corresponding to a given ASS_ID 
        self.data=self.data.loc[self.data['ASS_ID'] == assid]
        self.data.drop('ASS_ID',axis = 1, inplace = True)
        self.data=self.data.groupby(['DATE','TPER_TEAM','DAY_WE_DS','WEEK_END']).sum()
        self.data.reset_index(inplace = True)
    
    def preprocess_date(self):   #Convert date from a string to a datetime format
        self.data["DATE"] = self.data["DATE"].apply(dateparse)
    
    def date_vector(self):  #Create features 'YEAR', 'MONTH', 'DAY', 'TIME' (first approach of feature engineering)
        self.data['YEAR'] = self.data['DATE'].apply(lambda x: x.tm_year)
        for year in ['2011','2012','2013']:
            self.data[year] = self.data['YEAR'].apply(lambda x: (int(year) == x)*1)
        
        self.data['MONTH'] = self.data['DATE'].apply(lambda x: x.tm_mon)
        for key, month in CONFIG.months.items():
            self.data[month] = self.data['MONTH'].apply(lambda x: int(x == key))
        
        self.data['TIME']= self.data['DATE'].apply(lambda x: x.tm_hour*60 + x.tm_min)
        #Creating time slots
        self.data['SLOT'] = self.data['TIME'].apply(lambda x: (x in range(450,1411))*(x-450)/30 + (x in range(0,451))*(x/30+1) + (x in range(1411,1441))*0)
        #self.data['TIME'] = self.data['TIME'].apply(lambda x: x in range(450,1411)*(x[0]-450)/30 + x in range(0,451)*(x/30+1) + x in range(1411,1441)*0)
        self.data['YEAR_DAY']= self.data["DATE"].apply(lambda x : x.tm_yday)
        self.data['DATE'] = self.data['DATE'].apply(lambda x: dt.datetime(x.tm_year,x.tm_mon,x.tm_mday,x.tm_hour,x.tm_min))        
        self.data.set_index('DATE', inplace = True)
    
    def normalize_Calls (self):
        mean = self.data['CSPL_RECEIVED_CALLS'].mean()
        variation = self.data['CSPL_RECEIVED_CALLS'].max()- self.data['CSPL_RECEIVED_CALLS'].min()
        self.data['CSPL_RECEIVED_CALLS']= (self.data['CSPL_RECEIVED_CALLS']-mean)/variation
    
    #Creating features 'before...'
    def lastvalue(self, nday):
        copy= pd.DataFrame()
        copy['before']= self.data.ix[self.data.index-dt.timedelta(days = nday)] ['CSPL_RECEIVED_CALLS']
        copy.index+=dt.timedelta(days=nday)
        self.data['before'+ str(nday)]=copy['before']
        self.data['before'+ str(nday)].fillna(self.data['CSPL_RECEIVED_CALLS'], inplace=True)
    
    #Creating feature 'MAX': maximum number of calls for the same day of the week and same hour over the past weeks        
    def valeur_max(self):
        self.data['MAX'] = np.maximum.reduce([self.data['before7'], self.data['before14'], self.data['before21'], self.data['before28'], self.data['before35']])
    
    def jour_nuit_creation(self):  #Création de la feature jour nuit
        
        tper_team = self.data['TPER_TEAM'].values.tolist()
        jour = []
        nuit = []
        nrows = len(self.data.index)
        
        for i in range(nrows):
            if(tper_team[i] == "Jours"):
                jour.append(1)
                nuit.append(0)
            else:
                nuit.append(1)
                jour.append(0)
        
        self.data['JOUR'] = jour
        self.data['NUIT'] = nuit

    def week_day_to_vector(self):  #Création des Features SUNDAY, MONDAY, TUESDAY, etc qui prennent les valeurs 0 ou 1
        week_day = self.data['DAY_WE_DS'].map(lambda day: find_day(day))
        self.data['WEEK_DAY'] = week_day
        for key,day in CONFIG.days.items():
            self.data[day] = self.data['WEEK_DAY'].apply(lambda x: int(x == key))
                
                
    def full_preprocess(self, assid, used_columns=CONFIG.default_columns, keep_all = False):
        self.ass_id_creation()
        self.select_assid(assid)
        self.preprocess_date()
        self.date_vector()
        for day in [7,14,21,28,35,56,84,112,119,140,168]:
            self.lastvalue(day) #Creating of features 'before...'
        self.valeur_max() #Creating of feature 'MAX'
        self.jour_nuit_creation()
        self.week_day_to_vector()
        #self.normalize_Calls()
        
        if not keep_all:
            self.data = self.data[used_columns]


if __name__ == "__main__": #execute the code only if the file is executed directly and not imported
    pp = feature_preprocessing('train_2011_2012_2013.csv',';')
    pp.full_preprocess(0, keep_all = False)
    print(pp.data.head(n=100000))
#    pp.data.to_csv('../data_test1.csv', sep=";")





#print(pp.data.sort_values(by=['CSPL_CALLS'], ascending=[0]))


