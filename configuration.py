class _Config:
    def __init__(self):
        # All samples
        self.useful_columns = ['DATE', 'DAY_WE_DS', 'WEEK_END', 'ASS_ASSIGNMENT', 'TPER_TEAM', 'CSPL_RECEIVED_CALLS']
        self.months = { 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        self.days = {1: "MONDAY", 2: "TUESDAY", 3: "WEDNESDAY", 4: "THURSDAY", 5: 'FRIDAY', 6: "SATURDAY", 0: "SUNDAY"}
        self.years = {1: "2011", 2: "2012", 3:"2013"}
        self.ass_assign = {'Téléphonie':0, 'RTC':1, 'Gestion Renault':2, 'Nuit':3,'Gestion - Accueil Telephonique':4, 'Regulation Medicale':5, 'Services':6,'Tech. Total':7, 'Gestion Relation Clienteles':8, 'Crises':9, 'Japon':10, 'Médical':11, 'Gestion Assurances':12, 'Domicile':13, 'Gestion':14, 'SAP':15, 'RENAULT':16, 'Gestion Amex':17, 'Tech. Inter':18, 'Gestion Clients':19, 'Manager':20, 'Tech. Axa':21, 'CAT':22, 'Gestion DZ':23, 'Mécanicien':24, 'CMS':25, 'Prestataires':26, 'Evenements':27}
        self.before = {1: 'before7',2:'before14', 3:'before21', 4:'before28', 6:'before56',7:'before84', 8:'before112',11:'before140',12:'before168'}
        
    
        self.default_columns = list(self.before.values())+['2012','2011','JOUR','MAX','CSPL_RECEIVED_CALLS']
        
        self.sub_columns = ['2012','2011','JOUR','prediction']        
#        self.sub_columns = list(self.before.values())+['2012','2011','JOUR','MAX','prediction']
        self.test_columns = list(self.before.values())+['2012','2011','JOUR','MAX','CSPL_RECEIVED_CALLS']
CONFIG = _Config()

print (CONFIG.default_columns)

