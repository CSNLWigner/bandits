
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import metrics
import datetime
from openpyxl import load_workbook
import warnings

import warnings
warnings.filterwarnings("once")


class Exponential_model():

    reward_coding = None
    cmw_coding = None
    
    def __init__(self, data, FSS, fit_a2, fit_eps, dropmissing):
        
        '''
        - data: data
        - name: 
        - FSS: Specify whether the data is feedback source sensitive. Eg. strategy aligned. 
        - fit_a2: {None, independent, pseudo}
        - fit_eps: True/False. It specify whether the lower reward remains 0 in the {0, 1} coding regime, 
            or 0 is changed "eps" which is a fitted value.
        '''
        
        #self.model_name = model_name
        self.data = data
        self.FSS = FSS
        self.fit_a2 = fit_a2
        self.fit_eps = fit_eps
        self.dropmissing = True
        
        
    @staticmethod    
    def sigmoid(x): ## priviously get_porob
        prob = np.exp(x)/(1+np.exp(x))
        return prob
        
    def sigmoid2(x):
        output = 1/(1+np.exp(-x))
        return output

    
    def get_probs_by_exponent(self, a = 700, a2 = 0, b = -0.5, c = 20, eps = 1e-5): #self, self.data, self.FSS, a=700, b=-0.5, c=20, eps=1e-5, dropmissing = True):
        
        #print ('self.FSS', self.FSS)

        ## Calculate the action value for each trial.
        cols = ['t-'+str(item) for item in range(1, 9)]  ## get the colum names with increasing t-n value. t-1, t-2, etc. 
        rewards = np.array(self.data[cols]) ## get the reward values as a numpy array


        if self.fit_eps == True:
            assert [0, 1 ] == list(np.unique(self.data.loc[8:, ['t-1', 't-2', 't-3','t-4', 't-5','t-6', 't-7', 't-8']])), f"Epsilon fitting is possible only on the {0, 1} coding regime. Please check the\reward coding!"
            rewards[rewards == 0] = eps ## Here I change all the values from 0 to an epsilon value, which is fitted.
            
        self.reward_coding = set(np.unique(rewards[~np.isnan(rewards)]))
        
        
        #############################################################
        ## Get strategy aligned or feedback source sensitive weights. 
        ## Calculate the Choice-Match Weight for each trial.
        cmw_cols =  ['CMW_t-'+str(item) for item in range(1, 9)] ## get the column names that contains the CMW values. 
        cmw = np.array(self.data[cmw_cols]) ## get CMW values as an array
        self.cmw_coding = set(np.unique(cmw[~np.isnan(cmw)]))
        
        ## calculate cmwA, cmw + cmwA = [1,1,1,1,1,1,1,1]
        cmwA = cmw-np.ones((1,8)) #subtract ones
        cmwA = cmwA*(-1) # multiply the array with -1. 
        cmwA = cmwA ## keep only the inner array eg. insead of [[111...1]] -->  [111...1]

        
        #############################################################
        ## Calculate the exponential term
        t_array = np.arange(1,9) ## define the time variable, t. 
        prod_tb = t_array*b # t*b
        exponent_term = np.exp(prod_tb) # np.exp(t*b)

        #############################################################
        ################## FEEDBACK SOURCE BRACNCH ##################
        #############################################################
        ## CMW calculate rewards with cmw - according to the function param.
        assert type(self.FSS) == bool, f"FSS must be defined. Provide FSS a boolean argument: get_probs_by_exponent(FSS = True/False)."
        
        if self.FSS == True: 

            aer = a*exponent_term*(rewards*cmw) ## a*e^(b*t)*(r*cmw)
            if self.fit_a2 == 'pseudo': 
                a2 = -a   
            ##print ('a and a2 before aer2 calculation', a, a2)
            aer2 = a2*exponent_term*(rewards*cmwA) ## a'*e^(b*t)*(r*cmw_A)
        
            
        elif self.FSS == False:
            aer = a*exponent_term*rewards ## a*e^(b*t)*r*
            if self.fit_a2 == 'pseudo': 
                a2 = -a
            ##print ('a and a2 before aer2 calculation', a, a2)
            aer2 = a2*exponent_term*rewards ## a'*e^(b*t)*r
        
        
        #############################################################
        ######################## FIT A2 BRANCH ######################
        ############################################################# 
        assert self.fit_a2  in [False, 'independent', 'pseudo'], "Check fit_a2 paremeter value! It seems to be invalid!"
        
        if self.fit_a2 == False:
            aer_terms = aer 
            
        elif self.fit_a2 == 'independent' or self.fit_a2 == 'pseudo':
            aer_terms = aer + aer2
            
        ## calculate the action value based on the aer terms. 
        aer_terms_sum = aer_terms.sum(axis = 1)
        action_value = aer_terms_sum + c
        
    
        ## Assigned the calculated action values to the dataframe
        self.data['action_value'] = action_value
        self.data['norm_action_value'] = action_value/100
        self.data['exponent_model'] = self.sigmoid(self.data['norm_action_value'])
    #    data['prob_sigma2'] = sigmoid2(data['norm_action_value'])

        #############################################################
        ########### Make some sanity checks and formatting ##########
        #############################################################
            
        if self.data['exponent_model'].isnull().sum() > 8: 
            missing = self.data['exponent_model'].isnull().sum()
            raise Exception("Too many missing data. Nr. missing = {}. A too big or two small b value might cause this error.".format(missing))

        if sum(self.data[['action_value', 'exponent_model']].isnull().sum().to_list()) > 2*8:
            raise Exception("Please check out the table, some unexpected nans occured.")

        if self.dropmissing == True:
            self.data = self.data.dropna()
        return (self.data)




class Optimize_exponent():
    
    name = None
    log_loss = None
    params = {'a':None, 'a2': None, 'b':None, 'c': None, 'eps': None}
    reward_coding = None
    cmw_coding = None
    initial_params = None
    fit_a2 = None
    fit_eps = None
    data = None

    def save_data(self):
        self.data.to_csv('./data/' + self.name + '.csv', index = False)
            
    @staticmethod
    def get_log_loss(x, data, FSS, fit_a2, fit_eps, dropmissing):
        a, a2, b, c, eps = x
        #print (a, a2, b, c, eps) ## with this one can follow the iteration of the optimisation function. 
                                 ## note on pseudo a2: it won't display the right a2 value, to see that see the 
                                 ## coment in the get_probs_by_exponent() function. 
        model = Exponential_model(data, FSS, fit_a2, fit_eps, dropmissing)
        data_exp = model.get_probs_by_exponent(a = a, a2 = a2, b = b, c = c, eps = eps)
        ll = metrics.log_loss(data_exp['stay'], data_exp['exponent_model'])
        return (ll)
    
    def optimise_model(self, data, name, FSS, fit_a2, fit_eps, bounds, verbose = True, dropmissing = True):
        
        ### lehet, hogy itt valójában 6 különböző optimizációs funkciót kellene bevezetni?  
        
        
        bounds = bounds
        x0 = np.array([100, -20, -0.5, 10, 1e-4]) ## NB! Here there is room for improvement!!!!
        
        optimisation_result = sp.optimize.minimize(Optimize_exponent.get_log_loss, x0, args = (data, FSS, fit_a2, fit_eps, dropmissing), bounds = bounds)
        a, a2, b, c, eps = optimisation_result.x
        model = Exponential_model(data, FSS, fit_a2, fit_eps, dropmissing)

        
        if fit_eps == False:
            if fit_a2 == False: 
                data_exp = model.get_probs_by_exponent(a = a, a2 = 0, b = b, c = c, eps = 0)
                ll = metrics.log_loss(data_exp['stay'], data_exp['exponent_model'])
                a2 = None
                eps = None
            
            elif fit_a2 == 'independent':     
                data_exp = model.get_probs_by_exponent(a = a, a2 = a2, b = b, c = c, eps = 0)
                ll = metrics.log_loss(data_exp['stay'], data_exp['exponent_model'])
                eps = None
                
            elif fit_a2 == 'pseudo':    
                data_exp = model.get_probs_by_exponent(a = a, a2 = -a, b = b, c = c, eps = 0)
                ll = metrics.log_loss(data_exp['stay'], data_exp['exponent_model'])
                eps = None
                a2 = -a

                
        elif fit_eps == True:  
            if fit_a2 == False:
                data_exp = model.get_probs_by_exponent(a = a, a2 = 0, b = b, c = c, eps = eps)
                ll = metrics.log_loss(data_exp['stay'], data_exp['exponent_model'])
                
            elif fit_a2 == 'independent': 
                data_exp = model.get_probs_by_exponent(a = a, a2 = a2, b = b, c = c, eps = eps)
                ll = metrics.log_loss(data_exp['stay'], data_exp['exponent_model'])
            
            elif fit_a2 == 'pseudo':
                data_exp = model.get_probs_by_exponent(a = a, a2 = -a, b = b, c = c, eps = eps)
                ll = metrics.log_loss(data_exp['stay'], data_exp['exponent_model'])
                a2 = -a


        ############################
        ## Add class attributes: 
        self.name = name
        self.log_loss = round(ll, 10)
        self.params = {'a':a, 'a2': a2, 'b':b, 'c':c, 'eps':eps}
        self.reward_coding = model.reward_coding
        self.cmw_coding = model.cmw_coding
        self.initial_params = {'a':x0[0], 'a2':x0[1], 'b':x0[2], 'c':x0[3], 'eps':x0[4]} 
        self.fit_a2 = fit_a2
        self.fit_eps = fit_eps
        self.data = data_exp



####################################################################################
################# Additional functions for model training and test #################
####################################################################################
        

def save_model_results(model):
    now = datetime.datetime.now()
    time_stamp = now.strftime("%Y-%m-%d %H:%M")
    

    wb_append = load_workbook("model_outputs/model_params.xlsx")
    sheet = wb_append.active
    model_params = [time_stamp, 
                    model.name, 
                    model.name[:2],
                    model.params['a'], model.params['a2'], model.params['b'], model.params['c'], model.params['eps'], 
                    model.log_loss, 
                    (model.reward_coding), (model.cmw_coding), 
                    (model.fit_a2), (model.fit_eps), 
                    model.initial_params['a'], model.initial_params['a2'], model.initial_params['b'], model.initial_params['c'], model.initial_params['eps']]
    model_params = [str(item) for item in model_params]
    sheet.append(model_params)
    wb_append.save('model_outputs/model_params.xlsx')



def print_model_params(model):
    print ('###################')
    print('Model name:', model.name)
    print ('###################')
    print('Fitted params:', model.params)
    print ('reward coding:', model.reward_coding)
    print ('cmw coding:', model.cmw_coding)
    #print ('fit a2:', model.fit_a2)
    #print ('initial params:', model.initial_params)
    r = 5 if model.log_loss > 0.00001 else 10   
    print ('###################')
    print('Log loss:', round(model.log_loss, r))    
    print ('###################')
    
    if model.fit_a2 == False or model.fit_a2 == 'pseudo': print ('Warning: Check if boundaries were set to zero for the a2 parameter.')
    print ('Warning: Check if boundaries were set to zero for the eps parameter.') if model.fit_eps == False else print ('Warning: Check for setting a resonable upper bound on epsilon.')




class Cross_Validate(): 

    def __init__(self, data, FSS):
        self.data = data
        self.FSS = FSS
    
    def cross_validate(self):
        params = {'a': [], 'b': [], 'c': []}
        preds = pd.DataFrame({'monkey':[], 'session':[], 'trial_id':[], 'block_id':[], 'best_target':[], 'target':[], 'feedback':[], 'stay':[], 'exponent_model':[]})

        #####################
        ### Trim the dataset to be able to devide by 10 calculate window size 
        #####################

        res = self.data.shape[0]%10
        self.data = self.data.loc[res:, :]
        self.data.set_index(np.arange(1, self.data.shape[0]+1), drop = True, inplace = True)
        window = int(self.data.shape[0]/10)

        for i in range(10):

            #print ('fold num:', i)

            #####################
            ### Train-test split
            #####################
            test_fold = self.data.loc[(window*i)+1:window*(i+1), :]
            test_indices = test_fold.index.to_list()
            train_fold = self.data.loc[~self.data.index.isin(test_indices)]

            #####################
            ### Model train
            #####################
            
            cross_val_train = Optimize_exponent()
            cross_val_train.optimise_model(data = train_fold, name = 'cross_val_train',
                                           FSS = self.FSS, fit_a2 = False, fit_eps = False, 
                                           bounds = ((None, 2000), (0, 0), (-2, 0), (-100, 100), (0, 0)))

            params['a'].append(cross_val_train.params['a'])
            params['b'].append(cross_val_train.params['b'])
            params['c'].append(cross_val_train.params['c'])
            
            #####################
            ### Model pred on hold-out
            #####################

            model = Exponential_model(test_fold, FSS = self.FSS, fit_a2 = False, fit_eps = False, dropmissing = True)
            data_exp = model.get_probs_by_exponent(a = cross_val_train.params['a'], 
                                                   b = cross_val_train.params['b'],
                                                   c = cross_val_train.params['c'])
            data_exp = data_exp[['monkey', 'session', 'trial_id', 'block_id', 'best_target', 'target', 'feedback', 'stay', 'exponent_model']]
            preds = pd.concat([preds, data_exp])

        
        preds = preds
        ll = metrics.log_loss(preds['stay'], preds['exponent_model'])
        return (preds, params, ll)