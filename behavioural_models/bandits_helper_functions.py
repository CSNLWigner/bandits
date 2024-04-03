
import numpy as np
import pandas as pd


def get_shift_stay_variables(data): 

    '''Here I add a variable which tells if the monkey shifted its policy or stayed with 
    the previous choice. Its value is 0 if the monkey chose the same button as in the 
    previous trial, and its 1 if choses another. '''


    data.reset_index(inplace=True, drop = True)
    
    for i in range(1, data.shape[0]):
        data.loc[0, '_shift'] = np.nan

        prev_target = data.loc[i-1, 'target']
        target = data.loc[i, 'target']

        if prev_target == target: 
            data.loc[i, '_shift'] = 0
            data.loc[i, 'stay'] = 1
        else: 
            data.loc[i, '_shift'] = 1
            data.loc[i, 'stay'] = 0
            
    return (data)



def get_n_back(data, n_back = 8, reward_code = 'feedback'):
    
    '''
    Here I get the t-1, t-2, ... t-8 trials' feedback for each trial and create a dataframe from it. 
    n_back is the number of last n trials, which I want to weight in the regression.
    
    The reward_code makes possible to change the reward coding from 0,1 to -1,1. For this, first a new 
    column need to be added to the df, which stores the new reward history, and the column name shoudl be 
    passed to the "reward code" argument, such as 'feedback_11'). 
    '''
    
    n_back_feedback = {}
    for i in range(n_back, data.shape[0]):

        t_min1_8 = list(data.loc[i-n_back:i-1, reward_code])
        n_back_feedback[i] = t_min1_8

    n_back_feedback_data = pd.DataFrame(n_back_feedback).transpose()
    n_back_feedback_data = n_back_feedback_data.rename(columns = {7: 't-1',
                                                                  6: 't-2', 
                                                                  5: 't-3',
                                                                  4: 't-4',
                                                                  3: 't-5',
                                                                  2: 't-6',
                                                                  1: 't-7',
                                                                  0: 't-8'})
    cols = ['t-'+str(item) for item in range(1, 9)] ## this part is just to reorder the columns
    data = pd.concat([data, n_back_feedback_data[cols]], axis=1)
    return data





def get_binary_choice_match_value(data, coding = '01', n_back = 8): 
    
    '''
    Here I calculate the choice weight of each n-back trial. If the chosen target in t-n equals 
    with the chosen target in t, the weight is 1, otherwise the choice weight is 0.

    With coding '-11' one can code the matching targets with one, the non-matching ones with -1.  
    '''
    choice_weights = {}
    substract1 = np.ones((1, 8))

    for i in range(n_back, data.shape[0]):
        choice_t = data.loc[i, 'target']
        compare_array = np.full((1, 8), choice_t)
        choice_t_min1_8 = np.array(data.loc[i-n_back:i-1, 'target'])
        matching_choices = np.array(choice_t_min1_8) == compare_array
        matching_choices = matching_choices.astype(int)

        if coding == '01':

            pass
        
        elif coding == '-11':
            matching_choices = matching_choices*2
            matching_choices = matching_choices-substract1
            
        else:
            raise "Coding cannot be identified."

        choice_weights[i] = matching_choices[0]

    choice_weights_data = pd.DataFrame(choice_weights).transpose()
    choice_weights_data = choice_weights_data.rename(columns = {7: 'CMW_t-1',
                                                                6: 'CMW_t-2', 
                                                                5: 'CMW_t-3',
                                                                4: 'CMW_t-4',
                                                                3: 'CMW_t-5',
                                                                2: 'CMW_t-6',
                                                                1: 'CMW_t-7',
                                                                0: 'CMW_t-8'})
    data = pd.concat([data, choice_weights_data], axis=1)
    return data







###### Note: to open the original pickle pandas 1.4.4. is needed. 
# I read the file with pandas 1.4.4 and then load save it, than upgraded my pandas again. 

# file = open('behavior.pkl', 'rb')
# data = pickle.load(file)
## pip install --upgrade pandas==1.4.4
#data.to_excel(path + 'behavioural_data.xlsx')