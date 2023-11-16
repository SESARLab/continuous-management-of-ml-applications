import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# utility filesystem
import pathlib
# load e dump for models
from joblib import dump, load
# to load dictionary
import pickle
# to show cpu time
from time import perf_counter_ns 
# 
import csv
# 
import math 
import os

class Arm(object):
    """
    Each arm's success in predictiong bails is modeled by a beta distribution
    """
    def __init__(self, idx, model, a=1, b=1):
        """
        Init with uniform prior
        """
        self.idx = idx
        self.a = a
        self.b = b
        self.model = model
        # initial baseline setting
        self.a_baseline = 0
        self.b_baseline = 0
        
    def record_success(self):
        self.a += 1
        
    def record_failure(self):
        self.b += 1

    def reset_ab(self, perc_keep):
        self.a = math.floor(self.a * perc_keep)
        self.b = math.floor(self.b * perc_keep)
        # alpha and beta must be at least set to 1 to have beta distribution work at first iteration
        if self.a == 0:
            self.a = 1
        if self.b == 0:
            self.b = 1     
        self.a_baseline = self.a
        self.b_baseline = self.b   


    def draw_model(self):
        """
        Draw samples from a Beta distribution
        """
        return np.random.beta(self.a, self.b, 1)[0]
    
    def mean(self):
        return self.a / (self.a + self.b)

def load_dataset(filename):
    """
    Load a dataset file into a dataframe

    Parameters:
    filename: csv file's full path

    Returns:
    dataset: dataframe ready for processing
    """
    # load csv file
    names = ['DOWNLOAD DATE', 'IDENTIFIER', 'LATEST ADMISSION DATE', 'RACE', 'GENDER','AGE','BOND AMOUNT','OFFENSE',
        'FACILITY', 'DETAINER']
    dataset = pd.read_csv(filename,sep=',', names=names, header=0)
    # prepare dataset for processing - removing not significant columns
    # remove not significant column DOWNLOAD DATE
    dataset = dataset.drop(columns="DOWNLOAD DATE")
    # remove not significant column
    dataset = dataset.drop(columns="IDENTIFIER")
    # remove not significant column BOND AMOUNT, it's the one to be predicted
    dataset = dataset.drop(columns="BOND AMOUNT")
    # remove missing values
    dataset = dataset.dropna(axis=0)
    # prepare date column for processing, better performances with int64 datatype
    dataset['LATEST ADMISSION DATE'] = pd.to_datetime(dataset['LATEST ADMISSION DATE'], format="%m/%d/%Y")
    dataset['LATEST ADMISSION DATE'] = dataset['LATEST ADMISSION DATE'].astype(np.int64) // 10**9
    # categorical fields 
    dataset = pd.get_dummies(dataset, columns=['RACE'], prefix = ['RACE'])
    dataset = pd.get_dummies(dataset, columns=['GENDER'], prefix = ['GENDER'])
    # remove spaces in test fields
    dataset.OFFENSE = dataset.OFFENSE.replace('\s+', '', regex=True)
    dataset.FACILITY = dataset.FACILITY.replace('\s+', '', regex=True)
    dataset.DETAINER = dataset.DETAINER.replace('\s+', '', regex=True)
    # categorical fields
    dataset['OFFENSE'] = dataset['OFFENSE'].astype('category')
    dataset['FACILITY'] = dataset['FACILITY'].astype('category')
    dataset['DETAINER'] = dataset['DETAINER'].astype('category')
    # load dictionary for categorical fields
    with open('le_modelv_v2.pkl', 'rb') as f2:
        model_dic= pickle.load(f2)
    # list fields to do Label encoding
    col_list =  ['OFFENSE', 'FACILITY', 'DETAINER']
    for col in col_list:
        encoder = model_dic[col]
        try:
            dataset[col]=encoder.transform(dataset[col].astype(str))
        except:
            dataset[col]=encoder.transform(dataset[col].astype(int))      

    return dataset  

def fair_prediction(row, model):
    """
    make prediction on the row with the model passed as parameter
    Generate a set of row with combined protected groups to be tested with fairness

    Parameters:
    row:  dataframe with a single row
    model: model to be used in prediction

    Returns:
    variance: variance of the predictions
    """    
    # make prediction of the real data and then generate other
    # protected groups data 
    # seek for gender in the sample
    for col1 in [col for col in row.columns if 'GENDER' in col]:
        if row[col1].values[0] == 1:
            col_grp1_setted = col1
    # seek for race in the sample
    for col1 in [col for col in row.columns if 'RACE' in col]:
        if row[col1].values[0] == 1:
            col_grp2_setted = col1   
    # prepare and set dataframe with data from sample. it'll contain generated data
    dataframe_row = row
    row = row.iloc[0:0]
    row = row.append(dataframe_row).reset_index(drop=True)
    # generate test data for all protected groups (gender and race)
    for col1 in [col for col in row.columns if 'GENDER' in col]:
        for col2 in [col for col in row.columns if 'RACE' in col]:
            if not ((col1 == col_grp1_setted) and (col2 == col_grp2_setted)):
                # fetch dataframe column and set the relevant ones
                for col in [col for col in row.columns if 'GENDER' in col]:
                    if col == col1:
                        dataframe_row[col] = 1
                    else:
                        dataframe_row[col] = 0
                for col in [col for col in row.columns if 'RACE' in col]:
                    if col == col2:
                        dataframe_row[col] = 1
                    else:
                        dataframe_row[col] = 0     
                row = row.append(dataframe_row).reset_index(drop=True)
    Xnew = row.values
    # make a prediction
    ynew = model.predict(Xnew)    
    # return the variance of the prediction on data
    return ynew.var()
        
def monte_carlo_simulation(arms, draw=100):
    """
    Monte Carlo simulation of thetas. Each arm's result
    follows a beta distribution.
    
    Parameters
    ----------
    arms list[Arm]: list of Arm objects.
    draw int: number of draws in Monte Carlo simulation
    
    Returns
    -------
    mc np.matrix: Monte Carlo matrix of dimension (draw, n_arms)
    p_winner list[float]: probability of each arm being the winner
    """
    # Monte Carlo sampling
    alphas = [arm.a for arm in arms]
    betas = [arm.b for arm in arms]
    mc = np.matrix(np.random.beta(alphas, betas, size=[draw, len(arms)]))
    # count frequency of each arm being winner 
    counts = [0 for _ in arms]
    winner_idxs = np.asarray(mc.argmax(axis=1)).reshape(draw,)
    for idx in winner_idxs:
        counts[idx] += 1
    # divide by draw to approximate probability distribution
    p_winner = [count / draw for count in counts]
    return mc, p_winner

def thompson_sampling(arms):
    """
    Stochastic sampling: take one draw for each arm
    divert predictions to best draw
    
    @param arms list[Arm]: list of Arm objects

    Returns
    -------
    idx int: index of winning arm from sample
    sample_p[idx]: beta distribution value
    """
    sample_p = [arm.draw_model() for arm in arms]
    idx = np.argmax(sample_p)
    return idx, sample_p[idx]

def should_terminate(experiment, p_winner, est_wins, mc, iteration, writervalue, real_winner_idx, previous_real_winner_idx, alpha=0.05):
    """
    Decide whether experiument should terminate. When value remaining in
    experiment is less than 1% of the winning arm's number of wins
    
    Parameters
    ----------
    experiment: current experiment for log purpose
    p_winner list[float]: probability of each arm being the winner.
    est_wins list[float]: estimated number of wins (from real performances)
    mc np.matrix: Monte Carlo matrix of dimension (draw, n_arms)
    iteration: current iteration (for logging purposes)
    writervalue: file writer (for logging purposes)
    real_winner_idx: actual (current iteration's) winner
    previous_real_winner_idx: previous windows' winner
    alpha: controlling for type I error
    
    Returns
    -------
    bool: True if experiment should term   inate.
    """
    winner_idx = np.argmax(p_winner)
    values_remaining = (mc.max(axis=1) - mc[:, winner_idx]) / mc[:, winner_idx]
    # logging: compute paper's "vs" ("reward from effective") for optimal service (the one of the previous window)
    log_vs = est_wins[previous_real_winner_idx]
    # logging: compute paper's reward from Monte Carlo ("v*")
    log_vstar = est_wins[winner_idx]
    # logging: compute paper's assurance for optimal service (the one of the previous window)
    log_assurance_ServiceUsed = log_vs / log_vstar
    # compute reward for Thompson sampling
    log_vstar_Thommy = est_wins[real_winner_idx]
    log_assurance_ServiceThommy = log_vs / log_vstar_Thommy
    # save value for statistics
    if real_winner_idx >= 2:
        real_winner_idx_prt = real_winner_idx + 1
    else:
        real_winner_idx_prt = real_winner_idx
    # add 1 to winner idx for plots
    real_winner_idx_prt += 1    
    if previous_real_winner_idx >= 2:
        previous_real_winner_idx_prt = previous_real_winner_idx + 1
    else:
        previous_real_winner_idx_prt = previous_real_winner_idx
    previous_real_winner_idx_prt += 1    
    if winner_idx >= 2:
        winner_idx_prt = winner_idx + 1      
    else:
        winner_idx_prt = winner_idx  
    # add 1 to winner idx for plots
    winner_idx_prt += 1     
    pctile = np.percentile(values_remaining, q=100 * (1 - alpha))
    # logging: compute paper's ws 
    log_value_rem = 1 - pctile
    # real_winner_idx_prt is the winner from Thompson sampling (the one experimented)
    # previous_real_winner_idx_prt is the arm which has won in previous windows
    # winner_idx_prt is Monte Carlo winner
    # residual abs: difference between previous windows chosen service's assurance and
    # assurance of the serivce chosen for every trace (which is always 1 since it's the defined as 
    # “ration between selected service s reward and the optimal Monte Carlo simulated one of DWMAB-M” 
    # (which is a ration between 2 identical values since the service is alwyas the same)
    #           experiment	        iteration	            pctile	        p_winner 1-2-4-5	est_wins 1-2-4-5	            real_winner_idx_prt	            previous_real_winner_idx_prt	            log_vs	            log_value_rem	            log_vstar	            log_assurance_serviceUsed	            winner_idx_prt assurance_selected	    residual abs	                        vstar_thom	                        assurance_thom	                            residual_thom	
    stringa = (experiment + ";" + str(iteration) + ";"  + str(pctile)+  ";"  + str(p_winner)+ ";"  + str(est_wins) + ";"  + str(real_winner_idx_prt) + ";"  + str(previous_real_winner_idx_prt) + ";"  + str(log_vs) + ";"  + str(log_value_rem) + ";"  + str(log_vstar) + ";"  + str(log_assurance_ServiceUsed) + ";"  + str(winner_idx_prt)+ ";1;"  + str(abs(log_assurance_ServiceUsed - 1)) + ";"  + str(log_vstar_Thommy) + ";"  + str(log_assurance_ServiceThommy) + ";"  +  str(abs(log_assurance_ServiceThommy - 1)))
    stringa = str(stringa).replace(",",";")    
    writervalue.writerow([stringa])    
    return pctile < 0.01 * est_wins[winner_idx]

def k_arm_bandit(experiment, models, inputdata, exp_memory, alpha=0.05, burn_in=1000, max_iter=100000, draw=100, silent=False):
    """
    Perform stochastic k-arm bandit test. Experiment is terminated when
    value remained in experiment drops below certain threshold
    
    Parameters 
    ----------
    experiment: current experiment (datafile name)
    models list[]: models to be tested.
    inputdata dataframe: data to be used for test
    exp_memory: memory to be kept while experimenting
    alpha float: terminate experiment when the (1 - alpha)th percentile
        of the remaining value is less than 1% of the winner's winnings
    burn_in int: minimum number of iterations
    max_iter int: maxinum number of iterations
    draw int: number of rows in Monte Carlo simulation
    silent bool: print status at the end of experiment
    
    """
    # vairance threshold under which the arm wins
    VARIANCE = 200
    n_arms = len(models)
    arms = [Arm(idx=i, model=models[i]) for i in range(n_arms)]
    history_p = [[] for _ in range(n_arms)]
    # variable used to balance draws number since we initialize alpha and beta to 1 each one
    subtract = 2
  
    start_slice = perf_counter_ns() 
    iteration = 0
    global_iteration = 0
    # initialize winner_idx
    winner_idx = 0   
    previous_winner_idx = 0
    # iterate through input data
    for index, row in inputdata.iterrows():
        iteration = iteration + 1
        global_iteration = global_iteration + 1
        # row object is a series, need to convert into dataframe and transpose
        row = row.to_frame().T
        # thompson sampling to choose the model (arm)
        idx, beta_dist_value = thompson_sampling(arms) 
        # save thompson sampling's decision
        stringa = (experiment + ";" + str(global_iteration) + ";"  + str(idx))
        stringa = str(stringa).replace(",",";")
        writerThom.writerow([stringa]) 
        # save beta dist. value
        stringa = (experiment + ";" + str(global_iteration) + ";"  + str(beta_dist_value))
        stringa = str(stringa).replace(",",";")
        writerBeta.writerow([stringa])
        # get te model to be evaluated
        arm, model = arms[idx], models[idx]
        #use model to make a prediction
        var = fair_prediction(row, model)       
        # update arm's beta parameters, win if variance < variance's threshold
        if var < VARIANCE:
            arm.record_success()
        else:
            arm.record_failure()
        # record current estimates of each arm being winner
        mc, p_winner = monte_carlo_simulation(arms, draw)
        for j, p in enumerate(p_winner):
            # storing est_models and p_winner
            history_p[j].append(p)
        # record current estimates of each arm's wins
        est_models = [arm.mean() for arm in arms]
        # save est_models value
        stringa = (experiment + ";" + str(global_iteration) + ";"  + str(est_models))
        stringa = str(stringa).replace(",",";")
        writerEstmod.writerow([stringa])        
        # terminate when value remaining is negligible
        draws = [arm.a + arm.b - 2 for arm in arms]
        # 
        if should_terminate(experiment, p_winner, est_models, mc,global_iteration, writerValuesRem, idx, previous_winner_idx, alpha) and iteration >= burn_in:
            # select winner's idx
            idx = np.where(est_models == np.amax(est_models)) 

            draws = [arm.a + arm.b - arm.b_baseline - arm.a_baseline - subtract for arm in arms]
            # save alphas, betas and baselines
            alphas = [arm.a for arm in arms]
            betas = [arm.b for arm in arms]
            alphas_baseline = [arm.a_baseline for arm in arms]
            betas_baseline = [arm.b_baseline for arm in arms]
            # variable used to balance draws number: after first iteration no need to balance
            subtract = 0
            winner_idx = idx[0]       
            
            # if winner idx >= 2 add 1  (for output purposes since we removed model which had id 2 fairmlmodel_NB_v2_03)
            if winner_idx.size > 1:
                winner_idx = winner_idx[0]     
                previous_winner_idx = winner_idx 
            else:    
                # save previous winner idx for logging
                previous_winner_idx = winner_idx[0]                  
            if winner_idx >= 2:
                winner_idx += 1

            # add 1 to winner idx for plots
            winner_idx += 1
            print( str(global_iteration) +" winner_idx1:", winner_idx, est_models, " draws: ", draws, "Terminated at iteration %i"%(iteration))
            stringa = ("experiment:", str(experiment),"winner_idx:", str(winner_idx), est_models, " draws: ", str(draws), "Terminated at iteration %i"%(iteration), "CPU time: ", perf_counter_ns()  - start_slice, " alphas: ", str(alphas), " draws: ", str(draws), " betas: ", str(betas), " alphas_baseline: ", str(alphas_baseline), " betas_baseline: ", str(betas_baseline), "Global iteration %i"%(global_iteration), "p_winner:", p_winner)
            stringa = str(stringa).replace(",",";")
            writer.writerow([stringa]) 
            #  keep previous alpha and beta values for arms, just reset hisory. Instead of destroying, keep percentage
            [arm.reset_ab(exp_memory) for arm in arms] # memory
            # reset iteration for run
            iteration = 0

            history_p = [[] for _ in range(n_arms)]         
            # set time for performances
            start_slice = perf_counter_ns()       

    return 

# processing starts here
# memory value
memory = 0.25
# save CPU time
start = perf_counter_ns() 

# get working directory
locapath = str(pathlib.Path().absolute())
# load models
models = []
models.append(load(locapath + '\\models\\fairmlmodel_NB_v2_01.joblib') )
models.append(load(locapath + '\\models\\fairmlmodel_NB_v2_02.joblib') )
# models.append(load(locapath + '\\models\\fairmlmodel_NB_v2_03.joblib') )  removed for low performances cfr. table 1
models.append(load(locapath + '\\models\\fairmlmodel_NB_v2_04.joblib') )
models.append(load(locapath + '\\models\\fairmlmodel_NB_v2_05.joblib') )
# open csv files for logging
# experiment used - for logging filenames
strExperimentSortedOrNot = 'expAll_'
outcsv = open('shift_detail_noreset_' + strExperimentSortedOrNot +  str(int(memory * 100)) + '_v2.csv','w')
writer = csv.writer(outcsv, delimiter =";")
outcsvThom = open('thommy_detail_noreset_' + strExperimentSortedOrNot +  str(int(memory * 100)) + '_v2.csv','w')
writerThom = csv.writer(outcsvThom, delimiter =";")
# logging beta dist
outcsvBeta = open('beta_detail_noreset_' + strExperimentSortedOrNot + str(int(memory * 100)) + '_v2.csv','w')
writerBeta = csv.writer(outcsvBeta, delimiter =";")
# logging est_models
outcsvEstmod = open('est_models_detail_noreset_' + strExperimentSortedOrNot + str(int(memory * 100)) + '_v2.csv','w')
writerEstmod = csv.writer(outcsvEstmod, delimiter =";")
# logging value remaining 
outcsvValuesRem = open('values_remaining_' + strExperimentSortedOrNot + str(int(memory * 100)) + '_v2.csv','w')
writerValuesRem = csv.writer(outcsvValuesRem, delimiter =";")   
# load test data
experiments = ['TestSet_01.csv','TestSet_02.csv','TestSet_03.csv','TestSet_04.csv','TestSet_05.csv',
'TestSet_06.csv','TestSet_07.csv','TestSet_08.csv','TestSet_09.csv','TestSet_10.csv']


for experiment in experiments:
    data_for_test = load_dataset(locapath + "\\data\\" + experiment)
    print(experiment)    
    # k armed bandit
    k_arm_bandit(experiment, models, data_for_test, memory, alpha=0.05, burn_in=100 ) 
    # display results
    print("CPU time for processing:", perf_counter_ns()  - start, "nanoseconds")
writer.writerow(['finito'])
outcsv.close()
outcsvThom.close()
outcsvBeta.close()
outcsvEstmod.close()
outcsvValuesRem.close()

