#
# postprocess for fig. 11 and 12 - compute services ranking row by row, then compute quality as
# described into the paper
# prerequisite: datafiles have to be sorted by experiment and row
# early sub with second classified from previous windows
#
import csv
import copy
from curses import beep
from decimal import *
import decimal
from itertools import compress, islice
from math import fabs
from operator import truediv
import numpy as np
from pyparsing import null_debug_action
from scipy.stats import rankdata
from time import perf_counter_ns 

from numpy.core.arrayprint import _void_scalar_repr

def rank_file(master_filename, input_filename, output_filename):
    with open(master_filename, 'r') as f_master, open(output_filename, 'w', newline='') as f_out:
        # read a row and its next one from master file (excluding header) 
        # - take previous winner_idx and actual winner_idx
        # - for "terminated at iteractions" rows from detail file
        #  read in main file a row and compute ranking        
        previousWinner = 0
        # consider the first window anyway, otherwise we can't compare the 2 processes (which have a first window with 
        # different dimensions). Force previous winner to 1
        firstWindow = False
        firstIteration = True
        # manage multiple experiments in data files
        current_experiment = ""
        bNewExperiment = False      
        rows_to_skip_experiment = 0
        #        
        rows_to_skip = 0
        rows_to_read = 0
        # manage window from-to
        window_from = 1
        # assurance loss threshold: for now 10% - see also bComputeEarly
        AssuranceLossThreshold = 0.10
        # parameter to tell if using early sub or not
        bComputeEarly = True 
        reader_master = csv.DictReader(f_master,delimiter=';', quoting=csv.QUOTE_NONE)
        writer = csv.writer(f_out,delimiter=';', quoting=csv.QUOTE_NONE)
        writer.writerow(["Experiment", "Iteration", "wins_for_arm1", "wins_for_arm2", "wins_for_arm4", "wins_for_arm5", "ranking_arm1", "ranking_arm2", "ranking_arm4", "ranking_arm5", "Switched", "AssuranceCumulatedLoss","previousWinner", "willBeWinner", "window_from", "window_to", "assurance_willbe", "assurance_prev", "scores_master_0", "scores_master_1", "scores_master_2", "scores_master_3", "ranks_master_0", "ranks_master_1", "ranks_master_2", "ranks_master_3","elapsed_early"])
        for row in reader_master:
            if firstWindow == False:
                if row["experiment"] == "finito":
                    break
                #
                previous_experiment = current_experiment       
                current_experiment = row["experiment"]      
                # detail lines to read (window size)
                rows_to_read = int(row[" terminated at iteration"])                
                if previous_experiment != current_experiment:
                    bNewExperiment = True
                    firstIteration = True
                    rows_to_skip_experiment += rows_to_skip
                    rows_to_skip = 0
                    #
                    window_from = 1
                    window_to = int(row[" terminated at iteration"]) 
                else:
                    # 
                    window_from = window_to + 1
                    window_to += int(row[" terminated at iteration"])           

                # ranking to find the second place (it will be willbewinner)
                scores_master = [float(row[" wins for arm 1"]),float(row[" wins for arm 2"]),float(row[" wins for arm 4"]),float(row[" wins for arm 5"])] 
                # service ranking calculation
                # Rank the data, reverse the order (highest p_winner is lowest (=best) rank) 
                # and convert it to integer/list
                ranks_master = rankdata([-1 * score for score in scores_master],method='ordinal').astype(int).tolist()    
                # winner of the current window (will be compared with the winner of the previous window)
                if firstIteration == True:
                    firstIteration = False
                    # on the first iteration force willbewinner
                    willBeWinner = int(row["winner_idx"])                 
                    previousWinner = willBeWinner                    
                #
                assurance_cumulated = 0
                assurance_cumulated_loss = 0
                vs_willbe = 0
                assurance_willbe = 0
                assurance_cumulated = 0
                simulated_winner = 0
                assurance_simulated_winner = 0
                vs_prev = 0
                vstar_current_optimal_service = 0
                assurance_prev = 0
                switched = False
                # Initialize performance counter
                elapsed_time = 0
                
                # read detail data file
                with open(input_filename, 'r') as f_det: 
                # use the lines to skip (+ 1 since I added windows' dimensions)
                    for row_det in islice(csv.reader(f_det), rows_to_skip_experiment + rows_to_skip + 1, None):
                        # save timestamp for performances
                        start_slice = perf_counter_ns()                         
                        row_det = row_det[0].split(';')                        
                        # fields:
                        #experiment	iteration	pctile	p_winner 1	p_winner 2	p_winner 4	p_winner 5	est_wins 1	est_wins 2	est_wins 4	est_wins 5	real_winner_idx_prt	previous_real_winner_idx_prt	log_vs	log_value_rem	log_vstar	log_assurance_serviceUsed	winner_idx_prt	assurance_selected	residual abs	 vstar_thom	assurance_thom	residual_thom
                        #0	        1	        2	    3	        4	        5	        6	        7	        8	        9	        10	        11	                12	                            13	    14	            15	        16	                        17	            18	                19	             20	        21	            22                    
                        current_row = int(row_det[1])                           
                        current_experiment_det = row_det[0]                     
                        # management of multiple experiments: by adding the "terminated at iteration" I don't get the exact number of rows_to_skip
                        # since the last window is usually not complete. I look for the new experiment by scrolling
                        if (current_experiment != current_experiment_det):
                            rows_to_skip_experiment += 1
                            continue
                        # multiple experiments management
                        if (current_row  ) > (rows_to_read + rows_to_skip):
                            break
                        # winner of current iteration
                        local_winner = int(row_det[17])   
                        # if early sub was already activated, no longer calculate assurance
                        if (switched == False):
                            #assurance previous winner
                            if previousWinner == 1:
                                ind_est_Wins_previous_service = 7
                            elif previousWinner == 2:
                                ind_est_Wins_previous_service = 8
                            elif previousWinner == 4:      
                                ind_est_Wins_previous_service = 9                  
                            elif previousWinner == 5:
                                ind_est_Wins_previous_service = 10
                            else:
                                # something went wrong, should never get here. Throw catastrophic error
                                1/0
                            if local_winner == 1:
                                ind_est_Wins_local_winner_service = 7
                            elif local_winner == 2:
                                ind_est_Wins_local_winner_service = 8
                            elif local_winner == 4:      
                                ind_est_Wins_local_winner_service = 9                  
                            elif local_winner == 5:
                                ind_est_Wins_local_winner_service = 10
                            else:
                                # something went wrong, should never get here. Throw catastrophic error
                                1/0
                            vs_prev = float(row_det[ind_est_Wins_previous_service])
                            vstar_current_optimal_service = float(row_det[ind_est_Wins_local_winner_service])
                            assurance_prev = vs_prev / vstar_current_optimal_service
                            # assurance will be winner
                            if willBeWinner == 1:
                                ind_est_Wins_willbe_service = 7
                            elif willBeWinner == 2:
                                ind_est_Wins_willbe_service = 8
                            elif willBeWinner == 4:      
                                ind_est_Wins_willbe_service = 9                  
                            elif willBeWinner == 5:
                                ind_est_Wins_willbe_service = 10
                            else:
                                # something went wrong, should never get here. Throw catastrophic error
                                1/0                    
                            # 
                            vs_willbe = float(row_det[ind_est_Wins_willbe_service])
                            assurance_willbe = vs_willbe / vstar_current_optimal_service
                            assurance_cumulated += assurance_prev
                            if (bComputeEarly == True):
                                # don't use abs, if the difference is negative have a profit and use it to compensate for previous losses
                                # if early sub was activated, no longer cumulate and no longer compute assurance
                                if (switched == False):
                                    assurance_cumulated_loss += (assurance_willbe - assurance_prev)                              
                                    # compare to tle late assurance
                                    if ((assurance_cumulated_loss > (assurance_prev * AssuranceLossThreshold)) or switched == True):
                                        simulated_winner = willBeWinner
                                        assurance_simulated_winner = assurance_willbe
                                        switched = True

                                    else:
                                        simulated_winner = previousWinner
                                        assurance_simulated_winner = assurance_prev

                                    residual_prev = float(row_det[19])
                                    residual_willbe = abs(assurance_willbe - 1)
                                    # percentage of deviation
                                    if assurance_prev == 0:
                                        loss_perc = 0
                                    else:
                                        loss_perc = abs(assurance_cumulated_loss) * 100 / assurance_prev
                        # prepare output file
                        # input fields:
                        #experiment;Iteration; wins for arm 1; wins for arm 2; wins for arm 4; wins for arm 5;new_winner;previous_winner;early_subs                                
                        # wins list
                        scores = [float(row_det[7]),float(row_det[8]),float(row_det[9]),float(row_det[10])] 
                        # Rank services, reverse the order (highest est_win is lowest (=best) rank) 
                        # and convert it to integer/list
                        ranks = rankdata([-1 * score for score in scores],method='ordinal').astype(int).tolist()         
                        if switched == True:
                            # save old ranking of the new winner to give it to the old winner
                            old_ranking_winner = ranks.index(1)
                            # replace winner for early subs                            
                            if simulated_winner == 1:                                
                                old_ranking = ranks[0]
                                ranks[0] = 1
                            elif simulated_winner == 2:
                                old_ranking = ranks[1]
                                ranks[1] = 1
                            elif simulated_winner == 4:      
                                old_ranking = ranks[2]
                                ranks[2] = 1                
                            elif simulated_winner == 5:
                                old_ranking = ranks[3]
                                ranks[3] = 1
                            else:
                                # something went wrong, should never get here. Throw catastrophic error
                                1/0      
                            # assign to the old winner the ranking that the new winner had
                            ranks[old_ranking_winner] = old_ranking      
                        # save timestamp for performances
                        end_slice = perf_counter_ns() 
                        elapsed_time += (end_slice - start_slice)                        
                        # write ranking file
                        writer.writerow([current_experiment, current_row, row_det[7], row_det[8], row_det[9], row_det[10], ranks[0], ranks[1], ranks[2], ranks[3], switched, assurance_cumulated_loss, previousWinner, willBeWinner, window_from, window_to, assurance_willbe, assurance_prev, scores_master[0], scores_master[1], scores_master[2], scores_master[3], ranks_master[0], ranks_master[1], ranks_master[2], ranks_master[3], elapsed_time])
                # detail rows to be skipped to get to the window of interest in ranking file
                rows_to_skip += rows_to_read                  
            else:
                rows_to_skip = int(row[" terminated at iteration"])
            # winner of the previous window (will be used in the current one)
            previousWinner = int(row["winner_idx"])
            # winner for early sub, it's the second in the standings (+1 because starting from 0)
            willBeWinner = ranks_master.index(2) + 1
            # decode 3rd and 4th services
            if willBeWinner == 3:      
                willBeWinner = 4                  
            elif willBeWinner == 4:
                willBeWinner = 5            
            firstWindow = False

def winner(services_list):
    previous_ranking = 999
    previous_winner = 999
    for key in services_list:
        if services_list[key][0] < previous_ranking:
            previous_winner = key
            previous_ranking = services_list[key][0]
    return previous_winner    

def compute_quality_win(filename_0, filename_toCompare, output_filename):
    # set precision used for calculations
    getcontext().prec = 15
    # *0 are experiments without memory, *Tc are experiment To Compare, with memory
    current_experimentTc = ""
    current_experiment0 = ""
    current_windowFrom0 = 0
    current_windowFromTc = 0
    with open(filename_0, 'r') as f_main, open(filename_toCompare, 'r') as f_tc, open(output_filename, 'w', newline='') as f_out:            
        # read a row in main file and compute ranking
        reader0 = csv.DictReader(f_main,delimiter=';', quoting=csv.QUOTE_NONE)
        readerTC = csv.DictReader(f_tc,delimiter=';', quoting=csv.QUOTE_NONE)
        writer = csv.writer(f_out,delimiter=';', quoting=csv.QUOTE_NONE)
        writer.writerow(["Experiment", "Iteration", "ranking0_s1", "ranking0_s2", "ranking0_s4", "ranking0_s5", "rankingTc_s1", "rankingTc_s2", "rankingTc_s4", "rankingTc_s5", "winner0", "winnerTc", "r0", "tk", "residualError", "cumulativeError"])
        bEndTc = False
        cumulativeError = 0
        current_experiment0_num = 0
        current_experimentTc_num = 0
        # read a row in "to compare" file
        row_tc = next(readerTC)
        previous_experimentTc = current_experimentTc
        # input fields:
        current_experimentTc = row_tc["Experiment"]
        current_experimentTc = current_experimentTc[0:10]
        current_rowTc = int(row_tc["Iteration"])     
        previous_windowFromTc = current_windowFromTc
        current_windowFromTc = row_tc["window_from"]           
        for row0 in reader0:
            previous_experiment0 = current_experiment0
            # input fields:
            current_experiment0 = row0["Experiment"]
            current_experiment0 = current_experiment0[0:10]
            current_experiment0_num = int(current_experiment0[8:10])
            previous_windowFrom0 = current_windowFrom0
            current_windowFrom0 = row0["window_from"]    
            current_row0 = int(row0["Iteration"])
            ranking0_s1 = int(row0["ranking_arm1"])
            ranking0_s2 = int(row0["ranking_arm2"])
            ranking0_s4 = int(row0["ranking_arm4"])
            ranking0_s5 = int(row0["ranking_arm5"])
            # experiment change management
            if previous_experiment0 == "":
                previous_experiment0 = current_experiment0
                previous_windowFrom0 = current_windowFrom0      
            if current_experiment0_num != current_experimentTc_num:
                cumulativeError = 0
            if current_experiment0_num > current_experimentTc_num or (current_experiment0_num == current_experimentTc_num and current_rowTc < current_row0):
                # advance in file "to compare"
                while current_experiment0_num > current_experimentTc_num or (current_experiment0_num == current_experimentTc_num and current_rowTc < current_row0):
                    row_tc = next(readerTC,'end')
                    if row_tc == 'end': 
                        bEndTc = True
                        break
                    current_experimentTc = row_tc["Experiment"]
                    current_experimentTc = current_experimentTc[0:10]
                    current_experimentTc_num = int(current_experimentTc[8:10])
                    # ignore previous values since experiment has changed
                    previous_experimentTc = current_experimentTc 
                    current_rowTc = int(row_tc["Iteration"])
                    previous_windowFromTc = current_windowFromTc
                    current_windowFromTc = row_tc["window_from"] 
                    rankingTc_s1 = int(row_tc["ranking_arm1"])
                    rankingTc_s2 = int(row_tc["ranking_arm2"])
                    rankingTc_s4 = int(row_tc["ranking_arm4"])
                    rankingTc_s5 = int(row_tc["ranking_arm5"])                         
            if current_experimentTc_num > current_experiment0_num or (current_experimentTc_num == current_experiment0_num and current_row0 < current_rowTc):
                # advance in file "no memory"
                continue   
            if bEndTc == False and current_row0 == current_rowTc and current_experiment0 == current_experimentTc:
                # if any windows has changed, compute cumulative
                if previous_windowFromTc != current_windowFromTc or previous_windowFrom0 != current_windowFrom0:
                    # prepare to find winner
                    values0 = {
                        "1": [ranking0_s1],
                        "2": [ranking0_s2],
                        "4": [ranking0_s4],
                        "5": [ranking0_s5]
                    }
                    winner0 = winner(values0)
                    valuesTc = {
                        "1": [rankingTc_s1],
                        "2": [rankingTc_s2],
                        "4": [rankingTc_s4],
                        "5": [rankingTc_s5]
                    }
                    winnerTc = winner(valuesTc)                
                    # compute quality
                    if winner0 == winnerTc:
                        r0 = 0
                        tk = 0
                        residualError = 0
                    else:
                        # find ranking in mem 0 of di mem. x's winner              
                        r0 = Decimal(values0[winnerTc][0])             
                        tk = Decimal(1) / Decimal((len(values0) - 1))
                        # use different penalities
                        # v0: dynamically computed penalities
                        # v2: 0 0.4 0.85 1
                        # v3: 0 0.6 0.9 1     
                        # v4: 0, 0.1, 0.7, 1
                        # v5: 0, 0.2, 0.7, 1 -> definitiva per il prof
                        # v6: 0, 0, 1, 1
                        # v7: 0, 1, 1, 1
                        # v8: 0 0.33 0.66 1
                        # v9: 0 0.5 0.8 1
                        # pp 202201 residualError = Decimal((r0 - 1) * tk)
                        if r0 == 1:
                            # never gets there, coded just for clarity
                            residualError = 0
                        elif r0 == 2:
                            residualError = 0.2
                        elif r0 == 3:
                            residualError = 0.7
                        elif r0 == 4:
                            residualError = 1                                 
                    cumulativeError += residualError
                    writer.writerow([current_experiment0, current_row0, ranking0_s1, ranking0_s2, ranking0_s4, ranking0_s5, rankingTc_s1, rankingTc_s2, rankingTc_s4, rankingTc_s5, winner0, winnerTc, r0, tk, residualError, cumulativeError])
            else: 
                if current_row0 < current_rowTc:
                    continue
                else:
                    # something went wrong, should never get here. Throw catastrophic error
                    1/0

# processing starts here
#
tmp_file0Early ='Fig11-12Tmp_data0_rankingEarlySubW2_expAll_v5_0.txt'
input_file0 = 'values_remaining_expAll_0_v2_mod.csv'
master_file0 = "shift_detail_noreset_expAll_0_v2_mod.csv"
tmp_file5Early ='Fig11-12Tmp_data5_rankingEarlySubW2_expAll_v5.txt'
input_file5 = 'values_remaining_expAll_5_v2_mod.csv'
master_file5 = "shift_detail_noreset_expAll_5_v2_mod.csv"
out_file5_Win ='Fig11-12_data5_0vs5EsW2_v5.txt'
tmp_file10Early ='Fig11-12Tmp_data10_rankingEarlySubW2_expAll_v5.txt'
input_file10 = 'values_remaining_expAll_10_v2_mod.csv'
master_file10 = "shift_detail_noreset_expAll_10_v2_mod.csv"
out_file10_Win ='Fig11-12_data10_0vs10EsW2_expAll_v5.txt'
tmp_file25Early ='Fig11-12Tmp_data25_rankingEarlySubW2_expAll_v5.txt'
input_file25 = 'values_remaining_expAll_25_v2_mod.csv'
master_file25 = "shift_detail_noreset_expAll_25_v2_mod.csv"
out_file25_Win ='Fig11-12_data25_0vs25EsW2_expAll_v5.txt'
# for mem. 0 use tmp_file0 from PostProcessData_Fig10.py Fig10Tmp_data0_ranking_expAll_v5.txt which is without early
# sub since mem 0 is considered fixed withoud early sub
tmp_file0 = "Fig10Tmp_data0_ranking_expAll_v5.txt" 
rank_file(master_file0,input_file0,tmp_file0Early)
rank_file(master_file5,input_file5,tmp_file5Early)
rank_file(master_file10,input_file10,tmp_file10Early)
rank_file(master_file25,input_file25,tmp_file25Early)
#
compute_quality_win(tmp_file0, tmp_file5Early, out_file5_Win)
compute_quality_win(tmp_file0, tmp_file10Early, out_file10_Win)
compute_quality_win(tmp_file0, tmp_file25Early, out_file25_Win)
# 
