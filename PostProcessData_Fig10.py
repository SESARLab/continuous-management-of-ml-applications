#
# postprocess for fig. 10 - compute services ranking row by row, then compute quality as
# described into the paper
# prerequisite: datafiles have to be sorted by experiment and row
#
import csv
import copy
from curses import beep
from decimal import *
import decimal
from itertools import compress, islice
import numpy as np
from scipy.stats import rankdata

from numpy.core.arrayprint import _void_scalar_repr
           
def winner(services_list):
    # return a winner from a services list
    previous_ranking = 999
    previous_winner = 999
    for key in services_list:
        if services_list[key][0] < previous_ranking:
            previous_winner = key
            previous_ranking = services_list[key][0]
    return previous_winner    

def rank_file_v2(master_filename, input_filename, output_filename):
    with open(master_filename, 'r') as f_master, open(output_filename, 'w', newline='') as f_out:
        # read a row and its next one from master file (excluding header) 
        # - take previous winner_idx and actual winner_idx
        # - for "terminated at iteractions" rows from detail file
        firstWindow = False
        # at the first iteration initialize the previous winner with the current one, not having a previous one 
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
        reader_master = csv.DictReader(f_master,delimiter=';', quoting=csv.QUOTE_NONE)
        writer = csv.writer(f_out,delimiter=';', quoting=csv.QUOTE_NONE)
        writer.writerow(["Experiment", "Iteration", "wins_for_arm1", "wins_for_arm2", "wins_for_arm4", "wins_for_arm5", "ranking_arm1", "ranking_arm2", "ranking_arm4", "ranking_arm5", "window_from", "window_to", "scores_master_0", "scores_master_1", "scores_master_2", "scores_master_3", "ranks_master_0", "ranks_master_1", "ranks_master_2", "ranks_master_3"])
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

                # I'll do a ranking to find the second place (it will be willbewinner)
                scores_master = [float(row[" wins for arm 1"]),float(row[" wins for arm 2"]),float(row[" wins for arm 4"]),float(row[" wins for arm 5"])] 
                # service ranking calculation
                # Rank the data, reverse the order (highest p_winner is lowest (=best) rank) 
                # and convert it to integer/list
                ranks_master = rankdata([-1 * score for score in scores_master],method='ordinal').astype(int).tolist()    

                if firstIteration == True:
                    firstIteration = False             
                #
                # read detail data file
                with open(input_filename, 'r') as f_det: 
                # use the lines to skip (+ 1 since I added windows' dimensions )
                    for row_det in islice(csv.reader(f_det), rows_to_skip_experiment + rows_to_skip + 1, None):
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
                        # wins list
                        scores = [float(row_det[7]),float(row_det[8]),float(row_det[9]),float(row_det[10])] 
                        # Rank services, reverse the order (highest est_win is lowest (=best) rank) 
                        # and convert it to integer/list
                        ranks = rankdata([-1 * score for score in scores],method='ordinal').astype(int).tolist()         
                        # write ranking file
                        writer.writerow([current_experiment, current_row, row_det[7], row_det[8], row_det[9], row_det[10], ranks[0], ranks[1], ranks[2], ranks[3], window_from, window_to, scores_master[0], scores_master[1], scores_master[2], scores_master[3], ranks_master[0], ranks_master[1], ranks_master[2], ranks_master[3]])
                # detail rows to be skipped to get to the window of interest in ranking file
                rows_to_skip += rows_to_read                  
            else:
                rows_to_skip = int(row[" terminated at iteration"])
            firstWindow = False                    

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
                        # v5: 0, 0.2, 0.7, 1 -> definitive ones
                        # v6: 0, 0, 1, 1
                        # v7: 0, 1, 1, 1
                        # v8: 0 0.33 0.66 1
                        # v9: 0 0.5 0.8 1
                        # v0: residualError = Decimal((r0 - 1) * tk)
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
master_file0 = "shift_detail_noreset_expAll_0_v2_mod.csv"
master_file5 = "shift_detail_noreset_expAll_5_v2_mod.csv"
master_file10 = "shift_detail_noreset_expAll_10_v2_mod.csv"
master_file25 = "shift_detail_noreset_expAll_25_v2_mod.csv"
input_file0 = 'values_remaining_expAll_0_v2_mod.csv'
input_file5 = 'values_remaining_expAll_5_v2_mod.csv'
input_file10 = 'values_remaining_expAll_10_v2_mod.csv'
input_file25 = 'values_remaining_expAll_25_v2_mod.csv'
tmp_file0 ='Fig10Tmp_data0_ranking_expAll_v5.txt'
tmp_file5 ='Fig10Tmp_data5_ranking_expAll_v5.txt'
tmp_file10 ='Fig10Tmp_data10_ranking_expAll_v5.txt'
tmp_file25 ='Fig10Tmp_data25_ranking_expAll_v5.txt'
out_file5 ='Fig10_data5_NoEarlySub_expAll_v5.txt'
out_file10 ='Fig10_data10_NoEarlySub_expAll_v5.txt'
out_file25 ='Fig10_data25_NoEarlySub_expAll_v5.txt'
out_file5_Win = out_file5.replace('.txt','_windows.txt') 
out_file10_Win = out_file10.replace('.txt','_windows.txt') 
out_file25_Win = out_file25.replace('.txt','_windows.txt') 
rank_file_v2(master_file0,input_file0,tmp_file0)
rank_file_v2(master_file5,input_file5,tmp_file5)
rank_file_v2(master_file10,input_file10,tmp_file10)
rank_file_v2(master_file25,input_file25,tmp_file25)
#
compute_quality_win(tmp_file0, tmp_file5, out_file5_Win)
compute_quality_win(tmp_file0, tmp_file10, out_file10_Win)
compute_quality_win(tmp_file0, tmp_file25, out_file25_Win)
