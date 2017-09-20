#!/usr/bin/env python

# Matthew J. Walker
# Created: 2 Sep 2017

# Deals with PMC names and finding gem5 equivalents
import numpy as np
import pandas as pd
import math
import compare
import apply_formulae

def get_lovely_pmc_name(existing_name, hw_cluster_id):
    hex_val = existing_name.find('0x')
    if hex_val < 0:
        return 'not a pmc'
        #raise ValueError("Can't find the PMC hex value")
    hex_val = '0x'+existing_name[hex_val+2:hex_val+4].upper()
    print("Hex val: "+str(hex_val))
    pmcs_df = get_pmcs_df(hw_cluster_id)
    pmcs_df = pmcs_df[pmcs_df['PMC_ID'] == hex_val]
    if len(pmcs_df.index) < 1:
        return "NA ("+hex_val+")"
        #raise ValueError("Can't find PMC_ID: "+hex_val)
    return pmcs_df['NAME'].iloc[0] + ':'+pmcs_df['PMC_ID'].iloc[0]
    

def get_new_stat(df, equation):
    import pandas as pd
    print ("Getting new stat: with formula: "+equation)
    return pd.eval(equation,engine='python')

def get_pmcs_df(hw_cluster_id):
    if hw_cluster_id == 'a15':
        return pd.read_csv('pmcs-and-gem5-stats.equations',sep='\t') 
    else:
        raise ValueError("Not yet implementent clusters other than a15")

def process_gem5_equiv_stat(df, result_name, gem5_eq):
    # 1. ensure no cpus0, cpus1 etc. - should be cpuX
    gem5_eq = gem5_eq.replace('cpus0','cpusX').replace('cpus1','cpusX').replace('cpus2','cpusX').replace('cpus3','cpusX')
    # 2. prepend 'gem5 stat '
    gem5_eq = gem5_eq.replace("df['","df['gem5 stat ").replace('df["','df["gem5 stat ')
    # 3. determine if it is a 'cpu' stat
    if gem5_eq.find('cpusX') > -1:
        # repeat four times with the different cpus
        gem5_eq = gem5_eq.replace('cpusX','cpus0')+' + '+gem5_eq.replace('cpusX','cpus1')+' + ' \
                +gem5_eq.replace('cpusX','cpus2')+' + '+gem5_eq.replace('cpusX','cpus3')
    print("The processed equation: "+gem5_eq)
    new_df = pd.DataFrame()
    new_df[result_name] = get_new_stat(df, gem5_eq)
    return new_df

def create_pmc_err_df(df, hw_cluster_id, gem5_cluster_id, hw_duration_col, gem5_duration_col, use_last_if_dup=False):
    pmcs_df = get_pmcs_df(hw_cluster_id)
    # create column of integer pmc_ids
    pmcs_df['PMC_ID'] = pmcs_df['PMC_ID'].fillna('-0x01')
    pmcs_df['gem5 equivalent'] = pmcs_df['gem5 equivalent'].fillna('None')
    pmcs_df['INT_PMC_ID'] = pmcs_df['PMC_ID'].apply(lambda x:  int(x,0))
    # get pmcs, order, and filter out negative:
    ordered_pmc_ints = sorted(list(set([x for x in pmcs_df['INT_PMC_ID'] if x >= 0])))
    print(ordered_pmc_ints)
    result_df = df[apply_formulae.important_cols+['workload A15 clusters']] # hack include clusters
    for pmc_int in ordered_pmc_ints:
        temp_pmcs_df = pmcs_df[pmcs_df['INT_PMC_ID'] == pmc_int]
        print temp_pmcs_df
        if len(temp_pmcs_df) > 1:
            print("Found multiple entries for the same PMC: "+str(pmc_int)+" ("+str(hex(pmc_int))+")")
            if not use_last_if_dup: 
                raise ValueError("ERROR: Found multiple entries for the same PMC: "+str(pmc_int)+" ("+str(hex(pmc_int))+")")
            else:
                # use the last one entered
                print("Using last value...")
                temp_pmcs_df = temp_pmcs_df[temp_pmcs_df.index == temp_pmcs_df.index[-1]]
        print temp_pmcs_df
        # skip if there is no gem5 equation for this stat
        if temp_pmcs_df['gem5 equivalent'].iloc[0] == 'None':
            continue
        # using diff!
        hw_cols = [x for x in df.columns.values.tolist() if x.find(hw_cluster_id) > -1 and x.find('total diff') > -1]
        selected_col = [x for x in hw_cols if x.find(temp_pmcs_df['PMC_ID'].iloc[0]) > -1]
        if len(selected_col) != 1:
            raise ValueError("Length of selected_col should be 1!: "+str(selected_col))
        print selected_col
        # now process gem5 column
        name_of_hw_col = 'HW '+hw_cluster_id+' PMC total '+str(temp_pmcs_df['PMC_ID'].iloc[0])+':'+str(temp_pmcs_df['NAME'].iloc[0])
        result_df[name_of_hw_col] = df[selected_col]
        name_of_gem5_col = 'gem5 '+hw_cluster_id+' PMC total '+str(temp_pmcs_df['PMC_ID'].iloc[0])+':'+str(temp_pmcs_df['NAME'].iloc[0])
        result_df[name_of_gem5_col] =  process_gem5_equiv_stat(df, name_of_gem5_col, temp_pmcs_df['gem5 equivalent'].iloc[0])
        result_df['PMC total MAPE '+str(temp_pmcs_df['PMC_ID'].iloc[0])+':'+str(temp_pmcs_df["NAME"].iloc[0])] = \
                    compare.mape(result_df[name_of_hw_col],result_df[name_of_gem5_col])
        result_df['PMC total MPE '+temp_pmcs_df['PMC_ID'].iloc[0]+':'+temp_pmcs_df["NAME"].iloc[0]] = \
                    compare.mpe(result_df[name_of_hw_col],result_df[name_of_gem5_col])
        result_df['PMC total SMPE '+temp_pmcs_df['PMC_ID'].iloc[0]+':'+temp_pmcs_df["NAME"].iloc[0]] = \
                    compare.smpe(result_df[name_of_hw_col],result_df[name_of_gem5_col])
        name_of_hw_rate_col = name_of_hw_col.replace('PMC total', 'PMC cluster rate')
        name_of_gem5_rate_col = name_of_gem5_col.replace('PMC total','PMC cluster rate')
        result_df[name_of_hw_rate_col] = result_df[name_of_hw_col]/df[hw_duration_col]
        result_df[name_of_gem5_rate_col] = result_df[name_of_gem5_col]/df[gem5_duration_col]
        result_df['PMC cluster rate MAPE '+temp_pmcs_df['PMC_ID'].iloc[0]+':'+temp_pmcs_df["NAME"].iloc[0]] = \
                    compare.mape(result_df[name_of_hw_rate_col],result_df[name_of_gem5_rate_col])
        result_df['PMC cluster rate MPE '+temp_pmcs_df['PMC_ID'].iloc[0]+':'+temp_pmcs_df["NAME"].iloc[0]] = \
                    compare.mpe(result_df[name_of_hw_rate_col],result_df[name_of_gem5_rate_col])
        result_df['PMC cluster rate SMPE '+temp_pmcs_df['PMC_ID'].iloc[0]+':'+temp_pmcs_df["NAME"].iloc[0]] = \
                    compare.smpe(result_df[name_of_hw_rate_col],result_df[name_of_gem5_rate_col])
        # new: find normalised values
        result_df['Normalised hw '+temp_pmcs_df['NAME'].iloc[0]+':'+temp_pmcs_df['PMC_ID'].iloc[0]] = \
                    result_df[name_of_hw_col]/result_df[name_of_hw_col] 
        result_df['Normalised gem5 '+temp_pmcs_df['NAME'].iloc[0]+':'+temp_pmcs_df['PMC_ID'].iloc[0]] = \
                    result_df[name_of_gem5_col]/result_df[name_of_hw_col] 
    return  result_df
