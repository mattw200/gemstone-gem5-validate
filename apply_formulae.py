#!/usr/bin/env python

# Matthew J. Walker
# Created: 31 July 2017

# Framework for analysing gem5 stats
# This script creates new stats by applying formulae to existing stats

def find_stats_per_group(df):
    import numpy as np
    unique_masks = df['xu3 stat core mask'].unique()
    print("Unique masks:"+str(unique_masks))

    unique_freq_combinations = np.unique(df[['xu3 stat Freq (MHz) C0','xu3 stat Freq (MHz) C4']].values)
    print("Unique freqs:"+str(unique_freq_combinations))
    
    unique_presets = df['gem5 stat workloads preset'].unique()
    print(unique_presets)

    mibench_workloads = [ df['xu3 stat workload name'].iloc[x] for x in range(0, len(df.index)) if df['gem5 stat workloads preset'].iloc[x].find('mibench') > -1 and df['gem5 stat workloads preset'].iloc[x].find('par') < 0 ]
    print ("mibench: "+str(mibench_workloads))

    lmbench_workloads = [ df['xu3 stat workload name'].iloc[x] for x in range(0, len(df.index)) if df['gem5 stat workloads preset'].iloc[x].find('lmbench') > -1 ]
    print("lmbench: "+str(lmbench_workloads))

    parsec_1_workloads = [ df['xu3 stat workload name'].iloc[x] for x in range(0, len(df.index)) if df['gem5 stat workloads preset'].iloc[x].find('parsec1') > -1 ]
    print("parsec_1: "+str(parsec_1_workloads))

    parsec_4_workloads = [ df['xu3 stat workload name'].iloc[x] for x in range(0, len(df.index)) if df['gem5 stat workloads preset'].iloc[x].find('parsec4') > -1 ]
    print("parsec_4: "+str(parsec_4_workloads))

    misc_workloads = [ df['xu3 stat workload name'].iloc[x] for x in range(0, len(df.index)) if df['gem5 stat workloads preset'].iloc[x].find('misc') > -1 ]
    print("misc: "+str(misc_workloads))

    parmibench_workloads = [ df['xu3 stat workload name'].iloc[x] for x in range(0, len(df.index)) if df['gem5 stat workloads preset'].iloc[x].find('parmibench') > -1 ]
    print("parmibench: "+str(parmibench_workloads))
    
    roy_workloads = [ df['xu3 stat workload name'].iloc[x] for x in range(0, len(df.index)) if df['gem5 stat workloads preset'].iloc[x].find('roy') > -1 ]
    print("roy: "+str(roy_workloads))

    # TODO make a function that gets data and error for groups and subgroups. 
    # have an 'analyse data' script that:
    # applies the formulae,
    # calculest the error of groups
    # allows plotting
    # does the error correlation. 

def convert_column_headings(stats_df):
    import pandas as pd
    stats_df.columns = [ x.replace('-','_').replace(' ','_') for x in stats_df.columns.values]
   
def compare_two_stats(stats_df, stat_A_name, stat_B_name):
    stats_df['comp. '+stat_A_name+' vs. '+stat_B_name+' (%)'] = \
            (((stats_df[stat_A_name] - stats_df[stat_B_name]).abs()) / stats_df[stat_A_name]) * 100.0

def apply_new_stat(df, stat_name, equation):
    import pandas as pd
    #df.eval(stat_name+' = '+equation,inplace=True)
    print df['xu3 stat duration (s)']
    #df['new'] = pd.eval("df['xu3 stat duration (s)'] * 1000.0" )
    df[stat_name] = pd.eval(equation,engine='python')

def apply_formulae(df, formulae_file_path):
    import pandas as pd
    equations_df = pd.read_csv(formulae_file_path, sep='\t')
    print("Applying the following equations:")
    print equations_df
    for i in range(0, len(equations_df.index)):
        print("Applying equation "+str(i)+": "+str(equations_df['Equation'].iloc[i]))
        apply_new_stat(df, equations_df['Stat Name'].iloc[i], equations_df['Equation'].iloc[i])
    return df

if __name__=='__main__':
    import argparse
    import os
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  dest='input_file_path', required=True, \
               help="The stats df on which to apply the formulae")
    args=parser.parse_args()
    
    df = pd.read_csv(args.input_file_path,sep='\t')
    old_cols = df.columns.values.tolist()
    apply_formulae(df,'gem5-stats.equations')
    
    important_cols = ['xu3 stat workload name','xu3 stat core mask','xu3 stat duration mean (s)','xu3 stat duration SD (s)','xu3 stat duration (s)', 'xu3 stat Freq (MHz) C0','xu3 stat Freq (MHz) C4', 'gem5 stat model name',	'gem5 stat workloads preset',	'gem5 stat workload name',	'gem5 stat core mask',	'gem5 stat A7 Freq (MHz)',	'gem5 stat A15 Freq (MHz)',	'gem5 stat m5out directory','gem5 stat sim_seconds']
    
    new_cols_only = [x for x in df.columns.values if x not in old_cols]
    condensed_df = df[important_cols + new_cols_only]
    print df[new_cols_only]
    condensed_df.to_csv('condensed.csv',sep='\t')
    find_stats_per_group(df)
    
