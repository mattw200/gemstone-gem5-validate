#!/usr/bin/env python

# Matthew J. Walker
# Created: 31 July 2017

# Framework for analysing gem5 stats
# This script creates new stats by applying formulae to existing stats

important_cols = ['xu3 stat workload name', 'xu3 stat iteration index', 'xu3 stat core mask','xu3 stat duration mean (s)','xu3 stat duration SD (s)','xu3 stat duration (s)', 'xu3 stat Freq (MHz) C0','xu3 stat Freq (MHz) C4', 'gem5 stat model name',	'gem5 stat workloads preset',	'gem5 stat workload name',	'gem5 stat core mask',	'gem5 stat A7 Freq (MHz)',	'gem5 stat A15 Freq (MHz)',	'gem5 stat m5out directory','gem5 stat sim_seconds']

def test_function(numA=1,numB=2):
    return numA+numB

# not used:
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
    #df['new'] = pd.eval("df['xu3 stat duration (s)'] * 1000.0" )
    print ("Adding new stat: "+stat_name+" with formula: "+equation)
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

def create_xu3_cluster_average(df):
   # find cols to average
   # TODO: filter so that it only looks at the columns where the A7/A15 is being used (i.e. filter
   # on the core mask). Then raise an exception if it finds more than four columns for the same PMC
   # A7:
    a7_pmcs = [ x[x.find('(0')+1: x.find('(0')+5] for x in df.columns.values \
             if x.find("cntr") > -1 and x.find('CPU 0') > -1 and x.find('diff') > -1 ]
    # A15
    a15_pmcs = [ x[x.find('(0')+1: x.find('(0')+5] for x in df.columns.values \
             if x.find("cntr") > -1 and x.find('CPU 4') > -1 and x.find('diff') > -1 ]
    if not all(d == 1 for d in  [a7_pmcs.count(x) for x in a7_pmcs]):
        raise ValueError("Duplicate pmcs for a7_pmcs: "+str(a7_pmcs))
    if not all(d == 1 for d in  [a15_pmcs.count(x) for x in a15_pmcs]):
        raise ValueError("Duplicate pmcs for a15_pmcs: "+str(a15_pmcs))

    # add cycle count average
    a7_cycle_counts = [x for x in df.columns.values if x.find('cycle count') > -1 and (x.find('CPU 0') > -1 or x.find('CPU 1') > -1 or x.find('CPU 2') > -1 or x.find('CPU 3') > -1) ]
    if len(a7_cycle_counts) != 8:
        raise ValueError("len of a7_cycle_counts is not 8! a7_cycle_counts: "+str(a7_cycle_counts))
    a15_cycle_counts = [x for x in df.columns.values if x.find('cycle count') > -1 and (x.find('CPU 4') > -1 or x.find('CPU 5') > -1 or x.find('CPU 6') > -1 or x.find('CPU 7') > -1) ]
    if len(a15_cycle_counts) != 8:
        raise ValueError("len of a15_cycle_counts is not 8! a15_cycle_counts: "+str(a15_cycle_counts))

    df['xu3new a7 cycle count total diff'] = df[[x for x in a7_cycle_counts if x.find('diff') > -1]].sum(axis=1)
    df['xu3new a7 cycle count avg rate'] = df[[x for x in a7_cycle_counts if x.find('rate') > -1]].mean(axis=1)
    df['xu3new a15 cycle count total diff'] = df[[x for x in a15_cycle_counts if x.find('diff') > -1]].sum(axis=1)
    df['xu3new a15 cycle count avg rate'] = df[[x for x in a15_cycle_counts if x.find('rate') > -1]].mean(axis=1)
    

    for pmc in a7_pmcs:
        cols_to_avg = [x for x in df.columns.values if (x.find('CPU 0') > -1 or x.find('CPU 1') > -1 or x.find('CPU 2') > -1 or x.find('CPU 3') > -1) and x.find('('+pmc+')') > -1  ]
        df['xu3new a7 '+pmc+' total diff'] = df[[x for x in cols_to_avg if x.find('diff') > -1]].sum(axis=1)
        df['xu3new a7 '+pmc+' avg rate'] = df[[x for x in cols_to_avg if x.find('rate') > -1]].mean(axis=1)

    for pmc in a15_pmcs:
       cols_to_avg = [x for x in df.columns.values if (x.find('CPU 4') > -1 or x.find('CPU 5') > -1 or x.find('CPU 6') > -1 or x.find('CPU 7') > -1) and x.find('('+pmc+')') > -1 ]
       df['xu3new a15 '+pmc+' total diff'] = df[[x for x in cols_to_avg if x.find('diff') > -1]].sum(axis=1)
       df['xu3new a15 '+pmc+' avg rate'] = df[[x for x in cols_to_avg if x.find('rate') > -1]].mean(axis=1)

if __name__=='__main__':
    import argparse
    import os
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  dest='input_file_path', required=True, \
               help="The stats df on which to apply the formulae")
    args=parser.parse_args()
    df = pd.read_csv(args.input_file_path,sep='\t')
    create_xu3_cluster_average(df)
    old_cols = df.columns.values.tolist()
    apply_formulae(df,'gem5-stats.equations')
    # NOTE: removing workloads that don't appear 6 times!
    workloads_to_remove = [x for x in df['xu3 stat workload name'].unique() if df['xu3 stat workload name'].value_counts()[x] != 6]
    print ("Workloads to remove:"+str(workloads_to_remove))
    df = df[~df['xu3 stat workload name'].isin(workloads_to_remove)]
    df.to_csv(args.input_file_path+'-applied_formulae.csv',sep='\t')
    new_cols_only = [x for x in df.columns.values if x not in old_cols]
    condensed_df = df[important_cols + new_cols_only]
    print df[important_cols + new_cols_only]
    condensed_df.to_csv(args.input_file_path+'-applied_formulae.csv'+'-condensed.csv',sep='\t')
    #find_stats_per_group(df)
    #print df[important_cols + ['gem5new clock tick diff A15'] +  ['gem5new A15 cycle count diff total'] +  ['gem5new A15 active cycles per cycle'] + ['xu3gem5 A15 cycle count total signed err'] +  ['xu3gemt A15 cycle count no idle total signed err']]

    # print average abs and signed errors:
    workloads_to_error = [x for x in df['xu3 stat workload name'].unique().tolist() if x.find('rl-') == -1]
    err_df = df[df['xu3 stat workload name'].isin(workloads_to_error)]
    print err_df['xu3 stat workload name'].tolist()
    print "No. unique workloads: "+str(len(err_df['xu3 stat workload name'].unique()))
    parsec_wls = [x for x in err_df['xu3 stat workload name'].unique() if x.find('parsec') > -1]
    print ("Parsec workloads: "+str(parsec_wls))
    print("ALL PARSEC abs: "+str(err_df[err_df['xu3 stat workload name'].isin(parsec_wls)]['xu3gem5 duration pc err'].mean()))
    print("ALL abs: "+str(err_df['xu3gem5 duration pc err'].mean()))
    print("ALL signed: "+str(err_df['xu3gem5 duration signed err'].mean()))
    print("A15 abs: "+str(err_df[err_df['xu3 stat core mask'] == '4,5,6,7']['xu3gem5 duration pc err'].mean()))
    print("A15 signed: "+str(err_df[err_df['xu3 stat core mask'] == '4,5,6,7']['xu3gem5 duration signed err'].mean()))
    print("A7 abs: "+str(err_df[err_df['xu3 stat core mask'] == '0,1,2,3']['xu3gem5 duration pc err'].mean()))
    print("A7 signed: "+str(err_df[err_df['xu3 stat core mask'] == '0,1,2,3']['xu3gem5 duration signed err'].mean()))

    print("All errors: "+str(len(err_df.index)))
    print("Positive errors: "+str(len(err_df[err_df['xu3gem5 duration signed err'] >= 0])))
    over_100_MAPE = err_df[err_df['xu3gem5 duration pc err'] > 100.0]['xu3 stat workload name'].unique().tolist()
    print("Over 100 errors: "+str(over_100_MAPE))
    err_df[err_df['xu3 stat workload name'].isin(over_100_MAPE)][['xu3 stat workload name', 'xu3 stat core mask', 'xu3 stat Freq (MHz) C4','xu3gem5 duration pc err','xu3gem5 duration signed err']].to_csv(args.input_file_path+'-over-100-mape.csv')

    # find missing  workloads:
    unique_wls = df['xu3 stat workload name'].unique()
    for wl in unique_wls:
        print(wl+": "+str(df['xu3 stat workload name'].value_counts()[wl]))

    df[df['xu3 stat workload name'] == 'dhrystone'].to_csv(args.input_file_path+'-dhrystone-temp.csv')

    chickentikka

   
    # collect stats of A15
    temp1_df = df[df['xu3 stat core mask'] == '4,5,6,7']
    signed_err_cols = [x for x in temp1_df.columns.values if x.find('signed err') > -1]
    temp1_df = temp1_df[important_cols + signed_err_cols]
    print signed_err_cols
    cols_signed_and_abs = []
    for s in signed_err_cols:
        temp1_df[s.replace('signed', 'ABS')] = temp1_df[s].abs() 
        cols_signed_and_abs.append(s)
        cols_signed_and_abs.append(s.replace('signed','ABS'))
    temp1_df = temp1_df[important_cols + cols_signed_and_abs]
    temp1_df.to_csv(args.input_file_path+'-temp1_df.csv',sep='\t')
    #signed
    temp1_signed_mean_df = temp1_df[signed_err_cols].mean()
    temp1_signed_mean_df.to_csv(args.input_file_path+'-temp1_signed_df.csv',sep='\t')
    # ABS
    temp1_abs_mean_df = temp1_df[[x for x in cols_signed_and_abs if x not in signed_err_cols]].mean()
    temp1_abs_mean_df.to_csv(args.input_file_path+'-temp1_abs_df.csv',sep='\t')
  
    # IPC and rates (e.g. L1D hit rate etc.)
    temp2_df = df[df['xu3 stat core mask'] == '4,5,6,7']
    rate_cols = [x for x in temp2_df.columns.values if x.find('rate') > -1 or x.find('ratio') > -1 or x.find('IPC') > -1 ]
    temp2_df = temp2_df[important_cols + rate_cols]
    temp2_df.to_csv(args.input_file_path+'-temp2_df.csv',sep='\t')
    temp2_mean_df = temp2_df[rate_cols].mean()
    temp2_mean_df.to_csv(args.input_file_path+'-temp2_mean.csv',sep='\t')

    # correlation analysis
    df_a15 = df[df['xu3 stat core mask'] == '4,5,6,7']
    duration_signed_col = 'xu3gem5 duration signed err'
    xu3_pmcs_diff_cols = [x for x in df_a15.columns.values if x.find('avg') > -1 and x.find('a15') > -1 and x.find('rate') > -1]
    gem5_pmcs_diff_cols = [x for x in df_a15.columns.values if x.find('gem5 stat') > -1 and x.find('bigCluster') > -1]
    print ("\n\nxu3 pmcs: "+str(xu3_pmcs_diff_cols))
    from scipy.stats.stats import pearsonr
    import numpy as np
    print(df_a15[xu3_pmcs_diff_cols].apply(np.mean))
    print(df_a15[xu3_pmcs_diff_cols].apply(lambda x: np.mean(x)))
    correlation_xu3_df = df_a15[xu3_pmcs_diff_cols].apply(lambda x: pearsonr(x,df_a15[duration_signed_col])[0])
    correlation_gem5_df = df_a15[gem5_pmcs_diff_cols].apply(lambda x: pearsonr(x,df_a15[duration_signed_col])[0])
    print (correlation_xu3_df)
    correlation_xu3_df.to_csv(args.input_file_path+'-correlation_xu3_pmcs.csv',sep='\t')
    correlation_gem5_df.to_csv(args.input_file_path+'correlation_gem5_pmcs.csv',sep='\t')
