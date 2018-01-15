#!/usr/bin/env python

# Matthew J. Walker
# Created: 31 July 2017

# Framework for analysing gem5 stats
# This script creates new stats by applying formulae to existing stats

import pmcs_and_gem5_stats

important_cols = ['xu3 stat workload name', 'xu3 stat iteration index', 'xu3 stat core mask','xu3 stat duration mean (s)','xu3 stat duration SD (s)','xu3 stat duration (s)', 'xu3 stat Freq (MHz) C0','xu3 stat Freq (MHz) C4', 'gem5 stat model name',	'gem5 stat workloads preset',	'gem5 stat workload name',	'gem5 stat core mask',	'gem5 stat A7 Freq (MHz)',	'gem5 stat A15 Freq (MHz)',	'gem5 stat m5out directory','gem5 stat sim_seconds']

workload_name_dict = {
        'parsec-blackholes-4' : 'parsec-blackscholes-4',
        'parsec-blackholes-1' : 'parsec-blackscholes-1',
         'basicmath' : 'mi-basicmath',
         'bitcount-small' : 'mi-bitcount-small',
         'qsort' : 'mi-qsort',
         'susan-smoothing' : 'mi-susan-smoothing',
         'susan-edges' : 'mi-susan-edges',
         'susan-corners' : 'mi-susan-corners',
         'jpeg-encode' : 'mi-jpeg-encode',
         'jpeg-decode' : 'mi-jpeg-decode',
         'typeset' : 'mi-typeset',
         'dijkstra' : 'mi-dijkstra',
         'patricia' : 'mi-patricia',
         'stringsearch' : 'mi-stringsearch',
         'blowfish-encode-small' : 'mi-blowfish-encode-small',
         'sha' : 'mi-sha',
         'adpcm-encode-small' : 'mi-adpcm-encode-small',
         'adpmc-decode-small' : 'mi-adpmc-decode-small',
         'crc32-small' : 'mi-crc32-small',
         'fft' : 'mi-fft',
         'inv-fft' : 'mi-inv-fft',
         'gsm-encode-small' : 'mi-gsm-encode-small'
}

def get_converted_workload_names(df, new_workload_col):
    convert_dict = {
        'backholes' : 'blackscholes'
    }
    # there are two workload column names:
    xu3
    list_of_workloads = df[workload_col].tolist()
    list_of_workloads = df[workload_col]
    df[new_workload_col] = df


def mape(actual, predicted):
    return ((actual - predicted)/actual).abs()*100.0

def signed_err(actual,predicted):
    return ((actual - predicted)/actual)*100.0

def wape(actual, predicted):
    return  (((actual - predicted).abs()).sum() / (actual.sum()))*100.0

def noramlise(df, value, value_col):
    return (value - df[value_col].min()) / (df[value_col].max() - df[value_col].min())


def make_gem5_cols_per_cluster(df, cluster_name,create_rates=False):
    # e.g. cluster_name = 'bigCluster'
    gem5_stat_cols_no_cpu = [x for x in df.columns.values if (x.find('gem5 stat ') > -1 and x.find('system.') > -1 and x.find('system.bigCluster.cpus') == -1 and x.find('system.littleCluster') == -1)]
    new_g5_corr_df = df[gem5_stat_cols_no_cpu]
    gem5_stat_cols_cpu = [x for x in df.columns.values if x.find('system.'+cluster_name+'.cpu') > -1]
    for cpu_stat in gem5_stat_cols_cpu:
        new_col_name = cpu_stat.replace('.cpus0.','.cpusX.').replace('.cpus1.','.cpusX.').replace('.cpus2.','.cpusX.').replace('.cpus3.','.cpusX.')
        new_col_val = 0
        key_error = 0
        if new_col_name in new_g5_corr_df.columns.values.tolist():
            continue # skip if done already
        try:
            new_col_val += df[new_col_name.replace('.cpusX.','.cpus0.')]
        except KeyError:
            key_error += 1
        try:
            new_col_val += df[new_col_name.replace('.cpusX.','.cpus1.')]
        except KeyError:
            key_error += 1
        try:
            new_col_val += df[new_col_name.replace('.cpusX.','.cpus2.')]
        except KeyError:
            key_error += 1
        try:
            new_col_val += df[new_col_name.replace('.cpusX.','.cpus3.')]
        except KeyError:
            key_error += 1
        if key_error > 3:
            raise KeyError("Failed at life")
        new_g5_corr_df[new_col_name] = new_col_val
    if create_rates:
        for col in new_g5_corr_df.columns.values:
            new_g5_corr_df['rate '+col] = new_g5_corr_df[col]/df['gem5 stat sim_seconds']
    return new_g5_corr_df.fillna(0)


def cluster_analysis(data, labels, level, filepath,show_plot=False):
    import scipy
    import matplotlib as mpl
    import scipy.cluster.hierarchy as sch
    import matplotlib.pyplot as plt
    import numpy as np
    D = scipy.spatial.distance.pdist(data, 'correlation')
    #print("PRINTING D")
    #np.set_printoptions(threshold=np.nan)
    #print D
    Y = scipy.cluster.hierarchy.linkage(D, method='single')
    d = scipy.cluster.hierarchy.dendrogram(Y, color_threshold=level, labels=labels)	
    if show_plot:
        plt.show()
    #plt.close()
    plt.savefig(filepath)
    plt.close()
    ind = sch.fcluster(Y, level, 'distance')
    #table = [labels, nice_event_names.get_names(labels), ind]
    table = [labels, ind]
    clusters_df = pd.DataFrame(np.transpose(table), columns=['Labels', 'Cluster_ID'])
    return clusters_df


def build_model(df, core_mask, freq_C0, freq_C4, y_col,var_select_func,num_inputs,filepath_prefix):
    #import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import math
    temp_df = df[(df['xu3 stat core mask'] == core_mask) & (df['xu3 stat Freq (MHz) C0'] == freq_C0) & (df['xu3 stat Freq (MHz) C4'] == freq_C4)]
    # remove gem5 stats and re-inster them 'per cluster
    gem5_cluster = 'bigCluster'
    if core_mask == '4,5,6,7':
        gem5_cluster = 'bigCluster'
    elif core_mask == '0,1,2,3':
        gem5_cluster = 'littleCluster'
    else:
        raise ValueError("Unrecognised core mask!")
    gem5_per_cluster  = make_gem5_cols_per_cluster(temp_df, gem5_cluster)
    temp_df = temp_df[[x for x in temp_df.columns.values if x.find('gem5 stat') == -1]]
    temp_df = pd.concat([temp_df, gem5_per_cluster], axis=1)
    temp_df =  temp_df._get_numeric_data().fillna(0)
    temp_df = temp_df.fillna(0)
    temp_df = temp_df.loc[:, (temp_df != 0).any(axis=0)] # remove 0 col
    temp_df = temp_df.loc[:, (temp_df != temp_df.ix[0]).any()] 
    temp_df = temp_df[[x for x in temp_df.columns.values.tolist() if not 0 in temp_df[x].tolist() ]]
    temp_df['const'] = 1
    # get var names:
    var_names = var_select_func(temp_df)
    model_inputs = []
    #model_inputs.append('const')
    models = []
    for i in range(0, num_inputs):
        best_r2 = 0
        best_var = ''
        best_model_res = 0
        var_names = [x for x in var_names if x in temp_df.columns.values.tolist() and x != y_col]
        
        for var in var_names:
            dep_vars = model_inputs + [var]
            print ("Trying with these vars: "+str(dep_vars))
            #formula = ''+y_col+' ~ '+' + '.join(["Q('"+x+"')" for x in dep_vars])+' '
            #print(formula)
            #mod = smf.ols(formula=formula,data=temp_df)
            X = temp_df[dep_vars]
            y = temp_df[y_col]
            #X = sm.add_constant(X) # use const
            res = sm.OLS(y,X).fit()
            r2 = res.rsquared 
            print res.summary()
            if r2 > best_r2:
                best_r2 = r2
                best_var = var
                best_model_res = res
        model_inputs.append(best_var)
        models.append(best_model_res)
    model_summary_df = pd.DataFrame(columns=['number_of_events', 'R2', 'adjR2', 'WAPE'])
    for i in range(0, len(models)):
        model = models[i]
        print "\nMODEL"
        print ("r2: "+str(model.rsquared))
        print ("params: "+(str(model.params)))
        print ("Predict:")
        print model.predict()
        print ("MAPE: "+str(mape(temp_df[y_col],model.predict()).mean()))
        print ("MAPE: ")
        print mape(temp_df[y_col], model.predict())
        print ("Actual: ")
        print temp_df[y_col]
        print "Predicted:"
        print model.predict()
        print "WAPE:"
        print wape(temp_df[y_col],model.predict())
        print model.summary()
        model_summary_df = model_summary_df.append({
                'number_of_events' : i,
                'R2' : model.rsquared,
                'adjR2' : model.rsquared_adj,
                'WAPE' :  wape(temp_df[y_col],model.predict()),
                'SER' : math.sqrt(model.scale)
                },ignore_index=True)
        #params_df = pd.concat([model.params, model.pvalues], axis=1)
        params_df = pd.DataFrame(columns=['Name', 'Value', 'p-Value'])
        params_df['Name'] = model.params.index
        params_df['Value'] = model.params.tolist()
        params_df['p-Value'] = model.pvalues.tolist()
        params_df['pretty name'] = params_df['Name'].apply(lambda x: pmcs_and_gem5_stats.get_lovely_pmc_name(x,'a15')+' (total)' if x.find('total diff') > -1  else pmcs_and_gem5_stats.get_lovely_pmc_name(x,'a15')+' (rate)')
        params_df.to_csv(filepath_prefix+'-model-'+str(i)+'.csv',sep='\t')
        #model.params.append(model.pvalues).to_csv(filepath_prefix+'-model-'+str(i)+'.csv',sep='\t')
    model_summary_df.to_csv(filepath_prefix+'-model-summary'+'.csv',sep='\t')
                

def workload_clustering(df, core_mask, freq_C0,freq_C4, graph_out_prefix_path):
    import numpy as np
    temp_df = df[(df['xu3 stat core mask'] == core_mask) & (df['xu3 stat Freq (MHz) C0'] == freq_C0) & (df['xu3 stat Freq (MHz) C4'] == freq_C4)]
    # cluster using xu3 diffs and rates
    xu3_cluster_id = 'a15'
    if core_mask == '0,1,2,3':
        xu3_cluster_id = 'a7'
    elif core_mask == '4,5,6,7':
        pass
    else:
        raise ValueError("Unrecognised core mask")
    #wl_cluster_df = temp_df[[x for x in temp_df.columns.values.tolist() if x.find('xu3new') > -1 and (x.find('total diff') > -1 or x.find('avg rate') > -1) and x.find(xu3_cluster_id) > -1]]
    # only cluster workloads with rates
    wl_cluster_df = temp_df[[x for x in temp_df.columns.values.tolist() if x.find('xu3new') > -1 and (x.find('avg rate') > -1) and x.find(xu3_cluster_id) > -1]]
    wl_cluster_df = wl_cluster_df.fillna(0)
    wl_cluster_df = wl_cluster_df.loc[:, (wl_cluster_df != 0).any(axis=0)] # remove 0 col
    wl_cluster_df = wl_cluster_df.loc[:, (wl_cluster_df != wl_cluster_df.iloc[0]).any()] 
    wl_cluster_df = wl_cluster_df[[x for x in wl_cluster_df.columns.values.tolist() if not 0 in wl_cluster_df[x].tolist() ]]
    data = wl_cluster_df.values
    levels_list = [ 0.012, 0.007]
    clusters_dfs = []
    for i in range(0, len(levels_list)):
       clusters_dfs.append(cluster_analysis((data), temp_df['xu3 stat workload name'].tolist(), levels_list[i], graph_out_prefix_path+'-plot-dendrogram-wls-'+str(i)+'.pdf',show_plot=False))
    wl_clusters_df = pd.DataFrame({'wl name':temp_df['xu3 stat workload name'].tolist()} )
    for i in range(0,len(levels_list)):
        wl_clusters_df['cluster '+str(i)] = clusters_dfs[i]['Cluster_ID']
        #wl_clusters_df['workloads '+str(i)] = clusters_dfs[i]['Labels']
        wl_clusters_df['stat name '+str(i)] = clusters_dfs[i]['Labels']
        print clusters_dfs[i]
    return wl_clusters_df

def corr_and_cluster_analysis(df, core_mask, freq_C0, freq_C4, cor_y, graph_out_prefix_path,corr_only=False):
    import numpy as np
    # cluster analysis
    temp_df = df[(df['xu3 stat core mask'] == core_mask) & (df['xu3 stat Freq (MHz) C0'] == freq_C0) & (df['xu3 stat Freq (MHz) C4'] == freq_C4)]
    # get get gem5 per-cluster stats:
    gem5_sum_cluster_df = 0
    xu3_cluster_id = 'a15'
    if core_mask == '0,1,2,3':
        gem5_sum_cluster_df = make_gem5_cols_per_cluster(temp_df, 'littleCluster')
        xu3_cluster_id = 'a7'
    elif core_mask == '4,5,6,7':
        gem5_sum_cluster_df = make_gem5_cols_per_cluster(temp_df, 'bigCluster')
    else:
        raise ValueError("Unrecognised core mask")
    xu3_sum_cluster_df = temp_df[[x for x in temp_df.columns.values.tolist() if x.find('xu3new') > -1 and (x.find('total diff') > -1 or x.find('avg rate') > -1) and x.find(xu3_cluster_id) > -1]]
    print('XU3 headings: '+str(xu3_sum_cluster_df.columns.values.tolist()))
    combined_df = pd.concat([xu3_sum_cluster_df,gem5_sum_cluster_df], axis=1)._get_numeric_data()
    combined_df = combined_df.fillna(0)
    combined_df = combined_df.loc[:, (combined_df != 0).any(axis=0)] # remove 0 col
    combined_df = combined_df.loc[:, (combined_df != combined_df.iloc[0]).any()] 
    combined_df = combined_df[[x for x in combined_df.columns.values.tolist() if not 0 in combined_df[x].tolist() ]]
    #combined_df = combined_df[[x for x in combined_df.columns.values.tolist() if combined_df[x].mean() > 5000 ]]
    #combined_df = combined_df[[x for x in combined_df.columns.values.tolist() if all(i >= 500 for i in combined_df[x].tolist())  ]]
    #combined_df = combined_df[[x for x in combined_df.columns.values.tolist() if all(i >= 1000000000 for i in combined_df[x].tolist())  ]]
    #combined_df.to_csv('debug-combined_df.csv', sep='\t')
    #print combined_df
    # filter for xu3 rate only:
    #combined_df = combined_df[[x for x in combined_df.columns.values if x.find('avg rate') > -1]]
    #combined_df = combined_df[[x for x in combined_df.columns.values if x.find('avg rate') > -1]]
    combined_df = combined_df[[x for x in combined_df.columns.values if x.find('gem5 stat') > -1]]
    data = combined_df.values
    levels_list = []
    if not corr_only:
        levels_list = [ 0.18, 0.1]
        clusters_dfs = []
        for i in range(0, len(levels_list)):
            clusters_dfs.append(cluster_analysis(np.transpose(data), combined_df.columns.values.tolist(), levels_list[i], graph_out_prefix_path+'-plot-dendrogram-pmcs-'+str(i)+'.pdf'))
    # now do correlation analysis and combine with cluster info into one DF
    from scipy.stats.stats import pearsonr
    correlation_combined_df = combined_df.apply(lambda x: pearsonr(x,temp_df[cor_y])[0])
    corr_and_cluster_df = pd.DataFrame({'stat name':correlation_combined_df.index, 'correlation':correlation_combined_df.values})
    for i in range(0,len(levels_list)):
        corr_and_cluster_df['cluster '+str(i)+' ('+str(levels_list[i])+')'] = clusters_dfs[i]['Cluster_ID']
    print corr_and_cluster_df
    corr_and_cluster_df['neat pmc names'] = corr_and_cluster_df['stat name'].apply(lambda x: pmcs_and_gem5_stats.get_lovely_pmc_name(x,'a15'))
    #corr_and_cluster_df = corr_and_cluster_df[corr_and_cluster_df['neat pmc names'] != 'not a pmc']
    corr_and_cluster_df.to_csv(graph_out_prefix_path+'-corr-and-cluster.csv-gem5-only.csv',sep='\t')
    # now make for forr > 0.3
    corr_and_cluster_df[(corr_and_cluster_df['correlation'] > 0.3) | (corr_and_cluster_df['correlation'] < -0.3)].to_csv(graph_out_prefix_path+'-corr-and-cluster-ovr-30.csv',sep='\t')
    corr_and_cluster_df[(corr_and_cluster_df['correlation'] > 0.25) | (corr_and_cluster_df['correlation'] < -0.25)].to_csv(graph_out_prefix_path+'-corr-and-cluster-ovr-25.csv',sep='\t')


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

    # remove roy from all!!!
    #df = df[[x for x in df['xu3 stat workload name'] if x.find('rl-') == -1 ]]
    df = df[df['xu3 stat workload name'].str.contains('rl-') == False]
    workload_names = df['xu3 stat workload name'].tolist()
    for i in range(0, len(workload_names)):
        for  key in workload_name_dict:
            print("New key: "+str(key)+" (i="+str(i)+")")
            if workload_names[i] == key:
                print("Key: "+str(key))
                print("i: "+str(i))
                workload_names[i] = workload_name_dict[key]
                print("Done")
    workload_names = [x.replace('-small','') for x in workload_names]
    print("here")
    df['xu3 stat workload name'] = workload_names
    df['gem5 stat workload name'] = workload_names

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

    # do freq breakdown:
    print("A15 1800: "+str(err_df[(err_df['xu3 stat Freq (MHz) C4'] == 1800) & (err_df['xu3 stat core mask'] == '4,5,6,7')]['xu3gem5 duration signed err'].mean()))
    print("A15 1000: "+str(err_df[(err_df['xu3 stat Freq (MHz) C4'] == 1000) & (err_df['xu3 stat core mask'] == '4,5,6,7')]['xu3gem5 duration signed err'].mean()))
    print("A15 600: "+str(err_df[(err_df['xu3 stat Freq (MHz) C4'] == 600) & (err_df['xu3 stat core mask'] == '4,5,6,7')]['xu3gem5 duration signed err'].mean()))
    print("A7 1400: "+str(err_df[(err_df['xu3 stat Freq (MHz) C0'] == 1400) & (err_df['xu3 stat core mask'] == '0,1,2,3')]['xu3gem5 duration signed err'].mean()))
    print("A7 1000: "+str(err_df[(err_df['xu3 stat Freq (MHz) C0'] == 1000) & (err_df['xu3 stat core mask'] == '0,1,2,3')]['xu3gem5 duration signed err'].mean()))
    print("A7 600: "+str(err_df[(err_df['xu3 stat Freq (MHz) C0'] == 600) & (err_df['xu3 stat core mask'] == '0,1,2,3')]['xu3gem5 duration signed err'].mean()))
    # df[(df['col1'] >= 1) & (df['col1'] <=1 )]

    print("All errors: "+str(len(err_df.index)))
    print("Positive errors: "+str(len(err_df[err_df['xu3gem5 duration signed err'] >= 0])))
    over_100_MAPE = err_df[err_df['xu3gem5 duration pc err'] > 100.0]['xu3 stat workload name'].unique().tolist()
    print("Over 100 errors: "+str(over_100_MAPE))
    err_df[err_df['xu3 stat workload name'].isin(over_100_MAPE)][['xu3 stat workload name', 'xu3 stat core mask', 'xu3 stat Freq (MHz) C4','xu3gem5 duration pc err','xu3gem5 duration signed err']].to_csv(args.input_file_path+'-over-100-mape.csv')

    # NEW CORR ANALYSIS
    workload_clustering_A15_df = workload_clustering(df, '4,5,6,7', 1000,1000, args.input_file_path+'-workloads-cluster')
    # get unique cluster numbers
    #unique_clusters = workload_clustering_A15['Cdd
    print workload_clustering_A15_df
    # find find errors of clusters
    cluster_name = 'cluster 1' # which cluster to analyse
    unique_clusters = workload_clustering_A15_df[cluster_name].unique()
    cluster_mape_dict = {}
    cluster_signed_dict = {}
    workload_mape_dict = {}
    workload_signed_dict = {}
    for cluster_id in unique_clusters:
        list_of_wls = workload_clustering_A15_df[workload_clustering_A15_df[cluster_name] == cluster_id]['wl name'].tolist()
        print("\n\nWorkloads in "+str(cluster_id)+": "+str(list_of_wls))
        # now calculate errors
        cluster_err_df = df[df['xu3 stat workload name'].isin(list_of_wls)]
        cluster_err_df = cluster_err_df[(cluster_err_df['xu3 stat Freq (MHz) C4'] == 1000) & (cluster_err_df['xu3 stat core mask'] == '4,5,6,7')]
        cluster_mape = mape(cluster_err_df['xu3 stat duration (s)'], cluster_err_df['gem5 stat sim_seconds']).mean()
        cluster_signed = signed_err(cluster_err_df['xu3 stat duration (s)'], cluster_err_df['gem5 stat sim_seconds']).mean()
        print(str(cluster_mape)+'%')
        print(str(cluster_signed)+'%')
        cluster_mape_dict[cluster_id] = cluster_mape
        cluster_signed_dict[cluster_id] = cluster_signed
        for wl in list_of_wls:
            temp_wl_err = cluster_err_df[cluster_err_df['xu3 stat workload name'] == wl]
            if len(temp_wl_err.index) > 1:
                raise ValueError("More than one workload with same name!"+str(cluster_err_df))
            workload_mape_dict[wl] = mape(temp_wl_err['xu3 stat duration (s)'], temp_wl_err['gem5 stat sim_seconds']).mean()
            workload_signed_dict[wl] = signed_err(temp_wl_err['xu3 stat duration (s)'], temp_wl_err['gem5 stat sim_seconds']).mean()
    cluster_mape_list = [cluster_mape_dict[x] for x in workload_clustering_A15_df[cluster_name]]
    cluster_signed_list = [cluster_signed_dict[x] for x in workload_clustering_A15_df[cluster_name]]
    workload_clustering_A15_df[cluster_name+'_MAPE'] = cluster_mape_list
    workload_clustering_A15_df[cluster_name+'_signed_err'] = cluster_signed_list
    workload_clustering_A15_df['worklood_MAPE'] = [workload_mape_dict[x] for x in workload_clustering_A15_df['wl name']]
    workload_clustering_A15_df['worklood_signed'] = [workload_signed_dict[x] for x in workload_clustering_A15_df['wl name']]
    # now add individual workload error column
    print workload_clustering_A15_df
    workload_clustering_A15_df.to_csv(args.input_file_path+'-clustering-wl-w-errors.csv',sep='\t')
    workload_clustering_A15_df[cluster_name] = [int(x) for x in workload_clustering_A15_df[cluster_name].tolist()]
    workload_clustering_A15_df.sort_values([cluster_name],ascending=True).to_csv(args.input_file_path+'-clustering-wl-w-errors-ordered-by-cluster.csv',sep='\t')
    
    # print df with workload clusters - for A15
    df_A15_c_n = df[(df['xu3 stat Freq (MHz) C4'] == 1000) & (df['xu3 stat core mask'] == '4,5,6,7')]
    # for adding the clusters numbers to the main df (added for the power modelling)
    df_A15_c_n['workload A15 clusters'] = df_A15_c_n['xu3 stat workload name'].apply(lambda x: workload_clustering_A15_df[workload_clustering_A15_df['wl name'] == x][cluster_name].iloc[0])
    print df_A15_c_n
    df_A15_c_n.to_csv(args.input_file_path+'-A15-with-formulae-and-clusters.csv',sep='\t')
    #candyfloss

    import pmcs_and_gem5_stats as ps
    #a15_100_df = df[(df['xu3 stat Freq (MHz) C4'] == 1000) & (df['xu3 stat core mask'] == '4,5,6,7') ]
    a15_100_df = df_A15_c_n[[x for x in df_A15_c_n.columns.values.tolist()]]
    pmc_compare_a15_df = ps.create_pmc_err_df(a15_100_df,'a15', 'bigCluster', 'xu3 stat duration (s)', 'gem5 stat sim_seconds',use_last_if_dup=True)
    pmc_compare_a15_df.to_csv(args.input_file_path+'-pmc-compare-a15.csv',sep='\t')
    print pmc_compare_a15_df
    pmc_compare_a15_df[[x for x in pmc_compare_a15_df.columns.values if x.find('total MPE') > -1]].mean().to_csv(args.input_file_path+'-pmc-compare-a15-total-mpe.csv',sep='\t')
    pmc_compare_a15_df[[x for x in pmc_compare_a15_df.columns.values if x.find('cluster rate MPE') > -1]].mean().to_csv(args.input_file_path+'-pmc-compare-a15-cluster-rate-mpe.csv',sep='\t')
    pmc_compare_a15_df[[x for x in pmc_compare_a15_df.columns.values if x.find('total SMPE') > -1]].mean().to_csv(args.input_file_path+'-pmc-compare-a15-total-smpe.csv',sep='\t')
    pmc_compare_a15_df[[x for x in pmc_compare_a15_df.columns.values if x.find('cluster rate SMPE') > -1]].mean().to_csv(args.input_file_path+'-pmc-compare-a15-cluster-rate-smpe.csv',sep='\t')
    # now average each cluster normalised values
    norm_cluster_df = pmc_compare_a15_df[[x for x in pmc_compare_a15_df.columns.values.tolist()]]
    print norm_cluster_df['workload A15 clusters']
    unique_clusters = norm_cluster_df['workload A15 clusters'].unique()
    norm_cluster_cols = ['workload A15 clusters']+[x for x in norm_cluster_df.columns.values.tolist() if x.find('Normalised gem5 ') > -1]
    final_norm_cluster_df = pd.DataFrame(columns=norm_cluster_cols)
    for cur_cluster in list(set(unique_clusters)):
        final_norm_cluster_df = final_norm_cluster_df.append(norm_cluster_df[norm_cluster_df['workload A15 clusters'] == cur_cluster][norm_cluster_cols].mean(),ignore_index=True)
    final_norm_cluster_df = final_norm_cluster_df.append(norm_cluster_df[norm_cluster_cols].mean(),ignore_index=True)
    # rename last row cluster id to 'mean'
    final_norm_cluster_df['workload A15 clusters'].iloc[-1] = 'mean'
    final_norm_cluster_df = final_norm_cluster_df.append(norm_cluster_df[norm_cluster_df['workload A15 clusters'] != 16][norm_cluster_cols].mean(),ignore_index=True)
    final_norm_cluster_df['workload A15 clusters'].iloc[-1] = 'mean-no-16'
    print final_norm_cluster_df
    final_norm_cluster_df[['workload A15 clusters']+[x for x in final_norm_cluster_df.columns.values.tolist() if x.find('Normalised gem5 ') > -1]].to_csv(args.input_file_path+'-normalised-pmcs-clustered.csv',sep='\t')
    # now transpose for plotting:
    norm_pmcs_cols = [x for x in final_norm_cluster_df.columns.values if x.find('Normalised gem5 ') > -1]
    cluster_vals = final_norm_cluster_df['workload A15 clusters'].tolist()
    final_norm_cluster_T_df = pd.DataFrame()
    for c in cluster_vals:
        #final_norm_cluster_T_df[c] = final_norm_cluster_df[final_norm_cluster_df['workload A15 clusters'] == c][norm_pmcs_cols]
        final_norm_cluster_T_df[c] = final_norm_cluster_df[final_norm_cluster_df['workload A15 clusters'] == c][norm_pmcs_cols].iloc[0]
    final_norm_cluster_T_df.index = [x[x.find('Normalised gem5 ')+16:] for x in final_norm_cluster_T_df.index.values]
    print final_norm_cluster_T_df
    final_norm_cluster_T_df.to_csv(args.input_file_path+'-normalised-pmcs-clustered_T.csv',sep='\t')
    tigermoth

    
    corr_and_cluster_analysis(df, '4,5,6,7', 1000, 1000, 'xu3gem5 duration signed err', args.input_file_path+'-cluster')
    corr_and_cluster_analysis(df, '0,1,2,3', 1000, 1000, 'xu3gem5 duration signed err', args.input_file_path+'-A7-cluster',corr_only=True)
    #build_model(df, '4,5,6,7', 1000, 1000, 'xu3gem5 duration signed err',[x for x in df.columns.values.tolist() if (x.find('xu3new') > -1 and x.find('a15') > -1) or (x.find('gem5 stat') > -1)],10)
    df['duration diff'] = df['xu3 stat duration (s)'] - df['gem5 stat sim_seconds']
    #build_model(df, '4,5,6,7', 1000, 1000, 'duration diff',[x for x in df.columns.values.tolist() if (x.find('xu3new') > -1 and x.find('a15') > -1) ],10)
    #col_filter = lambda f : [x for x in f.columns.values.tolist() if x.find('gem5 stat') > -1 ]
    #build_model(df, '4,5,6,7', 1000, 1000, 'duration diff',col_filter,10,'model-build-a15-gem5-')
    col_filter = lambda f : [x for x in f.columns.values.tolist() if x.find('xu3new ') > -1 ]
    col_filter = lambda f : [x for x in f.columns.values.tolist() if x.find('gem5 stat ') > -1 ]
    #build_model(df[(df['xu3 stat Freq (MHz) C4'] == 1000) & (df['xu3 stat core mask'] == '4,5,6,7')], '4,5,6,7', 1000, 1000, 'duration diff',col_filter,7,args.input_file_path+'-model-build-a15-xu3-')
    build_model(df, '4,5,6,7', 1000, 1000, 'duration diff',col_filter,10,args.input_file_path+'-model-build-a15-xu3-')
    #build_model(df, '4,5,6,7', 1000, 1000, 'xu3gem5 duration signed err',df.columns.values.tolist(),10)
    #elephant

    # find missing  workloads:
    unique_wls = df['xu3 stat workload name'].unique()
    for wl in unique_wls:
        print(wl+": "+str(df['xu3 stat workload name'].value_counts()[wl]))

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
    correlation_gem5_df.to_csv(args.input_file_path+'-correlation_gem5_pmcs.csv',sep='\t')

    # correlation with exeuction time
    temp_cor_df = df[(df['xu3 stat core mask'] == '4,5,6,7') & (df['xu3 stat Freq (MHz) C4'] == 1000)]
    error_col_w_xu3_time = pearsonr(temp_cor_df['xu3 stat duration (s)'], temp_cor_df['xu3gem5 duration signed err'])
    #error_col_w_gem5_time = pearsonr(temp_cor_df['xu3 stat duration (s)'], temp_cor_df['xu3gem5 duration signed err'])
    print("Correlation of duration with MPE:"+str(error_col_w_xu3_time))
    cakes
    

    # now find top 20 gem5 PMCs:
    csv_string = ''
    for i in range(0, len(correlation_gem5_df.index)):
        if correlation_gem5_df.iloc[i] > 0.3 or correlation_gem5_df.iloc[i] < -0.3:
            print(str(correlation_gem5_df.index[i]) + "    " + str(correlation_gem5_df.iloc[i]))
            csv_string += (str(correlation_gem5_df.index[i]) + "\t" + str(correlation_gem5_df.iloc[i]))+'\n'
        #print(str(corr.index)+" "+str(corr))
    with open(args.input_file_path+'-correlation-top-gem5-pmcs.csv', 'w') as f:
        f.write(csv_string)
    f.closed

    # gem5 correlation analysis 2:
    # 1. Filter
    g5_cor_an_df = df[(df['xu3 stat core mask'] == '4,5,6,7') & (df['xu3 stat Freq (MHz) C4'] == 1000)]
    # 2. Combine events from different CPUs:
    gem5_stat_cols_no_cpu = [x for x in g5_cor_an_df.columns.values if (x.find('gem5 stat ') > -1 and x.find('system.bigCluster.cpus') == -1 and x.find('system.littleCluster') == -1) or  x == duration_signed_col]
    new_g5_corr_df = g5_cor_an_df[gem5_stat_cols_no_cpu]
    gem5_stat_cols_cpu = [x for x in g5_cor_an_df.columns.values if x.find('system.bigCluster.cpu') > -1]
    for cpu_stat in gem5_stat_cols_cpu:
        new_col_name = cpu_stat.replace('.cpus0.','.cpusX.').replace('.cpus1.','.cpusX.').replace('.cpus2.','.cpusX.').replace('.cpus3.','.cpusX.')
        new_col_val = 0
        key_error = 0
        if new_col_name in new_g5_corr_df.columns.values.tolist():
            continue # skip if done already
        try:
            new_col_val += g5_cor_an_df[new_col_name.replace('.cpusX.','.cpus0.')]
        except KeyError:
            key_error += 1
        try:
            new_col_val += g5_cor_an_df[new_col_name.replace('.cpusX.','.cpus1.')]
        except KeyError:
            key_error += 1
        try:
            new_col_val += g5_cor_an_df[new_col_name.replace('.cpusX.','.cpus2.')]
        except KeyError:
            key_error += 1
        try:
            new_col_val += g5_cor_an_df[new_col_name.replace('.cpusX.','.cpus3.')]
        except KeyError:
            key_error += 1
        if key_error > 3:
            raise KeyError("Failed at life")
        new_g5_corr_df[new_col_name] = new_col_val
    print new_g5_corr_df
    # the new_g5_corr_df should now have the non-cpu and cpuX columns only
    # 3.  Add the signed error back on
    cols_to_apply = [x for x in new_g5_corr_df.columns.values.tolist() if x != duration_signed_col]
    cols_to_apply = [x for x in cols_to_apply if x in new_g5_corr_df._get_numeric_data().columns.values.tolist()]
    print ("duration signed col:")
    print new_g5_corr_df[duration_signed_col]
    new_g5_cor_result = new_g5_corr_df[cols_to_apply].apply(lambda x: pearsonr(x,new_g5_corr_df[duration_signed_col])[0])
    new_g5_cor_result.to_csv(args.input_file_path+'-new-g5-cor-anal.csv',sep='\t')
    # now find top 20 gem5 PMCs:
    csv_string = ''
    for i in range(0, len(new_g5_cor_result.index)):
        if new_g5_cor_result.iloc[i] > 0.3 or new_g5_cor_result.iloc[i] < -0.3:
            print(str(new_g5_cor_result.index[i]) + "    " + str(new_g5_cor_result.iloc[i]))
            csv_string += (str(new_g5_cor_result.index[i]) + "\t" + str(new_g5_cor_result.iloc[i]))+'\n'
        #print(str(corr.index)+" "+str(corr))
    with open(args.input_file_path+'-new-g5-cor-anal-over-30', 'w') as f:
        f.write(csv_string)
    f.closed
    # do cluster analysis
    # SPLIT INTO CLUSTER with both XU3 and gem5 stats
    
