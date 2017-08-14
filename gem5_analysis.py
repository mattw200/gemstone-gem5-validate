#!/usr/bin/env python

# Matthew J. Walker
# Created: 31 July 2017

# Framework for analysing gem5 stats

def apply_new_stat(stats_df, stat_name, equation):
    stats_df.eval(stat_name+' = '+equation)
    
def compare_two_stats(stats_df, stat_A_name, stat_B_name):
    stats_df['comp. '+stat_A_name+' vs. '+stat_B_name+' (%)'] = \
            (((stats_df[stat_A_name] - stats_df[stat_B_name]).abs()) / stats_df[stat_A_name]) * 100.0
    
def process_gem5_dump(dump_text):
    stats_lines = dump_text.split('\n')
    ordered_headers = []
    stats_dump_dict = {}
    if len(stats_lines) < 10:
        #raise ValueError("Not enough lines in stats")
        print("Skipping this workload")
        return
    for i in range(0, len(stats_lines)):
        if stats_lines[i].find("End Simulation Statistics") > -1:
            continue
        fields = stats_lines[i].split()
        if len(fields) < 2:
            print("Skipping this row: "+str(fields))
            continue
        print fields
        print("PROCESSNG: "+fields[0]+": "+str(float(fields[1])))
        ordered_headers.append(fields[0])
        stats_dump_dict[fields[0]] =  float(fields[1])
    return { 'ordered_headers' : ordered_headers, 'stats' : stats_dump_dict}

def stats_to_df(path_to_stats_file):
    # Open stats file
    # Create DF from stats
    # Careful: stats that don't change don't appear in file!
    # Handle multiple dumps in same file
    # Each row is a dump
    # Each column is a stat
  
    import os
    import pandas as pd
    
    m5out_text = ""
    with open(path_to_stats_file, 'r') as f:
        m5out_text = f.read()
    f.closed
    m5out_split_wls = m5out_text.split("---------- Begin Simulation Statistics ----------")
    print("Number of workloads: "+str(len(m5out_split_wls)))
    is_df_set_up = False
    stats_df = None
    for i in range(0, len(m5out_split_wls)):
        print("\n\ndump "+str(i))
        temp_workload_name = 'dump '+str(i)
        results = process_gem5_dump(m5out_split_wls[i])
        if not results:
            continue
        # add workload/dump (temporary) name
        results['ordered_headers'].insert(0, 'workload name temp') 
        results['stats']['workload name temp'] = temp_workload_name
        if not is_df_set_up:
            stats_df = pd.DataFrame(columns=results['ordered_headers'])
            is_df_set_up = True
        temp_row  = pd.Series(results['stats'], index=stats_df.columns.values)
        stats_df = stats_df.append(temp_row, ignore_index=True)
    return stats_df

# TODO apply a list of workload names (in the m5out dir) to the stats
def apply_workload_names_to_stats():
    pass

def apply_stat_equations_from_file(stats_df, path_to_equation_file):
    '''
    equation_lines = []
    with open(path_to_equation_file, 'r') as f:
        equation_lines = f.read().split('\n')
    f.closed
    for i in range(0, len(equation_lines)):
    '''
    import pandas as pd
    equations_df = pd.read_csv(path_to_equation_file, sep='\t')
    print("Applying the following equations:")
    print equations_df
    for i in range(0, len(equations_df.index)):
        print("Applying equation "+str(i)+": "+str(equations_df['Equation'].iloc[i]))
        apply_new_stat(stats_df, equations_df['Stat Name'].iloc[i], equations_df['Equation'].iloc[i])

def get_experiment_number_from_full_directory_path(path):
    import os
    dir_name =  os.path.basename(os.path.normpath(path))
    if dir_name.startswith('gem5out-'):
        return int(dir_name.split('-')[1])
    else:
        return None
    
def get_experiment_directories(top_search_dir, experiment_numers):
    import os
    # check experiment numbers have been converted to ints:
    experiment_numers = [int(x) for x in experiment_numers]
    #match_dirs = [x[0] for x in os.walk(top_search_dir) if os.path.basename(os.path.normpath(x[0])).find('gem5out-'+experiment_number_string) > -1]
    match_dirs = [x[0] for x in os.walk(top_search_dir) \
           if get_experiment_number_from_full_directory_path(x[0]) in experiment_numers ]
    print ("The folling directories match:")
    print match_dirs
    return match_dirs

def get_info_from_terminal_out(terminal_out_filepath):
    lines = []
    with open(terminal_out_filepath, 'r') as f:
        lines = f.read().split('\n')
    f.closed
    workloads_start = []
    workloads_complete = []
    workload_masks = []
    info = {}
    for line in lines:
        if line.find('-----POWMON PRESET') > -1:
            info['preset'] = line.split(':')[1].strip()
        elif line.find('-----POWMON WORKLOAD COMPLETE') > -1:
            workloads_complete.append(line.split(':')[1].strip())
            workload_masks.append(line.split(':')[2].strip())
        elif line.find('-----POWMON WORKLOAD START') > -1:
            workloads_start.append(line.split(':')[1].strip())
    info['workloads complete'] = workloads_complete
    info['workloads start'] = workloads_start
    info['masks'] = workload_masks
    return info

   


def analyse_gem5_results(
          top_search_directory,
          experiment_numbers_list,
          stats_formulae_filepath,
          output_filename
          ):
    import os
    gem5outs_to_process = get_experiment_directories(top_search_directory,\
           experiment_numbers_list)
    stats_dfs = []
    for dir_i in range(0, len(gem5outs_to_process)):
        stats_file = os.path.join(gem5outs_to_process[dir_i],'stats.txt')    
        terminal_out_file = os.path.join(gem5outs_to_process[dir_i],'system.terminal')
        terminal_out_info = get_info_from_terminal_out(terminal_out_file)
        print("Analysing experiment: "+gem5outs_to_process[dir_i])
        print("Benchmark preset: "+str(terminal_out_info['preset']))
        print("Workloads (complete): "+str(terminal_out_info['workloads complete']))
        print("Masks (complete): "+str(terminal_out_info['masks']))
        print("WARNING: incomplete workloads: "+str([x for x in terminal_out_info['workloads start'] if x not in terminal_out_info['workloads complete']]))
        stats_df = stats_to_df(stats_file)
        row_count = len(stats_df.index)
        experiment_name = os.path.basename(os.path.normpath(gem5outs_to_process[dir_i]))
        # add the stats:
        # 1) experiment number (from dir name)
        # 2) model name
        # 2) workload name
        # 3) core mask
        # 4) a7 and a15 freq (from dir name)
        # 4) preset name
        # full experiment dir
        experiment_number = [ int(experiment_name.split('-')[1]) ]*row_count
        model_name = [ experiment_name.split('-')[2] ]*row_count
        a7_freq_mhz = [ int(experiment_name.split('-')[3]) ]*row_count
        a15_freq_mhz = [ int(experiment_name.split('-')[4]) ]*row_count
        preset = [ terminal_out_info['preset'] ]*row_count
        experiment_dirname = [ experiment_name ]*row_count
        masks = terminal_out_info['masks']
        workloads_complete = terminal_out_info['workloads complete']
        # NOTE: add workload and mask because of the random extra entry
        # Another note: THERE IS ONLY AN EXTRA WORKLOAD IF ALL WORKLOADS COMPLETE
        if False:
            print("Length of index:"+str(len(stats_df.index)))
            print("Length of workload names:"+str(len(workloads_complete)))
            print("stats_df workloads:")
            print(stats_df['workload name temp'])
            print("stats_df times:")
            print(stats_df['sim_seconds'])
            print("workloads complete: ")
            print workloads_complete
        if not len(stats_df.index) == len(workloads_complete):
            workloads_complete.append("NA")
            masks.append("NA")
        stats_df.insert(0, 'm5out directory', experiment_name)
        stats_df.insert(0, 'A15 Freq (MHz)', a15_freq_mhz)
        stats_df.insert(0, 'A7 Freq (MHz)', a7_freq_mhz)
        stats_df.insert(0, 'core mask', masks)
        stats_df.insert(0, 'workload name', workloads_complete)
        stats_df.insert(0, 'workloads preset', preset)
        stats_df.insert(0, 'model name', model_name)
        stats_dfs.append(stats_df) 
        # TODO check for duplicates and missing data
        # e.g. check that the same workload,freq,mask combo isn't
        # in two experiments.
        # And check that one freq/mask isn't missing a workload the
        # other one has. 
    # now concatinate stats
    results_df = stats_dfs[0]
    for i in range(1, len(stats_dfs)):
        results_df = results_df.append(stats_dfs[i])
    print results_df
    # in old versions of pandas the cols get mixed up
    results_df = results_df[stats_dfs[0].columns.values]
    # now apply equations:
    apply_stat_equations_from_file(results_df, 'stats.equations')
    results_df.to_csv(output_filename, sep='\t')

# give it a directory, give it an experiment number (then recursively goes
# through all the directories looking for gem5out directories matching 
# that experiment number). 
# Then derivces experiment details and adds to DF:
#   - experiment number
#   - date/time run
#   - a7 and a15 frequency
#   - benchmark suite (preset name)
#   - workload name
#   - checkpoint name (for checking/debug)
#   - core mask
#   - benchmark suite (actual)

if __name__=='__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', dest='results_dir', required=True, \
               help="The directory in which to search for results directories recursively")
    parser.add_argument('--experiments', dest='experiment_numbers', required=True, \
               help="A list of experiment numbers of the experiments to include. E.g. '19,20,21'")
    parser.add_argument('-o', '--output-file', dest='output_file', required=False)
    args=parser.parse_args()
    if not args.output_file:
        args.output_file = 'temp.csv'
    args.experiment_numbers = args.experiment_numbers.split(',')
    analyse_gem5_results(args.results_dir, args.experiment_numbers,'stats.equations', args.output_file)
    banananannanana
    stats_filepath = os.path.join(args.results_dir, 'stats.txt')
    stats_df = stats_to_df(stats_filepath)
    test_eqn = "sim_milliseconds = sim_seconds*1000.0"
    #apply_new_stat(stats_df, "g5_stat_", " new_name", test_eqn)
    apply_stat_equations_from_file(stats_df, 'stats.equations')
    print stats_df
    
