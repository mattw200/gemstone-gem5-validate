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
        results['ordered_headers'].insert(0, 'workload name') 
        results['stats']['workload name'] = temp_workload_name
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
    
        
        

if __name__=='__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--m5out-dir', dest='gem5_out_dir', required=True, \
               help="The gem5 m5out directory")
    '''
    parser.add_argument('--powmon_dir', dest='powmon_out_dir', required=True, \
               help="The powmon experiment output directory")
    '''
    args=parser.parse_args()
    stats_filepath = os.path.join(args.gem5_out_dir, 'stats.txt')
    stats_df = stats_to_df(stats_filepath)
    test_eqn = "sim_milliseconds = sim_seconds*1000.0"
    #apply_new_stat(stats_df, "g5_stat_", " new_name", test_eqn)
    apply_stat_equations_from_file(stats_df, 'stats.equations')
    print stats_df
    
