#!/usr/bin/env python

# Matthew J. Walker
# Created: 31 July 2017

# gives summary of gem5 results
# Total time, instructions (per-core), L1 caches, L2 caches, mem

keys = [
    {
        'contains' : 'sim_seconds',
        'not_contains' : None
    },
    {
        'contains' : 'numCycles',
        'not_contains' : None
    },
    {
        'contains' : 'idleCycles',
        'not_contains' : None
    {
        'contains' : 'committedInsts',
        'not_contains' : '.commit.'
    },
    {
        'contains' : '.iew.iewExecutedInsts',
        'not_contains' : None
    },
    {
        'contains' : 'iew.iewExecSquashedInsts',
        'not_contains' : None
    },
    {
        'contains' : 'l2.prefetcher.num_hwpf_issued',
        'not_contains' : None
    },
    {
        'contains' : 'l2.tags.data_accesses',
        'not_contains' : None
    },
    {
        'contains' : 'l2.demand_hits::total',
        'not_contains' : None
    },
    {
        'contains' : 'l2.overall_misses::total',
        'not_contains' : None
    },
    {
        'contains' : 'l2.demand_accesses::total',
        'not_contains' : None
    },
    {
        'contains' : 'system.membus.trans_dist::ReadReq',
        'not_contains' : None
    },
    {
        'contains' : 'system.membus.trans_dist::ReadResp',
        'not_contains' : None
    },
    {
        'contains' : 'system.membus.trans_dist::WriteReq',
        'not_contains' : None
    },
    {
        'contains' : 'system.membus.trans_dist::WriteResp',
        'not_contains' : None
    },
    {
        'contains' : 'iq.FU_type_0::',
        'not_contains' : None
    }
]

def process_workload_stats_text(workload_text,ordered_keys_list):
    #found_keys = [None]*len(ordered_keys_list)
    #found_values = [None]*len(ordered_keys_list)
    found_keys = []
    found_values = []
    stats_lines = workload_text.split('\n')
    if len(stats_lines) < 10:
        #raise ValueError("Not enough lines in stats")
        print("Skipping this workload")
        return
    for i in range(0, len(stats_lines)):
        fields = stats_lines[i].split()
        if len(fields) < 2:
            print("Skipping this row: "+str(fields))
            continue
        for j in range(0, len(keys)):
            item = keys[j]
            if fields[0].find(item['contains']) > -1:
                if  item['not_contains']:
                    if fields[0].find(item['not_contains']) > -1:
                        # not a match
                        continue
                # a match!
                print("Found: "+fields[0]+"   value: "+str(float(fields[1]))+"  (key index: "+str(j)+")")
                #found_keys[j] = fields[0]
                #found_values[j] = float(fields[1])
                found_keys.append(fields[0])
                found_values.append(float(fields[1]))
    return { 'headers' : found_keys, 'values' : found_values}
            
if __name__=='__main__':
    import argparse
    import pandas
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', dest='gem5_out_dir', required=True, \
               help="The gem5 m5out directory")
    args=parser.parse_args()
    gem5_filepath = os.path.join(args.gem5_out_dir, 'stats.txt')
    m5out_text = ""
    with open(gem5_filepath, 'r') as f:
        m5out_text = f.read()
    f.closed
    m5out_split_wls = m5out_text.split("---------- Begin Simulation Statistics ----------")
    print("Number of workloads: "+str(len(m5out_split_wls)))
    results_df = None
    setup_df = False
    for i in range(0, len(m5out_split_wls)):
        print("\n\nWorkload "+str(i))
        results = process_workload_stats_text(m5out_split_wls[i],keys)
        if not results:
            continue
        if not setup_df:
            headers = [h for h in results['headers'] if h != None]
            results_df = pandas.DataFrame(columns=headers)
            setup_df = True
        row_data = [d for d in results['values'] if d != None]
        temp_row  = pandas.Series(row_data, index=results_df.columns.values)
        results_df = results_df.append(temp_row, ignore_index=True)
        # TODO create pandas DF
    
    print results_df
    results_path = os.path.join(args.gem5_out_dir, 'results-summary.csv')
    results_df.to_csv(results_path, sep='\t')
    
