#!/usr/bin/env python

# Matthew J. Walker
# Created: 31 July 2017

# Uses the workloads list and creates a bootscript

if __name__=='__main__':
    import argparse
    import pandas
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', dest='workloads_list', required=True, \
               help="The workloads list file path")
    parser.add_argument('--mask', dest='cpu_mask', required=True, \
               help="The cpu mask, e.g. '4,5,6,7'")
    parser.add_argument('--path-find', dest='path_find', required=False, \
               help="If the string is in the workloads path, the workload is included")
    parser.add_argument('-o', dest='output_file', required=False, \
               help="The output filename (should end in .rCS)")
    args=parser.parse_args()

    workloads_df = pandas.read_csv(args.workloads_list, sep='\t')

    print (workloads_df)

    output_text = "#!/bin/sh\n\n"

    # sleep for 1 second (acts as marker)
    output_text += 'echo "Doing sleep 1"\n'
    output_text += '/sbin/m5 resetstats\n'
    output_text += 'sleep 1\n'
    output_text += '/sbin/m5 dumpstats\n\n'

    for i in range(0, len(workloads_df.index)):
        workload_path = workloads_df['Directory'].iloc[i]
        if args.path_find:
            if workload_path.find(args.path_find) < 0:
                print ('excluding workload: '+workloads_df['Name'].iloc[i])
                continue
        print ('INCLUDING workload: '+workloads_df['Name'].iloc[i])
        
        output_text += 'echo "'+workloads_df['Name'].iloc[i]+'"\n'
        output_text += 'cd '+workloads_df['Directory'].iloc[i].replace('/home/odroid/','/home/gem5/')+'\n'
        output_text += '/sbin/m5 resetstats\n'
        output_text += 'taskset -c '+args.cpu_mask+' '+workloads_df['Command'].iloc[i]+'\n'
        output_text += '/sbin/m5 dumpstats\n\n'
        
    # sleep for 1 second (acts as marker)
    output_text += 'echo "Doing sleep 1"\n'
    output_text += '/sbin/m5 resetstats\n'
    output_text += 'sleep 1\n'
    output_text += '/sbin/m5 dumpstats\n\n'

    output_filename = 'temp.rCS'
    if args.output_file:
        output_filename = args.output_file

    with open(output_filename, 'w') as f:
        f.write(output_text)
    f.closed
