#!/usr/bin/env python

# Matthew J. Walker
# Created: 31 July 2017

# Uses the workloads list and creates a bootscript

# TODO:
# add m5 exit
# add special lables
# use odroid results to calculate time of rcS file.
# allow to split into different chunks depending on time

presets = {}

presets['test'] = [
    'jpeg-decode',
    'susan-corners'
]

presets['mibenchA'] = [
    'basicmath',
    'bitcount-small',
    'qsort',
    'susan-smoothing',
    'susan-edges',
    'susan-corners',
    'jpeg-encode',
    'jpeg-decode',
    'typeset'
]

presets['mibenchB'] = [
    'dijkstra',
    'patricia',
    'stringsearch',
    'blowfish-encode-small',
    'sha',
    'adpcm-encode-small',
    'adpmc-decode-small',
    'crc32-small',
    'fft',
    'inv-fft',
    'gsm-encode-small'
]

presets['parmibench'] = [
    'par-basicmath-square-small',
    'par-basicmath-cubic-small',
    'par-basicmath-rad2deg-small',
    'par-bitcount',
    'par-susan-smoothing',
    'par-susan-edges',
    'par-susan-corners',
    'par-patricia',
    'par-stringsearch-small',
    'par-sha',
    'par-dijkstra-mqueue'
]

presets['parmibenchA'] = [
    'par-basicmath-square-small',
    'par-basicmath-cubic-small',
    'par-basicmath-rad2deg-small',
    'par-bitcount'
]

presets['parmibenchB'] = [
    'par-patricia',
    'par-stringsearch-small',
    'par-sha',
    'par-dijkstra-mqueue'
]

presets['parmibenchC'] = [
    'par-susan-smoothing',
    'par-susan-edges',
    'par-susan-corners'
]

presets['lmbench'] = [
    'lmb3-lat-ctx-0-18',
    'lmb3-lat-ctx-500-18',
    'lmb3-lat-fs',
    'lmb3-lat-ops',
    'lmb3-pat-proc-fork',
    'lmb3-bw-mem-200m-rd',
    'lmb3-tlb',
    'lmb3-line'
]

presets['lat'] = [
#lmb3-lat-mem-rd-2-256
#lmb3-lat-mem-rd-32-256
    'lmb3-lat-mem-rd-2-8',
    'lmb3-lat-mem-rd-32-8'
]

presets['misc'] = [
    'dhrystone',
    'whetstone'
]

presets['parsec4A'] = [
    'parsec-blackholes-4',
    'parsec-bodytrack-4',
    'parsec-canneal-4',
    'parsec-fluidanimate-4',
    'parsec-streamcluster-4'
]


presets['parsec4B'] = [
    'parsec-swaptions-4',
    'parsec-ferret-4',
    'parsec-dedup-4',
    'parsec-freqmine'
]

presets['parsec1A'] = [
    'parsec-blackholes-1',
    'parsec-bodytrack-1',
    'parsec-canneal-1',
    'parsec-fluidanimate-1',
    'parsec-streamcluster-1'
]

presets['parsec1B'] = [
    'parsec-ferret-1',
    'parsec-swaptions-1',
    'parsec-dedup-1'
]

presets['roy'] = [
    'rl-busspeed-small',
    'rl-dhrystone-small',
    'rl-linpack-small',
    'rl-linpack-FSSP-small',
    'rl-linpack-neon-small',
    'rl-whetstone-small'
]

def get_presets():
    keys = []
    for key in presets:
        keys.append(key)
    return keys

def create_rcs_from_preset(workloads_df_path, xu3_data_path, cpu_mask, preset, output_file):
    import pandas
    import os
    import sys
    workloads_df = pandas.read_csv(workloads_df_path, sep='\t')
    names_to_include = presets[preset]
    create_rcs(workloads_df, names_to_include, cpu_mask, preset, output_file)
    accum = 0
    print("Creating bootscript: "+preset+"  ("+cpu_mask+")")
    for name in names_to_include:
        time = get_workload_time(xu3_data_path, name)
        accum += time
        print(name+"\t\t\ttime:"+str(time)+"\t\t\t\taccum:"+str(accum))

def get_workload_time(xu3_file_path, workload_name):
    # This is the most inefficient piece  of code ever written
    if xu3_file_path:
        import pandas
        xu3_df = pandas.read_csv(xu3_file_path, sep='\t')
        for i in range(0, len(xu3_df.index)):
            if xu3_df['workload name'].iloc[i] == workload_name:
                return float(xu3_df['duration (s)'].iloc[i])
    return 0
    

def create_rcs(workloads_df, names_to_include, cpu_mask, preset, output_file):
    import pandas 
    output_text = "#!/bin/sh\n\n"
    if preset:
        output_text += 'echo "-----POWMON PRESET : '+preset+'"\n\n'
    # sleep for 1 second (acts as marker)
    #output_text += 'echo "Doing sleep 1"\n'
    #output_text += '/sbin/m5 resetstats\n'
    #output_text += 'sleep 1\n'
    #output_text += '/sbin/m5 dumpstats\n\n'

    # Change this: now the order depends on the 'names to include'
    '''
    for i in range(0, len(workloads_df.index)):
        workload_path = workloads_df['Directory'].iloc[i]
        if workloads_df['Name'].iloc[i] not in names_to_include:
            #print ('excluding workload: '+workloads_df['Name'].iloc[i])
            continue
        #print ('INCLUDING workload: '+workloads_df['Name'].iloc[i])
    '''
    for name in names_to_include:
        i = -1
        for x in range(0, len(workloads_df.index)):
            if workloads_df['Name'].iloc[x] == name:
                i = x
                break
        if i == -1:
            raise ValueError("Couldn't find "+name+"in workloads df")
        output_text += 'echo "-----POWMON WORKLOAD START : ' \
                +workloads_df['Name'].iloc[i]+' : '+cpu_mask+'"\n'
        output_text += 'cd '+workloads_df['Directory'].iloc[i].replace('/home/odroid/','/home/gem5/')+'\n'
        output_text += '/sbin/m5 resetstats\n'
        output_text += 'taskset -c '+cpu_mask+' '+workloads_df['Command'].iloc[i]+'\n'
        output_text += '/sbin/m5 dumpstats\n'
        output_text += 'echo "-----POWMON WORKLOAD COMPLETE : ' \
                +workloads_df['Name'].iloc[i]+' : '+cpu_mask+'"\n\n'
        
    # sleep for 1 second (acts as marker)
    output_text += 'echo "-----POWMON WORKLOAD START : ' \
                +'sleep 1 : na "\n'
    output_text += '/sbin/m5 resetstats\n'
    output_text += 'sleep 1\n'
    output_text += '/sbin/m5 dumpstats\n'
    output_text += 'echo "-----POWMON WORKLOAD COMPLETE : ' \
                +'sleep 1 : na "\n\n'

    output_text += 'echo "-----POWMON FINISHED BOOTSCRIPT"\n\n'

    output_text += '/sbin/m5 exit\n\n'

    output_filename = 'temp.rCS'
    if output_file:
        output_filename = output_file

    with open(output_filename, 'w') as f:
        f.write(output_text)
    f.closed



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
    parser.add_argument('--xu3-results', dest='xu3_results', required=False, \
               help="An xu3 results file for timeing info")
    parser.add_argument('-o', dest='output_file', required=False, \
               help="The output filename (should end in .rCS)")
    parser.add_argument('--preset', dest='preset', required=False, \
               help="Preset - see top of code")
    args=parser.parse_args()

    workloads_df = pandas.read_csv(args.workloads_list, sep='\t')

    print (workloads_df)

    print("Listing all workloads: ")
    accum = 0
    for i in range(0, len(workloads_df.index)):
        time = get_workload_time(args.xu3_results, workloads_df['Name'].iloc[i])
        accum += time
        text = workloads_df['Name'].iloc[i]
        if (time > 0):
            text += '\t\t\t\ttime: '+str(time)+'\t\t\t\taccum: '+str(accum)
        print(text)
    print("Finished listing")
    names_to_include = []

    if not args.preset:
        for i in range(0, len(workloads_df.index)):
            workload_path = workloads_df['Directory'].iloc[i]
            if args.path_find:
                if workload_path.find(args.path_find) < 0:
                    continue
            names_to_include.append(workloads_df['Name'].iloc[i])     
    else:
        print("Using preset: "+args.preset)
        names_to_include = presets[args.preset]
    print ("Including: "+str(names_to_include))
    accum = 0
    for name in names_to_include:
        time = get_workload_time(args.xu3_results, name)
        accum += time
        print (name+"\t\t\t\ttime:"+str(time)+"\t\t\t\taccum:"+str(accum))


    #create_rcs(workloads_df, names_to_include, args)
    create_rcs(workloads_df, names_to_include, args.cpu_mask, args.preset, args.output_file)
