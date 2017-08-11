#!/usr/bin/env python

# Matthew J. Walker
# Created: 11 August 2017

# Generates all the bootscripts and iridis scripts. 
# Not as flexible, to be adapted as needed in future
# Assumes a local copy workload list and xu3 results

import create_bootscript

# freq is a preset as it needs to be supported by checkpoints
freqs_dict = {
        '600-600' : {
            'a7_freq' : "0.6GHz",
            'a15_freq' : "0.6GHz",
            'checkpoint_path_bko' : 'm5out-bko-l600-b600/cpt.4553141051121'
         },
        '1000-1000' : {
            'a7_freq' : "1.0GHz",
            'a15_freq' : "1.0GHz",
            'checkpoint_path_bko' :  'm5out-bko-l1000-b1000/cpt.3861983892000'
        },
        '1400-1800' : {
            'a7_freq' : "1.4GHz",
            'a15_freq' : "1.8GHz",
            'checkpoint_path_bko' :  'm5out-bko-l1400-b1800/cpt.3566806185128'
        }
}

models_list = ['bko']

def create_iridis_run_script(checkpoint_dir, little_clock, big_clock, bootscript_path, m5out_dir, wall_hours, run_script_filepath):
    if wall_hours >= 60:
        raise ValueError("Wall time must be less than 60")
    script_text = "#!/bin/bash\n"
    script_text += "#PBS -l walltime="+str(int(wall_hours))+":00:00\n"
    script_text += "#PBS -m ae -M mw9g09@ecs.soton.ac.uk\n"
    script_text += "module load python\n"
    script_text += "module load gcc/4.9.1\n"
    script_text += "cd gem5-repo/gem5/\n"
    script_text += "build/ARM/gem5.opt --outdir="+m5out_dir+"  configs/example/arm/fs_bigLITTLE.py --restore-from="+checkpoint_dir+" --caches --little-cpus 4 --big-cpus 4 --big-cpu-clock "+big_clock+" --little-cpu-clock "+little_clock+" --bootscript "+bootscript_path+" --cpu-type exynos\n"
    script_text += 'echo "Finished iridis run script"\n'
    with open(run_script_filepath, "w") as f:
        f.write(script_text)
    f.close

if __name__=='__main__':
    import argparse
    import pandas
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', dest='gem5_dir', required=True, \
               help="The gem5 directory")
    parser.add_argument('--preset-list', dest='preset_list', required=True, \
            help="List of which presets to use. Available presets: " \
            +str(create_bootscript.get_presets()))
    parser.add_argument('--model', dest='model', required=True, \
            help="Which gem5 mode to use (i.e. checkpoint). Avail.:" \
            +str(models_list))
    parser.add_argument('--freq', dest='freq', required=True, \
            help="Which freq to use (i.e. checkpoint). Avail.:" \
            +str(freqs_dict))
    args=parser.parse_args()

    # automatically does for big and little
    # use consistent nameing
    # gem5/gem5-auto/
    # g5-out-XXX-bko-1000-1400-4_5_6_7-mibenchA

    if not args.model in models_list:
        print("Error: "+args.model+" not in models list")
        sys.exit()

    print("Checkpoint selection: "+freqs_dict[args.freq]['checkpoint_path_'+args.model]+".")

    '''
    python create_bootscript.py --list ../workloads-small.config.armv7 --mask 4,5,6,7 --xu3-results ../../powmon-experiment-060-high-f/pmc-events-log.out-analysed.csv  --preset "parmibench"
    '''
    gem5_auto_dir = os.path.join(args.gem5_dir, 'gem5-auto')
    bootscripts_dir = os.path.join(gem5_auto_dir, 'bootscripts')
    runscripts_dir = os.path.join(gem5_auto_dir, 'runscripts')
    
    if not os.path.exists(gem5_auto_dir):
        os.makedirs(gem5_auto_dir)    
    if not os.path.exists(bootscripts_dir):
        os.makedirs(bootscripts_dir)    
    if not os.path.exists(runscripts_dir):
        os.makedirs(runscripts_dir)    

    this_file_dir = dir_path = os.path.dirname(os.path.realpath(__file__))
    workload_list_filepath = os.path.join(this_file_dir, '../workloads-small.config.armv7')
    xu3_results_filepath = os.path.join(this_file_dir, 'xu3-results.example.csv')

    experiment_label = '000'
    with open(os.path.join(this_file_dir, 'gem5-auto-counter'), 'r') as f:
        experiment_label = "{0:0>3}".format(int(f.read()))
    f.closed
    with open(os.path.join(this_file_dir, 'gem5-auto-counter'), 'w') as f:
        f.write(str(int(experiment_label)+1))
    f.closed

    run_all_script  = "#!/bin/bash\n"
    for preset in presets:
        #for big and for little
        cores_masks = ['0,1,2,3','4,5,6,7']
        for mask in core_masks:
            filename_prefixes = experiment_label+'-'+args.model+'-'+args.freq+'-'+mask.replace('-','_')+preset
            bootscript_filepath = os.path.join(bootscripts_dir,filename_prefixes+'.rcS')
            runscript_filepath = os.path.join(runscripts_dir, filename_prefixes+'.sh')
            m5out_dir_path = os.path.join(args.gem5_dir, 'gem5out-'+filename_prefixes)
            create_bootscript.create_rcs_from_preset(workload_list_filepath, xu3_results_filepath, mask, preset, bootscript_filepath)
            create_iridis_run_script(
                    freqs_dict[args.freq]['checkpoint_path_'+args.model],
                    freqs_dict[args.freq]['a7_freq'],
                    freqs_dict[args.freq]['a15_freq'],
                    bootscript_filepath,
                    m5out_dir_path,
                    58,
                    runscript_filepath
                )
            run_all_script += "qsub "+runscript_filepath+"\n"
    with open(os.path.join(args.gem5_dir, 'run-all-'+experiment_label), 'w') as f:
        f.write(run_all_script)
    f.closed


