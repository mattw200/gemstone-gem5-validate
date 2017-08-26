#!/usr/bin/env python

# Matthew J. Walker
# Created: 14 August 2017

# opens gem5 combined df and xu3 combined df
# applies the new stats to both
# merges them together
# compares specific columns of each one
# then allow optional filtering etc.

gem5_formulae_path = 'gem5-stats.equations'
xu3_formulae_path = 'xu3-stats.equations'

xu3_col_prefix = 'xu3 stat '
gem5_col_prefix = 'gem5 stat '

def combine_xu3_and_gem5(xu3_df, gem5_df, model):
    import pandas as pd

    # filter
    gem5_df = gem5_df[gem5_df['gem5 stat workloads preset'] != 'parmibench'] # parmibench replaced by parmibenchA, parmibenchB, and parmibenchC

    # remove unwanted columns from xu3_df:
    xu3_wanted_cols = [col for col in xu3_df.columns.values if not col.find("Unnamed:") > -1]
    xu3_df = xu3_df[xu3_wanted_cols]
    
    # for each value in xu3
    # match on: workload,freq combination, core mask, model
    failed_finds = []
    temp_list = xu3_df.columns.values.tolist() + gem5_df.columns.values.tolist()
    print temp_list
    print("Length of temp list: "+str(len(temp_list)))
    combined_df = pd.DataFrame(columns=temp_list)

    # go through the xu3 experiment - it is reliable etc.
    for xu3_i in range(0, len(xu3_df.index)):
        print xu3_df.columns.values
        workload_name = xu3_df[xu3_col_prefix+'workload name'].iloc[xu3_i]
        a7_freq = xu3_df[xu3_col_prefix+'Freq (MHz) C0'].iloc[xu3_i]
        a15_freq = xu3_df[xu3_col_prefix+'Freq (MHz) C4'].iloc[xu3_i]
        core_mask = xu3_df[xu3_col_prefix+'core mask'].iloc[xu3_i]
        # now go through the gem5 df to find the equivalent
        print("Looking for: "+str(workload_name)+", "+str(core_mask)+", "+str(a7_freq)+"-"+str(a15_freq))
        filtered_df = gem5_df[gem5_df[gem5_col_prefix+'model name'] == model]
        filtered_df = filtered_df[filtered_df[gem5_col_prefix+'workload name'] == workload_name]
        filtered_df = filtered_df[filtered_df[gem5_col_prefix+'core mask'] == core_mask]
        filtered_df = filtered_df[filtered_df[gem5_col_prefix+'A7 Freq (MHz)'] == a7_freq]
        filtered_df = filtered_df[filtered_df[gem5_col_prefix+'A15 Freq (MHz)'] == a15_freq]
        print filtered_df
        if len(filtered_df.index) == 0:
            # not found
            failed_finds.append([model,workload_name,core_mask,a7_freq,a15_freq])
            continue
        elif len(filtered_df.index) > 1:
            filtered_presets = filtered_df['gem5 stat workloads preset'].unique()
            print("ATTN: found multiple workloads from different workload presents ("+str(filtered_presets)+").")
            print("Choose which one you want to use:")
            for i in range(0, len(filtered_presets)):
              print(str(i)+": "+filtered_presets[i])
            user_sel = int(raw_input("Select the number of which one you want to use: "))
            filtered_df = filtered_df[filtered_df['gem5 stat workloads preset'] == filtered_presets[user_sel]]
            print ("Proceeding with this selection: ")
            print (filtered_df)
            raw_input("Press <enter> to confirm (crtl-c to abort and start over)")
            if len(filtered_df.index) > 1:
                raise ValueError("Multiple entries for the same workload.")
        #gem5_i = gem5_df.iloc[0].index
        #print("Gem5 index: "+str(gem5_i))
        print("DF:")
        print combined_df
        temp_dict = {}
        for colname in xu3_df.columns.values:
            temp_dict[colname] = xu3_df[colname].iloc[xu3_i]
        for colname in filtered_df.columns.values:
            temp_dict[colname] = filtered_df[colname].iloc[0]
        print("Length of temp list: "+str(len(temp_list)))
        print ("number of keys in dict: "+str(len(temp_dict)))
        import collections
        print("fuplicates in list:")
        print [item for item, count in collections.Counter(temp_list).items() if count > 1]
        combined_df = combined_df.append(temp_dict, ignore_index=True)
        #combined_df = combined_df.append([xu3_df.iloc[xu3_i],filtered_df.iloc[0]], ignore_index=True)
    print combined_df
    return combined_df
        
    
if __name__=='__main__':
    import argparse
    import os
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--gem5-csv',  dest='gem5_input_csv', required=True, \
               help="The combined gem5out csv file")
    parser.add_argument('--xu3-csv',  dest='xu3_input_csv', required=True, \
               help="The combined gem5out csv file")
    parser.add_argument('--model',  dest='model', required=True, \
               help="Only uses gem5 stats from one model,e.g. 'bko'")
    args=parser.parse_args()
    gem5_input_df = pd.read_csv(args.gem5_input_csv,sep='\t')
    xu3_input_df = pd.read_csv(args.xu3_input_csv,sep='\t')

    # rename headers
    gem5_input_df.columns = [ "".join([gem5_col_prefix, str(x)]) for x in gem5_input_df.columns.values]
    xu3_input_df.columns = [ "".join([xu3_col_prefix, str(x)]) for x in xu3_input_df.columns.values]
    # combine
    combined_df = combine_xu3_and_gem5(xu3_input_df, gem5_input_df, args.model)
    # rename the columns so the formulae works
    # apply formulae
    combined_df.to_csv('results.csv',sep='\t')

    
