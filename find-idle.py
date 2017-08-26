
#gem5_workload_name = 'sleep 1'
#xu3_workload_name = 'idle'
gem5_workload_name = 'parsec-blackholes-4'
xu3_workload_name = 'parsec-blackholes-4'

if __name__=='__main__':
    import argparse
    import os
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--gem5-combined',  dest='gem5_combined', required=True, \
               help="The gem5 combiend file")
    parser.add_argument('--xu3-combined',  dest='xu3_combined', required=True, \
               help="The xu3 combiend file")
    parser.add_argument('-o',  dest='output', required=True, \
               help="output_dir")
    args=parser.parse_args()
    gem5_df = pd.read_csv(args.gem5_combined,sep='\t')
    xu3_df = pd.read_csv(args.xu3_combined,sep='\t')

    gem5_cols = gem5_df.columns.values
    gem5_df = gem5_df[gem5_df['workload name'] == gem5_workload_name]
    cycles_cols = [x for x in gem5_cols if x.find('.numCycles') > -1]
    inst_spec = [x for x in gem5_cols if x.find('iewExecutedInsts') > -1]
    inst_arch = [x for x in gem5_cols if x.find('.committedInsts') > -1 and x.find('.commit.') == -1]

    if len(cycles_cols) != 8:
        raise ValueError('cycles cols not 8 long')
    if len(inst_spec) != 4:
        raise ValueError('inst spec not 8 long')
    if len(inst_arch) != 8:
        raise ValueError('inst arch not 8 long')

    print('Cycles: '+str(cycles_cols))
    print('Inst spec: '+str(inst_spec))
    print('inst arch: '+str(inst_arch))

    cols_list = gem5_cols[2:11].tolist()+cycles_cols+inst_spec+inst_arch
    gem5_df = gem5_df[cols_list]

    #%gem5_df[gem5_df['core mask'] == '4,5,6,7'].to_csv(os.path.join(args.output,'gem5-idle-compare-output-big.csv'))
    #gem5_df[gem5_df['core mask'] == '0,1,2,3'].to_csv(os.path.join(args.output,'gem5-idle-compare-output-little.csv'))
    gem5_df.to_csv(os.path.join(args.output,'gem5-'+gem5_workload_name.replace(' ','-')+'-compare-output.csv'))

    xu3_df = xu3_df[xu3_df['workload name'] == xu3_workload_name]
    xu3_df = xu3_df[xu3_df['Freq (MHz) C0'] == 1000]
    xu3_df = xu3_df[xu3_df['Freq (MHz) C4'] == 1000]

    xu3_cols = xu3_df.columns.values
    cycles_cols = [x for x in xu3_cols if x.find('cycle count diff') > -1]
    inst_spec = [x for x in xu3_cols if x.find('(0x1B)') > -1]
    inst_arch = [x for x in xu3_cols if x.find('(0x08)') > -1]

    print ("\nCycles: "+str(cycles_cols))
    print ("\nspec: "+str(inst_spec))
    print ("\narch: "+str(inst_arch))

    xu3_df = xu3_df[xu3_cols[2:20].tolist()+cycles_cols+inst_spec+inst_arch]

    xu3_df[xu3_df['core mask'] == '4,5,6,7'].to_csv(os.path.join(args.output,'xu3-'+xu3_workload_name.replace(' ','-')+'-compare-output-big.csv'))
    xu3_df[xu3_df['core mask'] == '0,1,2,3'].to_csv(os.path.join(args.output,'xu3-'+xu3_workload_name.replace(' ','-')+'-compare-output-little.csv'))
