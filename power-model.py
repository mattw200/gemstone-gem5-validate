#!/usr/bin/env python

# Matthew J. Walker
# Created: 19 August 2017

prefix = 'power model '

def run_model(data_df, params_df, map_dict, individual_components=True, compare_with=None):
    import math
    print(params_df['Name'])
    print(params_df['Value'])
    df = data_df
    power = 0
    for i in range(0, len(params_df.index)):
        param_name = params_df['Name'].iloc[i]
        param_value = float(params_df['Value'].iloc[i])
        param_vars = param_name.split(':')
        formula_string = " * ".join(["("+str(map_dict[x])+")" for x in param_vars])+" * "+str(param_value)
        #print param_name+" = "+formula_string  
        this_val_power =  pd.eval(formula_string,engine='python')
        if individual_components:
            df[prefix+param_name+' power'] = this_val_power
        power += this_val_power
    df[prefix+'total power'] = power
    if compare_with:
        df[prefix+'vs '+compare_with+' signed err'] = ((df[compare_with] - power)/power)*100.0
        df[prefix+'vs '+compare_with+' MAPE'] = ((df[compare_with] - power).abs()/power)*100.0
        print("mean signed error: "+str(df[prefix+'vs '+compare_with+' signed err'].mean()))
        print("mean MAPE: "+str(df[prefix+'vs '+compare_with+' MAPE'].mean()))
        

def map_dict_from_file(path):
    lines = []
    with open(path,'r') as f:
        lines = f.read().split('\n')
    f.closed
    map_dict = {}
    for l in lines:
        fields = l.split('\t')
        if len(fields) != 2:
            continue
        map_dict[fields[0].strip()] = fields[1].strip()
    return map_dict
        
if __name__=='__main__':
    import argparse
    import os
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--params',  dest='model_params', required=True, \
               help="The parameters file for the model to use")
    parser.add_argument('-i', '--input-data',  dest='input_data', required=True, \
               help="The data")
    parser.add_argument('-m', '--var-map',  dest='var_map', required=True, \
               help="Maps the names of the variables in the data to the model parameters")
    args=parser.parse_args()
    data_df = pd.read_csv(args.input_data,sep='\t')
    params_df = pd.read_csv(args.model_params,sep='\t')
    run_model(data_df, params_df, map_dict_from_file(args.var_map),compare_with='Power A15')
    #print data_df
