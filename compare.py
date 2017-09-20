#!/usr/bin/env python
import pandas

def mape(actual, predicted):
    return ((actual - predicted)/actual).abs()*100.0

def mpe(actual,predicted):
    return ((actual - predicted)/actual)*100.0

def wape(actual, predicted):
    return  (((actual - predicted).abs()).sum() / (actual.sum()))*100.0

def noramlise(df, value, value_col):
    return (value - df[value_col].min()) / (df[value_col].max() - df[value_col].min())

def smape(actual, predicted):
    return  ((predicted - actual).abs() / ((actual.abs() + predicted.abs())/2))*100


def smpe(actual, predicted):
    return  ((actual - predicted) / ((actual.abs() + predicted.abs())/2))*100
