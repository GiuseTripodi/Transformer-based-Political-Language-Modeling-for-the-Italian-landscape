"""
This script allows you to join together all the csv files in a folder and return as single csv file
"""
import os
from os import listdir
from os.path import isfile, join
import pandas as pd

def preprocess_dataframe_speech(df: pd.DataFrame):
    """
    The method takes a dataframe and does some preprocessin in order to delete the useless rows
    :param df:
    :return:
    """
    #df.drop(df[df.transcript.str.split().str.len() < 10].index, inplace=True) # for speeches
    df.drop(df[df.text.str.split().str.len() < 10].index, inplace=True) # fpr tweets
    

def join_csv(input_dir, output_dir, output_name):
    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    li = []
    for file in files:
        df = pd.read_csv(join(input_dir, file), index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    # preprocess dataframe
    preprocess_dataframe_speech(frame)

    #save the output as a csv file
    frame.to_csv(join(output_dir, output_name))


if __name__ == '__main__':
    HOME = ""
    output_dir = f"{HOME}/data/output_data_modelling/it_output"

    # TWEETS
    input_dir  = f"{HOME}/data/output_data_modelling/it_output/tweets_pol"
    output_name = f"political_tweets.csv"

    # SPEECHES
    #input_dir  = f"{HOME}/data/output_data_modelling/it_output/speeches_pol_afther_adding_punctualization"
    #output_name = f"political_speech_after_punctualization.csv"

    join_csv(input_dir, output_dir, output_name)