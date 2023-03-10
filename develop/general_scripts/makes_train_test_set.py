"""
The script allows, given the output of the data modelling process, to generate the dataset that will
be used to train, test and validate the models.
Three dataset are going to be generated:
1) Training set -> 80% of speeches and tweets together
    1.1) Validation set -> 20% of the training set
2) Test set ->  20% speeches and tweets together
3) full_training_set -> 100% of speeches and tweets
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import pytz

def read_and_process_speech(path_speech) -> pd.DataFrame:
    df = pd.read_csv(path_speech, index_col=None, header=0)

    #drop useless columns
    df = df.drop(["Unnamed: 0", "tags", "title"], axis=1)

    #rename some columns
    df = df.rename({"id":"video_id", "publishedAt": "created_at", "transcript" : "text"}, axis='columns')

    # change the date format, since it is in ISO format and we want it in UTC format
    df["created_at"] = df["created_at"].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S+00:00"))
    return df


def read_and_process_tweets(path_tweets) -> pd.DataFrame:
    df = pd.read_csv(path_tweets, index_col=None, header=0)

    # drop useless columns
    df = df.drop(["Unnamed: 0.1","Unnamed: 0", "author_id"], axis=1)
    #rename some columns
    df = df.rename({ "like_count": "likeCount", "id": "tweet_id"}, axis='columns')
    # change some labels value
    df["label"] = df["label"].replace(
        {"GiuseppeConteIT_25-09-21_to_25-09-2022": "GiuseppeConte", "berlusconi_25-09-21_to_25-09-2022": "SilvioBerlusconi",
         "matteosalvinimi_25-09-21_to_25-09-2022": "MatteoSalvini", "MatteoRenzi_25-09-21_to_25-09-2022":"MatteoRenzi",
         "CarloCalenda_25-09-21_to_25-09-2022":"CarloCalenda","EnricoLetta_25-09-21_to_25-09-2022":"EnricoLetta",
         "GiorgiaMeloni_25-09-21_to_25-09-2022":"GiorgiaMeloni"})

    return df

def makes_sets(path:str, file_names: [], output_dir: str):
    # takes the csv and generate the dataframe
    datasets = []
    for file in file_names:
        file_path = os.path.join(path, file)
        if "speech" in file:
            df = read_and_process_speech(file_path)
        elif "tweets" in file:
            df = read_and_process_tweets(file_path)


        # print statistics
        print(df.info())
        print("\n")
        datasets.append(df)

    full_dataframe = pd.concat(datasets, axis=0, ignore_index=True).fillna(0)
    full_dataframe[["tweet_id", "viewCount", "commentCount", "retweet_count", "reply_count", "quote_count"]] = full_dataframe[["tweet_id", "viewCount", "commentCount", "retweet_count", "reply_count", "quote_count"]].astype(np.int64)
    print(full_dataframe.info())
    # shuffle the dataframe rows
    full_dataframe = full_dataframe.sample(frac=1)

    # reduce the size, for sample
    #full_dataframe = full_dataframe[0:1000]

    # MAKES DATASET

    # split the dataset in training and test set
    train_set, test_set = train_test_split(full_dataframe, test_size=0.2, random_state=42, shuffle=True)
    # split the train set into train_set and validation set
    train_set_2, val_set = train_test_split(train_set, test_size=0.2, random_state=42, shuffle=True)

    # SAVE DATASET FOR TEXT CLASSIFICATION 1
    where = "text_classification_1/it"
    #where = "text_classification_smaller_dataset_sample/it"
    train_set_2.to_csv(os.path.join(output_dir, f"{where}/train_set.csv"))
    val_set.to_csv(os.path.join(output_dir,  f"{where}/val_set.csv"))
    test_set.to_csv(os.path.join(output_dir,  f"{where}/test_set.csv"))

    # SAVE DATASET FOR TEXT CLASSIFICATION 2

    train_set.to_csv(os.path.join(output_dir, "text_classification_2/it/train_set.csv"))
    test_set.to_csv(os.path.join(output_dir, "text_classification_2/it/val_set.csv"))


if __name__ == '__main__':
    path = ""
    file_names = ["political_speech_after_punctualization.csv", "political_tweets.csv"]
    output_dir = ""
    makes_sets(path, file_names, output_dir)


