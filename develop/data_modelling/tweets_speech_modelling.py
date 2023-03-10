"""
This script allows tweets and
transcripts to be uploaded and processed to obtain a type of data that is perfectly usable by transformers.

"""
import os
import pandas as pd
from develop.data_modelling.support_functions import preprocess_text, load_programs_json, save, \
    decompose_dataframe_by_text, add_punctualization


def make_structure(PATH: str, screen_names: [], text_field_name: str, summarize_text=True, translate_text=True,
                   remove_unwanted_char=True, lemmatization=True) -> []:
    """
    takes the file indicated by the screen names and put all the contents in a single dataframe. After this it just
    adds the label indicator. This process is done for every single screen name
    :param translate_text:
    :param summarize_text:
    :param text_field_name: name of the field of the text in the csv files
    :param screen_names: array with the names of the programs file
    :param PATH: general path where are located all the programs
    :return: array
    """
    ret = []
    for pol_name in screen_names:
        print(f"Retrieving content from {pol_name} file\n")
        # take the class name
        class_name = pol_name[:pol_name.index(".")]
        # load the content, the content could be either the speeches program or the tweets
        content = pd.read_csv(os.path.join(PATH, pol_name))

        # add punctualization divide the transcript in more phrases (Only for speeches)
        # content[text_field_name] = content[text_field_name].apply(lambda x: add_punctualization(x))
        # content = decompose_dataframe_by_text(content, text_field_name, phrase_length=50)

        # preprocessing text column
        content[text_field_name] = content[text_field_name].apply(
            lambda x: preprocess_text(x, summarize_text=summarize_text, translate_text=translate_text,
                                      remove_unwanted_char=remove_unwanted_char, lemmatization=lemmatization))

        # add label column
        content.insert(len(content.columns), "label", class_name)
        print(f"{content.info()}\n")
        ret.append(content)

    return pd.concat(ret)

def speech_modelling(HOME: str):
    # SPEECHES MODELLING

    path_speech = f"{HOME_PATH}/data/politician_speech/text"
    screen_names_speech = [f for f in os.listdir(path_speech) if os.path.isfile(os.path.join(path_speech, f))][0]

    print("Make Structure for speeches\n")
    speech = make_structure(path_speech, screen_names_speech, "transcript")

    # save the speeches
    output = f"{HOME_PATH}/data/train_test_val_csv"
    filename_speech = "political_speech_v2.csv"
    save(os.path.join(output, filename_speech), speech, speech.columns)


def tweets_modelling(HOME):
    # TWEETS MODELLING

    path_tweet = f"{HOME_PATH}/data/retrieved_tweets"
    screen_names_tweets = ["SilvioBerlusconi.csv", "CarloCalenda.csv", "MatteoRenzi.csv", "GiuseppeConte.csv","EnricoLetta.csv", "GiorgiaMeloni.csv","matteosalvinimi.csv"]

    print("Make Structure for tweets\n")
    tweets = make_structure(path_tweet, screen_names_tweets, "text", summarize_text=False)

    #save the tweets
    output = f"{HOME_PATH}/data/train_test_val_csv"
    filename_tweets = "political_tweets.csv"
    save(os.path.join(output, filename_tweets), tweets)


if __name__ == '__main__':
    HOME_PATH = ""
    speech_modelling(HOME=HOME_PATH)





