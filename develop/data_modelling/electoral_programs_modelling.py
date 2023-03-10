"""
This script allows you to load election program data. Perform preprocessing.
Create a suitable data structure and load all the information into it.
After that save the structure to a data file that will be taken by the HuggingFace tokenizer.

"""

import os
import pandas as pd
from develop.data_modelling.support_functions import decompose_dataframe_by_text, preprocess_text


def make_structure(PATH: str, screen_name: [], summarize_text=True, translate_text=True) -> []:
    """
    Take all the electoral programs and create the final structure with all the needed information
    to run the huggingFace model. The programs are divided in phrases and each phrase is
    put in a final structure
    :param PATH: general path where are located all the programs
    :param screen_name: array with the names of the programs file
    :return: array
    """
    ret = []
    for program_name in screen_name:
        print(f"Making structure for {program_name}")
        # take the class name
        class_name = program_name[:program_name.index(".")]
        # load the program
        #content = load_programs_json(os.path.join(PATH, program_name))
        content = pd.DataFrame.from_dict(content,  orient='index', columns=['text'])

        # preprocessing text column
        #content["text"] = content["text"].apply(lambda x: preprocess_text(x, summarize_text=summarize_text, translate_text=translate_text))

        # add label column
        content.insert(len(content.columns), "label", class_name)
        print(f"{content.info()}\n")
        ret.append(content)

    return pd.concat(ret)


def make_structures_topic_similarity(PATH: str, screen_names: [], summarize_text=True, translate_text=True):
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
    for screen_name in screen_names:
        print(f"Retrieving content from {screen_name} file\n")

        # load the content, the content could be either the speeches program or the tweets
        content = pd.read_csv(os.path.join(PATH, screen_name))

        #preprocessing the columns
        #for col in content.columns:
            #content[col] = content[col].apply(lambda x : preprocess_text(x, summarize_text=summarize_text, translate_text=translate_text))


        print(f"{content.info()}\n")
        # save the dataframe
        content.to_csv(f"./{screen_name}", index=False)

def make_structures_by_category_text_class_2(PATH: str, screen_names: [], output_path, summarize_text, translate_text, remove_unwanted_char, lemmatization):
    """
    takes the file indicated by the screen names and put all the contents in a single dataframe. After this it just
    adds the label indicator. Generates the test set for the text classification 2
    :param translate_text:
    :param summarize_text:
    :param text_field_name: name of the field of the text in the csv files
    :param screen_names: array with the names of the programs file
    :param PATH: general path where are located all the programs
    :return: array
    """
    li = []
    indexes = []
    for screen_name in screen_names:
        print(f"Retrieving content from {screen_name} file\n")
        content = pd.read_csv(os.path.join(PATH, screen_name), index_col=None,  header=0)
        # take the class name
        class_name = screen_name[:screen_name.index(".")]
        li.append(content)
        indexes.append(class_name)
    df = pd.concat(li, axis=0, ignore_index=True)
    df["index"] = indexes
    df.set_index("index", inplace=True)

    # Transpose the dataset
    tmp = []
    for column in df.columns:
        for index in df.index:
            tmp.append({"text":df.loc[index][column], "label":index, "category":column})

    df = pd.DataFrame(tmp)

    # divide the text in more phrase
    df = decompose_dataframe_by_text(df, "text", 50)

    # preprocessing text column
    df["text"] = df["text"].apply(lambda x : preprocess_text(x, summarize_text=summarize_text, translate_text=translate_text, remove_unwanted_char=remove_unwanted_char, lemmatization=lemmatization))

    # delete the null rows
    df = df.dropna(axis=0, subset=["text"])
    # save the dataframe
    df.to_csv(f"{output_path}/programs_by_index_by_nltk.csv", index=False)

if __name__ == '__main__':
    path = ""
    output_path = ""
    screen_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    make_structures_by_category_text_class_2(path, screen_names, output_path, summarize_text=False, translate_text=False, remove_unwanted_char=True, lemmatization=False)
