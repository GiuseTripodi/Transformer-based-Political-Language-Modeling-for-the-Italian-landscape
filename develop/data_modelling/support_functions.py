"""
This script contains the support functions used during the data
modelling step. Most of the functions are used both for electoral programs
and tweets, these functions are described below.
"""
import json
import os
import csv
import re
import nltk
nltk.download('stopwords')
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import heapq
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from transformers import pipeline
from nltk.stem import WordNetLemmatizer

def load_speech_tweets_csv(path: str) -> dict:
    """
    Load the json file from path, and return it.
    Used to load the speeches and the tweets
     from json file after data extraction
    :param path:
    :return:
    """
    with open(path) as file:
        ret = csv.DictReader(file)
    return ret

def load_programs_json(path: str) -> dict:
    """
    Load the json file from path, and return it. Used to load the electoral
    programs from json file after data extraction
    :param path:
    :return: file as a dict
    """
    with open(path) as file:
        ret = json.load(file)
    return ret

def save(output_path: str, dataframe: pd.DataFrame):
    """
    The function takes a dataframe and save it in the output path file
    :param output_path:
    :param dataframe:
    :return:
    """
    with open(output_path, "w", encoding="UTF8", newline="") as f:
        dataframe.to_csv(f)


def remove_unwanted_char_function(text: str) -> str:
    """
    removes unwanted char, unwanted text and puts everything on lowercase
    :param text:
    :return: text after preprocessing
    """
    # remove the link inside the text
    text = re.sub(r'http\S+', '', text)

    # Remove all the special characters
    text = re.sub(r'\W', ' ', text)

    # remove all single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # Remove single characters from the start
    text = re.sub(r'\^[a-zA-Z]\s+', ' ', text)

    # Substituting multiple spaces with single space
    text = re.sub(r'\s+', ' ', text, flags=re.I)

    # Removing prefixed 'b'
    text = re.sub(r'^b\s+', '', text)

    # Converting to Lowercase
    text = text.lower()
    return text


def summarize(text: str) -> str:
    """
    gets a text and if it is longer than 300 words summarize it
    :param text:
    :return:
    """
    # since the text can only summarize block of 512 words i split the text and summarize the block by itself
    summarized_text = ""
    phrases = text.split(".")
    i = 0  # phrases index
    while i < len(phrases):
        chunk = ""
        while len(chunk.split()) < 300 and i < len(phrases):
            chunk = chunk + phrases[i]
            i += 1
        # the chunk has the right number of words, we can summarize it
        try:
            summarized_text += newsum(chunk)[0]["summary_text"].strip()
        except Exception as e:
            print(f"Error while summarizing chunk: {chunk}")
            print(e)
    # all the phrases has been summarized, now summarize the whole text

    try:
        ret = newsum(summarized_text, min_length=120, max_length=240)[0]["summary_text"].strip()
    except Exception as e:
        print(f"Error while summarizing text: {summarized_text}")
        print(e)
    return ret

    # return summarized_text


def extractive_summarization(text: str) -> str:
    """
    Gets a text and return an extractive summarization of the text

    code from: https://stackabuse.com/text-summarization-with-nltk-in-python/
    """
    # preprocessing
    # Removing Square Brackets and Extra Spaces
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # converting text to sentences
    sentence_list = sent_tokenize(text, language="italian")

    # Find Weighted Frequency of Occurrence
    stopwords = nltk.corpus.stopwords.words('italian')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    # divide the number of occurances of all the words by the frequency of the most occurring word
    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    # Calculating Sentence Scores
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    # retrieves top 4 sentences  and return it
    summary_sentences = heapq.nlargest(4, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

def lemmatizer(text: str) -> str:
    """
    Takes the text and does the stemming of the text, to reduce each words to
    the relative radix
    :param text:
    :return: string with the text after the stemming
    """
    lemmatizer = WordNetLemmatizer()
    ret = []
    for token in word_tokenize(text):
        ret.append(lemmatizer.lemmatize(token))
    return " ".join(ret)


def stemmer(text: str) -> str:
    """
    Takes the text and does the stemming of the text, to reduce each words to
    the relative radix
    :param text:
    :return: string with the text after the stemming
    """
    snowball = SnowballStemmer(language='english')
    list = []
    for token in word_tokenize(text):
        list.append(snowball.stem(token))
    return ' '.join(list)

def translate_to_en(text: str) -> str:
    """
    takes a text in italian and return the text in english
    :param text: string, italian text
    :return: english test
    """
    #since the text can only translate block of 512 words i split the text and translatte the block by itself
    translated_text = ""
    phrases= text.split(".")
    i = 0 #phrases index
    while i < len(phrases):
        chunk = ""
        while len(chunk.split()) < 300 and i < len(phrases):
            chunk = chunk + phrases[i]
            i += 1
        # the chunk has the right number of words, we can summarize it
        try:
            translated_text += it_en_translator(chunk)[0]["translation_text"].strip()
        except Exception as e:
            print(f"Error while translating chunk: {chunk}")
            print(e)
    return translated_text

def splitter(n, s):
    """
    code from: https://stackoverflow.com/questions/3861674/split-string-by-number-of-words-with-python

    takes a long string s and divides it in bloks of length n
    :param n:
    :param s:
    :return:
    """
    pieces = s.split()
    return (" ".join(pieces[i:i+n]) for i in range(0, len(pieces), n))

def splitter_by_dot(s):
    """
    takes a long string and splits it by the dots without splitting the decimal numbers
    :param s:
    :return:
    """
    #ret = sent_tokenize(s, language="italian")
    ret = re.split('["."][^0-9]', s)
    # preprocess each phrase
    for i in range(len(ret)):
        # remove the link inside the text
        ret[i] = re.sub(r'http\S+', '', ret[i])

        # Substituting multiple spaces with single space
        ret[i] = re.sub(r'\s+', ' ', ret[i], flags=re.I)

        # Removing prefixed 'b'
        ret[i] = re.sub(r'^b\s+', '', ret[i])

        # Converting to Lowercase
        ret[i] = ret[i].lower()
    return ret

def decompose_dataframe_by_text(df: pd.DataFrame, text_field_name:str, phrase_length:int) -> pd.DataFrame:
    """
    code from: https://stackoverflow.com/questions/3861674/split-string-by-number-of-words-with-python

    Takes a dataframe and return the same dataframe decomposed by text, the text is divided
    by phrase and for each phrase is created a new row.
    :param phrase_length: number of words of the phrase you want to extract
    :param text_field_name:
    :param df:
    :return:
    """
    ret = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():

        # split the phrase by the number of words
        #texts = splitter(phrase_length, row[text_field_name])

        # split the phrase by dots
        #texts = splitter_by_dot(row[text_field_name])
        texts = nltk.sent_tokenize(row[text_field_name], "italian")
        for text in texts:
            # splits the phrase too long
            if len(text.split()) > 500:
                short_phrases = splitter(250, text)
                for short in short_phrases:
                    new_row = row
                    new_row[text_field_name] = short
                    ret = ret.append(new_row, ignore_index=True)

            # consider only the phrase with more than 10 words
            elif len(text.split()) >= 10:
                new_row = row
                new_row[text_field_name] = text
                ret = ret.append(new_row, ignore_index=True)

    return ret


def preprocess_text(text: str, summarize_text=True, translate_text=True, remove_unwanted_char=True, lemmatization=True) -> str:
    """
    Preprocess the text in the electoral program, delete unwanted char.
    Then it does the lemmarization
    Then it reduces the size of the text by doin
    :param translate_text:
    :param summarize_text:
    :param text:
    :return: text after preprocessing
    """
    print(f"preprocess text: {text[:10]}...")
    try:
        # summarize italian text
        if summarize_text:
            # text = summarize(text) # abstractive summarization
            text = extractive_summarization(text)  # extractive summarization

        # Translate text to en
        if translate_text:
            text = translate_to_en(text)

        # Remove unwanted characters
        if remove_unwanted_char:
            text = remove_unwanted_char_function(text)

        # lemmatizing text
        if lemmatization:
            text = lemmatizer(text)
        return text
    except Exception as e:
        print(f"Error in text: {text}")
        print(e)
        return None

def add_punctualization(text:str):
    return model.restore_punctuation(text)