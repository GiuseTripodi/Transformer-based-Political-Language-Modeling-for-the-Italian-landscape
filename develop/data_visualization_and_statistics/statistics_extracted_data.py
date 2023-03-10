import nltk
import pandas as pd
import os
import PyPDF2



def medium_number_of_sentences(df: pd.DataFrame, text_column_name):
    number_of_sentences = 0
    for id, row in df.iterrows():
        number_of_sentences += len(nltk.sent_tokenize(row[text_column_name], "italian"))

    return number_of_sentences, round(number_of_sentences / df.shape[0])


def medium_number_of_sentences_programs(text):
    number_of_sentences = 0
    for id, row in df.iterrows():
        number_of_sentences += len(nltk.sent_tokenize(row[text_column_name], "italian"))

    return number_of_sentences, round(number_of_sentences / df.shape[0])


def tweets_statistics():
    path = ""
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in files:
        print(f"File: {file}")
        df = pd.read_csv(os.path.join(path, file))
        # number of tweets (dataframe's rows)
        print(f"number of tweets = {df.shape[0]}")

        # Number of words per politician,
        print(f"Number of words: {round(df['text'].str.split().str.len().sum())}")

        # Mean Number of words per tweets,
        print(f"Sentence Length (number of words) (mean for tweet): {round(df['text'].str.split().str.len().mean())}")

        tot, mean = medium_number_of_sentences(df, "text")
        # Number of Sentences
        print(f"Number of Sentences: {tot}")

        # Mean number of sentences per tweets
        print(f"Mean Number of Sentences (mean for tweet): {mean}")
        print("\n")

        "----  Other metrics based on views and likes"
        print(f"Mean Likes: {df['like_count'].mean()}")
        print(f"Quote mean: {df['quote_count'].mean()}")
        print(f"retweet count mean: {df['retweet_count'].mean()}")
        print(f"reply count mean: {df['reply_count'].mean()}")
        print("\n")


def transcript_statistics():
    path = ""
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in files:
        print(f"File: {file}")
        df = pd.read_csv(os.path.join(path, file))

        # number of transcript (dataframe's rows)
        print(f"number of transcript = {df.shape[0]}")

        # Number of words per politician,
        print(f"Number of words: {round(df['transcript'].str.split().str.len().sum())}")

        # Mean Number of words per tweets,
        print(
            f"Sentence Length (number of words) (mean for transcript): {round(df['transcript'].str.split().str.len().mean())}")

        tot, mean = medium_number_of_sentences(df, "transcript")
        # Number of Sentences
        print(f"Number of Sentences: {tot}")

        # Mean number of sentences per tweets
        print(f"Mean Number of Sentences (mean for transcript): {mean}")
        print("\n")

        "----  Other metrics based on views and likes"
        print(f"Mean Likes: {df['likeCount'].mean()}")
        print(f"View count mean: {df['viewCount'].mean()}")
        print(f"comment count mean: {df['commentCount'].mean()}")
        print("\n")


def electoral_programs_statistics():
    path = ""
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in files:
        print(f"File: {file}")
        # load pfg
        pdf_file = open(os.path.join(path, file), 'rb')
        pdfreader = PyPDF2.PdfFileReader(pdf_file)

        text = ""
        for pageNum in range(pdfreader.numPages):
            pageObj = pdfreader.getPage(pageNum)
            text += pageObj.extractText()

        # Number of words per programs,
        print(f"Number of words: {len(text.split())}")

        # Number of Sentences
        print(f"Number of Sentences: {len(nltk.sent_tokenize(text, 'italian'))}")

        # Sentence Length
        sum = 0
        for sent in nltk.sent_tokenize(text, 'italian'):
            sum += len(sent)
        print(f"Mean words per Sentences : {sum / len(nltk.sent_tokenize(text, 'italian'))}")
        print("\n")


if __name__ == '__main__':
    # tweets_statistics()
    # transcript_statistics()
    electoral_programs_statistics()
