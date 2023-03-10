from datetime import date
from sentence_transformers import SentenceTransformer, util
import pandas as pd

today = date.today()
today = today.strftime("%b-%d-%Y")

class SentenceSimilarity:
    """
    A class used to compute the similarity between two sentences
    """

    def __init__(self, sentence_transformer: str):
        """
        Parameters
        -------------
        :parm sentence_transformer: str
            sentence transformer to use to compute the similarity
        """
        # initialize the sentence transformer
        self.model = SentenceTransformer(sentence_transformer)

    def similarity(self, sentences1: [], sentences2: []):
        """
        Does and return the similarity between the sentences
        :return:
        """
        # compute embedding for both texts
        embedding_text1 = self.model.encode(sentences1, convert_to_tensor=True)
        embedding_text2 = self.model.encode(sentences2, convert_to_tensor=True)

        # compute the similarity
        return util.pytorch_cos_sim(embedding_text1, embedding_text2)


def programs_similarity(df_programs: pd.DataFrame, ss: SentenceSimilarity, program1: str, program2: str,arguments: []) -> float:
    """
    The function calculates the similarity between the arguments of two election programs
    :param df_programs: pd.DataFrame
        electoral programs (rows) divided by arguments (columns)
    :param ss: class SentenceSimilarity
        computes the similarity between two sentences
    :param program1: str
        name of the first electoral program
    :param program2: str
        name of the second electoral programs
    :param arguments:
    :return:
    """
    if program1 == program2:
        return 1
    sentence_program1 = df_programs.loc[program1][arguments].values
    sentence_program2 = df_programs.loc[program2][arguments].values
    cosine_scores = ss.similarity(sentence_program1, sentence_program2)
    sum = 0
    for i in range(len(sentence_program1)):
        sum += cosine_scores[i][i].item()
    return sum / len(arguments)

def similarity_matrix(df_programs: pd.DataFrame, arguments: [], model='all-MiniLM-L6-v2') -> pd.DataFrame:
    """
    The method computes the similarity matrix between every program and return it as a dataframe.
    The similarity is computed between every program,
    but it is done by considering only the argument in the array arguments.

    if arguments = ["Lavoro", "Diritti"], it is computed the similarity between the program of every politician but only
    considering the two indicated argument. So it is done the mean between the two results.
    """
    matrix = []
    ss = SentenceSimilarity(model)
    for program in df_programs.index:
        similarity_program = []
        for program2 in df_programs.index:
            sim = programs_similarity(df_programs, ss, program, program2, arguments)
            similarity_program.append(sim)
        matrix.append(similarity_program)
    return pd.DataFrame(matrix, index=df_programs.index, columns=df_programs.index)
