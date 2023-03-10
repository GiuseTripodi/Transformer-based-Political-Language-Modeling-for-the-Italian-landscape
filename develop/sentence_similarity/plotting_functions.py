from matplotlib import pyplot as plt
from nltk.lm import preprocessing
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay

from develop.sentence_similarity.similarity import today, similarity_matrix


def plot_similarity_matrix(similarity_matrix, title:str):
    """
    Plots the general similarity matrix
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=similarity_matrix.to_numpy(),  display_labels=similarity_matrix.index)

    disp.plot(cmap=plt.cm.Blues)
    disp.ax_.set_title(title, fontsize=15)
    disp.figure_.set_figwidth(13)
    disp.figure_.set_figheight(10)
    plt.savefig(f"{title}_{today}.png")

def plot_similarity_matrix_by_argument(df_programs:pd.DataFrame, arguments: [], model:str):
    """
    plots the similarity matrix separately for every argument in arguments
    """
    for argument in arguments:
        cm = similarity_matrix(df_programs, [argument] , model)
        plot_similarity_matrix(cm, f"{argument} Similarity")


def plot_scatter_plot(df_programs: pd.DataFrame, model: str):
    """
    compute the scatter plot of the embedding of the full program for every politician
    """
    model = SentenceTransformer(model)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
    df_programs["program"] = df_programs[df_programs.columns].apply("_".join, axis=1)
    le = preprocessing.LabelEncoder()
    title = "Embedding Scatter Plot"

    # create embedding for arguments
    embedding = []
    labels = []
    columns = []
    for column in df_programs.columns:
        for index in df_programs.index:
            text = df_programs.loc[index][column]
            embedding.append(model.encode(text, convert_to_tensor=False, show_progress_bar=False))
            labels.append(index)
            columns.append(column)
    # create the dataframe with the embedding
    embedding = pd.DataFrame(embedding)
    embedding = tsne.fit_transform(embedding)
    embedding = pd.DataFrame(embedding, columns=["X", "Y"])
    embedding["labels"] = labels
    embedding["category"] = columns

    map_labels = {
        "program": "o",
        'Economia e Giustizia': "v",
        'Lavoro': "1",
        'Cultura e Turismo': ">",
        'Scuola, Università e Ricerca': "s",
        'Esteri': "+",
        'Ambiente': "x",
        'Salute': "D",
        'Infrastrutture': "X",
        'Diritti': "3"
    }
    map_colors = {
        'FratelliDItalia': 'tab:blue',
        'Movimento5Stelle': 'tab:orange',
        'TerzoPolo': 'tab:green',
        'Lega': '#9467bd',
        'PD': 'tab:purple',
        'ForzaItalia': 'tab:brown'
    }

    N = len(df_programs.index)
    x = embedding["X"]
    y = embedding["Y"]
    colors = embedding["labels"].map(map_colors).values
    markers = embedding["category"].map(map_labels).values
    area = [300 if embedding.loc[i]["category"] != "program" else 3000 for i in embedding.index]
    fig = plt.figure(figsize=(15, 10))
    plot_lines = []
    for i in range(len(embedding.index)):
        l = plt.scatter(x[i], y[i], s=area[i], c=colors[i], marker=markers[i], alpha=0.5)
        plot_lines.append([l, embedding["category"][i]])
    plot_lines = pd.DataFrame(plot_lines, columns=["lines", "category"])
    legend1 = plt.legend(plot_lines["lines"].values[0:-6:6], plot_lines["category"].values[0:-6:6], markerscale=0.5,
                         loc='upper left', bbox_to_anchor=(1, 1))
    plt.gca().add_artist(legend1)
    plt.legend(plot_lines["lines"].values[-6:], df_programs.index, markerscale=0.2, bbox_to_anchor=(1, 0.5),
               loc='upper left')
    plt.title(title, fontsize=20)
    plt.savefig(f"{title}_{today}.png")

    plt.show()