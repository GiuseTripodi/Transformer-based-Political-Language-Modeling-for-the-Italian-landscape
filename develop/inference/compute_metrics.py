import evaluate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date



def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


today = date.today()
today = today.strftime("%b-%d-%Y")
PLOT_PATH = ""


class ComputeMetrics:
    """
    A class used to compute metrics on model output and plot the results.
    ...

    Methods
    ---------
    compute_metrics()
        returns the computed metrics

    plot_consistency_for_politician()
        plots a bar pot of true positive and total prediction

    confusion_matrix_plot()
        plots the confusion matrix

    misclassification_pie_chart()
        draws a pie chart of false positives


    """

    def __init__(self, model_predictions, model_inputs, mapping, tc2=False, tags=""):
        """
        :param model_predictions: dict
            output of the TextClassificationPipeline
        :param model_inputs: DataFrame
            model input
        :param mapping:
            mapping between label and associated id, used to map input labels to ids used by models
        :parma tc2: boolean
            This is true if you perform text classification on election programs. Input labels are different in text classification of election programs.
        """
        self.model_predictions = model_predictions
        self.model_inputs = model_inputs
        self.mapping = mapping
        self.tags = tags
        self.tc2 = tc2

        # DEFINE Y_PRED AND Y_TRUE
        self.references_labels = self.model_inputs["label"].map(mapping).tolist()  # y_true
        if self.tc2:
            #  If tc2, the labels must be changed because "CarloCalenda" and "MatteoRenzi" have the same labels on the test set.
            mapping_prediction_label = {
                "CarloCalenda": self.mapping["TerzoPolo"],
                "EnricoLetta": self.mapping["PD"],
                "GiorgiaMeloni": self.mapping["FratelliDItalia"],
                "GiuseppeConte": self.mapping["Movimento5Stelle"],
                "MatteoRenzi": self.mapping["TerzoPolo"],
                "MatteoSalvini": self.mapping["Lega"],
                "SilvioBerlusconi": self.mapping["ForzaItalia"]
            }
            self.predictions_labels = pd.DataFrame(self.model_predictions)["label"].map(
                mapping_prediction_label).tolist()  # y_pred
        else:
            self.predictions_labels = pd.DataFrame(self.model_predictions)["best_class_code"].tolist()  # y_pred

    def compute_metrics(self):
        """
        Prints the values of: Accuracy, F1, precision and recall
        """
        # load and define the different metrics
        accuracy = evaluate.load('accuracy')
        f1 = evaluate.load('f1', average='macro')
        precision = evaluate.load('precision')
        recall = evaluate.load('recall', average='macro')
        roc_auc_score = evaluate.load("roc_auc", "multiclass")

        # print metrics
        print(accuracy.compute(predictions=self.predictions_labels, references=self.references_labels))
        print(f1.compute(predictions=self.predictions_labels, references=self.references_labels, average='weighted'))
        print(precision.compute(predictions=self.predictions_labels, references=self.references_labels,
                                average='weighted'))
        print(
            recall.compute(predictions=self.predictions_labels, references=self.references_labels, average='weighted'))

        # ROC AUC
        pred_scores = pd.DataFrame(self.model_predictions)["logits"].transform(softmax)
        try:
            print(self.roc_auc_score.compute(references=self.references_labels, prediction_scores=pred_scores,
                                             multi_class='ovr', labels=[0, 1, 2, 3, 4, 5, 6]))
        except:
            pass

    def plot_consistency_for_politician(self):
        """
        plots a bar pot of true positive and total prediction number
        """
        # compute the confusion matrix
        matrix = confusion_matrix(self.references_labels, self.predictions_labels,
                                  labels=np.arange(len(self.mapping.keys())))
        # takes only the TP
        diagonal = matrix.diagonal()
        # takes the number of predictions
        tot_ele = []
        for i in range(len(matrix)):
            tot_ele.append(sum(matrix[i]))

        # plot the results
        politician = self.mapping.keys()
        X_axis = np.arange(len(politician))

        fig = plt.figure(figsize=(10, 5))
        # creating the bar plot
        plt.bar(X_axis - 0.2, diagonal, color="maroon", width=0.4, label="Correct predictions")
        plt.bar(X_axis + 0.2, tot_ele, color="#E5BABA", width=0.4, label="Total number of predictions")

        plt.xticks(X_axis, politician)
        plt.xlabel("Italian Politician")
        plt.ylabel("number of predictions")
        plt.title("Italian Politician Accuracy", fontsize=12)
        plt.legend()
        plt.savefig(f"{PLOT_PATH}/accuracy_for_politician_{'tc2' if self.tc2 else 'tc1'}_{self.tags}_{today}.png")

    def confusion_matrix_plot(self):
        """
        Plots the confusion matrix
        """
        disp = ConfusionMatrixDisplay.from_predictions(y_true=self.references_labels, y_pred=self.predictions_labels,
                                                       labels=np.arange(len(self.mapping.keys())),
                                                       display_labels=list(self.mapping.keys()), cmap=plt.cm.Blues)
        fig = disp.ax_.get_figure()
        fig.set_figwidth(15)
        fig.set_figheight(10)
        plt.title("Confusion Matrix", fontsize=14)
        plt.savefig(f"{PLOT_PATH}/confusion_matrix_{'tc2' if self.tc2 else 'tc1'}_{self.tags}_{today}.png")

    def misclassification_pie_chart(self):
        y_pred = np.array(self.predictions_labels)
        y_true = np.array(self.references_labels)

        # takes only the misclassified element
        y_pred_mis = y_pred[y_pred != y_true]
        y_true_mis = y_true[y_pred != y_true]
        matrix = confusion_matrix(y_true_mis, y_pred_mis, labels=np.arange(len(self.mapping.keys())))
        politician_names = list(self.mapping.keys())

        colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(matrix[0])))
        fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 15))
        fig.tight_layout()
        i = 0
        for ax in axs.ravel():
            if i < len(matrix):
                ax.set_title(politician_names[i], fontsize=15)
                ax.pie(matrix[i], colors=colors,
                       labels=[politician_names[pol_name] if matrix[i][pol_name] != 0 else None for pol_name in
                               range(len(politician_names))])
                i += 1
            else:
                # last pie
                ax.pie([1, 0, 0, 0, 0, 0, 0])
        plt.savefig(f"{PLOT_PATH}/politician_misclassification_{'tc2' if self.tc2 else 'tc1'}_{self.tags}_{today}.png")

