"""
Custom pipeline for text classification, usage example:
>>> pipe = pipeline(model="roberta-large-mnli")
>>> pipe("This restaurant is awesome")
"""
from transformers import Pipeline, TextClassificationPipeline
import numpy as np

def softmax(outputs):
    maxes = np.max(outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)



class MyTextClassificationPipeline(TextClassificationPipeline):
    """
    Custom text classification pipeline
    """
    def _sanitize_parameters(self, **kwargs):
        """
        Checks the parameters passed. Returns three dict of kwargs
        that will be passed to preprocess, _forward and postprocess.
        :param kwargs: 
        :return: 
        """""
        return {}, {}, {}

    def preprocess(self, inputs):
        """
        Takes the input and turn it into something feedable to the model
        :param inputs:
        :param maybe_arg:
        :return:
        """
        return self.tokenizer(inputs, return_tensors=self.framework)

    def _forward(self, model_inputs):
        """
        Forward step
        :param model_inputs:
        :return:
        """
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        """
        Turns the forward step output into the final output
        :param model_outputs:
        :return:
        """
        logits = model_outputs.logits[0].numpy()
        probabilities = softmax(logits)

        best_class = np.argmax(probabilities)
        label = self.model.config.id2label[best_class]
        score = probabilities[best_class].item()
        logits = logits.tolist()
        return {"label": label, "best_class_code": best_class, "score": score, "logits": logits}