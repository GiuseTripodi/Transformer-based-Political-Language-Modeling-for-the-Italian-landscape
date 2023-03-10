"""
This script allows you to fine tune every model on the
right data defined. You just need to add the checkpoint and
the script run a trained job to fine-tune the model itself.
"""
import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from sklearn import preprocessing
import numpy as np
import evaluate
from transformers.integrations import TensorBoardCallback

#CONSTANT
NUM_LABELS = 7


class FineTuneModel:
    def __init__(self, checkpoint, tags, dataset, output_dir, batch_size=32, learning_rate=2e-5, num_epochs=5):
        self.checkpoint = checkpoint
        self.dataset = dataset
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.tags = tags

        # define the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.label_encoder = preprocessing.LabelEncoder()
        # get the mapping id to label
        self.label_encoder.fit(self.dataset["train"]["label"])
        self.label2id = None
        self.id2label = None

        # load metric
        self.metric = evaluate.load("accuracy", "precision")

    def compute_id2labels(self):
        """
        Computes the variable id2label and label2id used during the training
        :return:
        """
        transformed_array = self.label_encoder.transform(self.label_encoder.classes_)
        # convert the id from int64 to int
        id_ = [transformed_array[i].item() for i in range(len(transformed_array))]
        self.label2id = dict(zip(self.label_encoder.classes_, id_))
        self.id2label = dict(zip(id_, self.label_encoder.classes_))

    def preprocess_function(self, examples):
        """
        Preprocessing function, tokenize text and truncate sequences
        to be no longer than model’s maximum input length.

        :return:
        """
        # transform non-numerical labels to numerical labels.
        examples["label"] = self.label_encoder.transform(examples["label"])
        return self.tokenizer(examples["text"], truncation=True, padding=True)

    def compute_metrics(self, eval_pred):
        """
        Computes the metrics between the labels and the output of the predictions
        :param eval_pred:
        :return:
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels)

    def fine_tune(self):
        # Preprocess
        self.compute_id2labels()

        encoded_dataset = self.dataset.map(self.preprocess_function, batched=True)
        # define config
        config = AutoConfig.from_pretrained(self.checkpoint, label2id=self.label2id, id2label=self.id2label,
                                            num_labels=NUM_LABELS)
        # load model with config
        model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, config=config,
                                                                   ignore_mismatched_sizes=True)

        # define train argument
        model_name = self.checkpoint.split("/")[-1]

        strategy = "steps"
        interval_steps = 20

        args = TrainingArguments(
            output_dir=self.output_dir + f"/{model_name}-finetuned-{self.tags}",

            save_strategy="epoch",
            evaluation_strategy="epoch",
            logging_strategy=strategy,

            save_steps=interval_steps,
            logging_steps=interval_steps,
            eval_steps=interval_steps,

            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            report_to="wandb",
            weight_decay=0.01,
            learning_rate=self.learning_rate,

            load_best_model_at_end=True
        )

        # define the trainer
        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5), TensorBoardCallback()],
            compute_metrics=self.compute_metrics
        )

        # trainer.train(resume_from_checkpoint="../input/models/bert-base-uncased-finetuned-textClass1/checkpoint-948") # only if continue training from a checkpoint
        trainer.train()

