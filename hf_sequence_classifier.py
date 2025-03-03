from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
from authors_dataset import AuthorsDataset

class SequenceClassifier:
    def __init__(self, num_labels, model_name="bert-base-uncased"):
        self.num_labels = num_labels
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, authors_data, epochs=3,batch_size=16, max_length=512):

        self.label2id = {label: i for i, label in enumerate(authors_data.keys())}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.model = AutoModelForSequenceClassification.from_pretrained(
        self.model_name,
        num_labels=self.num_labels,
        id2label=self.id2label,
        label2id=self.label2id
        ).to(self.device)

        train_texts = []
        train_labels = []

        for author, samples in authors_data.items():
            for sample in samples:
                train_texts.append(sample)
                train_labels.append(self.label2id[author])

        train_dataset = AuthorsDataset(train_texts, train_labels, self.tokenizer, max_length)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

    def classify(self, sample):
        inputs = self.tokenizer(sample, return_tensors="pt", truncation=True, max_length=self.max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        pred= outputs.logits.argmax().item()
        return self.id2label[pred]
