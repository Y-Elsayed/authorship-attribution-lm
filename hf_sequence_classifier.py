from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
from authors_dataset import AuthorsDataset
from transformers import DataCollatorWithPadding
import os

class SequenceClassifier:
    def __init__(self, num_labels, model_name="bert-base-uncased",max_length=512,output_dir='./results'):
        self.num_labels = num_labels
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.output_dir = output_dir

    def train(self, authors_data, epochs=3,batch_size=16, save_model = True):

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
                if isinstance(sample, list): 
                    sample = " ".join(sample)
                train_texts.append(sample)
                train_labels.append(self.label2id[author])

        train_dataset = AuthorsDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="no"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        trainer.train()
        if save_model:
            os.makedirs(self.output_dir, exist_ok=True)
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            print(f"Model and tokenizer saved to {self.output_dir}")

    def classify(self, sample):
        inputs = self.tokenizer(sample, padding=True, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        pred= outputs.logits.argmax(dim=1).tolist()
        # pred_label = self.id2label.get(pred[0], "UNKNOWN_LABEL")
        # print(f"Predicted author: {pred_label}") # Debugging
        return [self.id2label.get(p, "UNKNOWN_LABEL") for p in pred]
    
    def evaluate_devset(self, dev_data, show_accuracy = False):
        print("Results on dev set:")
        for author, samples in dev_data.items():
            all_accuracies = []
            correct = 0
            total = len(samples)
            if total == 0: 
                print(f"Skipping {author} (no test samples available)")
                continue
            for sample in samples:
                if self.classify(sample) == author:
                    correct+=1
            accuracy = correct/total
            if show_accuracy:
                print(f"{author} \t {accuracy*100:.2f}% correct")
            all_accuracies.append(accuracy)
        return sum(all_accuracies)/len(all_accuracies) if all_accuracies else 0

    def predict(self, test_data,save_predictions = True):
        predictions = []
        for sample in test_data:
            predictions.append(self.classify(sample))
        if save_predictions:
            self.__save_predictions(predictions)
        return predictions

    
    def __save_predictions(self,predictions, output_file = "predictions.txt"):
        with open(output_file, "w") as f:
            for pred in predictions:
                f.write(f"{pred}\n")
