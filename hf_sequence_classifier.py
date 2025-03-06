from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import torch
from authors_dataset import AuthorsDataset
from transformers import DataCollatorWithPadding
import os
import random
from sklearn.model_selection import train_test_split

class SequenceClassifier:
    def __init__(self,  model_name="bert-base-uncased",max_length=512,output_dir='./results'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.output_dir = output_dir

    def train(self, authors_data, epochs=3,batch_size=16, save_model = True):

        self.label2id = {label: i for i, label in enumerate(authors_data.keys())}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.num_labels = len(authors_data.keys()) 

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

        # Shuffle the data
        combined = list(zip(train_texts, train_labels))
        random.shuffle(combined)
        train_texts, train_labels = map(list, zip(*combined))

        train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42
        )

        train_dataset = AuthorsDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = AuthorsDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=8)

        warmup_steps = max(1, int(0.1 * len(train_dataset) / batch_size)) # 10% of train data
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )

        trainer.train()
        if save_model:
            os.makedirs(self.output_dir, exist_ok=True)
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            print(f"Model and tokenizer saved to {self.output_dir}")

    def classify(self, sample):
        try:
            inputs = self.tokenizer(sample, padding=True, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            pred= outputs.logits.argmax(dim=1).tolist()[0]
            return self.id2label.get(pred, "UNKNOWN_LABEL")
        except Exception as e:
            print(f"Error classifying sample: {sample}, error: {e}")
            return "UNKNOWN_LABEL"
    
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
                classified = self.classify(sample)
                if classified == author:
                    correct+=1
                print("predicted:", classified, "actual:", author)
            accuracy = correct/total if total > 0 else -1
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
