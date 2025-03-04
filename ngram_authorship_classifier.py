from nltk.lm import MLE, Laplace
from nltk.lm.models import StupidBackoff, KneserNeyInterpolated
from nltk.util import ngrams
from nltk.lm.preprocessing import padded_ngram_pipeline
from collections import Counter




class NgramAuthorshipClassifier:

    def __init__(self, n = 1, smoothing="lp"):
        self.n = int(n)
        self.smoothing = smoothing
        self.models = {}


    def train(self, authors_data): # not sure if this is the best way to pass the authors data but it will be a map of author to their processed text
        print("Training LMs... (this may take a while)")
        for k, v in authors_data.items():

            train_data, vocab = padded_ngram_pipeline(self.n, v)
            vocab_counter = Counter(vocab)


            if self.smoothing == "sb":
                model = StupidBackoff(order = self.n) 
            elif self.smoothing == "lp":
                model = Laplace(order = self.n) 
            elif self.smoothing == "kn":
                model = KneserNeyInterpolated(order=self.n)
            else:
                if self.smoothing != "mle":
                    print(f"The {self.smoothing} smoothing is not supported. Defaulting to MLE")
                model = MLE(self.n)
            
            model.fit(train_data, vocab_counter)
            self.models[k] = model


    def classify(self, sample):
        ngrams_lst = list(ngrams(sample, self.n))

        perplexities = {}
        for author, model in self.models.items():
            try:
                perplexities[author] = model.perplexity(ngrams_lst)
            except ZeroDivisionError: # I added this the zero division error that occured when the perplexity was zero
                perplexities[author] = float('inf')

        return min(perplexities, key=perplexities.get)

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
        return sum(all_accuracies)/len(all_accuracies) # Returning the average accuracy of all authors to use in the notebook to choose which models and ngrams to use

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

    def __generate_text(self, author, num_words=20):
        model = self.models[author]
        return " ".join(model.generate(num_words))

    def generate_authors_text(self, prompts, num_words=20):
        generated_texts = {}

        for author, prompt in prompts.items():
            generated_text = self.__generate_text(author, num_words)
            generated_texts[author] = f"{prompt} {generated_text}"

        return generated_texts 
    