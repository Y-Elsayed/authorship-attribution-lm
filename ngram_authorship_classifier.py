from ngram_helper import MLE, Laplace, KneserNeyInterpolated, StupidBackoff, generate_ngrams #Inspired by NLTK framework
from collections import Counter
import random
from nltk.corpus import stopwords
import string


class NgramAuthorshipClassifier:

    def __init__(self, n = 1, smoothing="lp"):
        self.n = int(n)
        self.smoothing = smoothing
        self.models = {}
        self.ngram_frequencies = {}


    def train(self, authors_data): # not sure if this is the best way to pass the authors data but it will be a map of author to their processed text
        print("Training LMs... (this may take a while)")
        for author, texts in authors_data.items():
            train_data = texts
            vocab = set(word for text in train_data for word in text)
            
            if self.smoothing == "lp":
                model = Laplace(self.n)
            elif self.smoothing == "kn":
                model = KneserNeyInterpolated(self.n)
            elif self.smoothing == "sb":
                model = StupidBackoff(self.n)
            else:
                if self.smoothing != "mle":
                    print(f"The {self.smoothing} smoothing is not supported. Defaulting to MLE")
                model = MLE(self.n)
            
            model.fit(train_data, vocab)
            self.models[author] = model
            
            self.ngram_frequencies[author] = self._compute_ngram_frequencies(train_data)


    def classify(self, sample, show_perplexity = False):
        ngrams_lst = list(generate_ngrams(sample, self.n))

        perplexities = {}
        for author, model in self.models.items():
            try:
                perplexities[author] = model.perplexity(ngrams_lst)
            except ZeroDivisionError: # I added this the zero division error that occured when the perplexity was zero
                perplexities[author] = float('inf')

        min_perplexity = min(perplexities, key=perplexities.get)
        if show_perplexity:
            print(f"Author: {min_perplexity} \t Perplexity: {perplexities[min_perplexity]}")
        return min_perplexity

    def evaluate_devset(self, dev_data, show_accuracy = False):
        print("Results on dev set:")
        all_accuracies = []
        for author, samples in dev_data.items():
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

    def __generate_text(self, author, prompt, num_words):
        model = self.models[author]
        if not model:
            raise ValueError(f"No trained model found for author: {author}")
        
        prompt_tokens = list(prompt.split())
        context = prompt_tokens[-(self.n - 1):] if self.n > 1 else []
        generated_tokens = model.generate(num_words, text_seed=context, random_seed=random.randint(1, 1000))

        return " ".join(prompt_tokens + generated_tokens)

    def generate_authors_text(self, prompts,authors, num_words=20):
        generated_texts = {}

        for author in authors:
                generated_texts[author] =  [self.__generate_text(author=author,prompt=prompt, num_words=num_words) for prompt in prompts]
        return generated_texts 
    
    def _compute_ngram_frequencies(self, train_data):
        ngram_counter = Counter()
        for sentence in train_data:
            ngrams = generate_ngrams(sentence, self.n)
            ngram_counter.update(ngrams)
        return ngram_counter

    def get_top_features(self, top_k=5):
        stop_words = set(stopwords.words("english"))
        # Got these by trial and err :')
        special_tokens = {
            "<s>", "</s>", "--", "’", "“", "”", "‘", "—", "…", "n't", "'s", "'m", "'re", "'ve", "'ll", "'d"
        } 

        top_features = {}

        for author, ngram_counts in self.ngram_frequencies.items():
            filtered_ngrams = [
                (ngram, count)
                for ngram, count in ngram_counts.most_common()
                if not any(
                    word in string.punctuation or
                    word.lower() in stop_words or 
                    word in special_tokens or 
                    word.strip() == "" 
                    for word in ngram
                )
            ]

            top_features[author] = filtered_ngrams[:top_k]

        return top_features
    