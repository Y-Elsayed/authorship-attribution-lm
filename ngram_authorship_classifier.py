from nltk.lm.models import MLE
from nltk.lm import StupidBackoff, Laplace
from nltk.lm.smoothing import KneserNey,AbsoluteDiscounting
from nltk.tokenize import word_tokenize as nltk_word_tokenize
from nltk.util import everygrams
from nltk.lm.preprocessing import padded_everygram_pipeline 

class NgramAuthorshipClassifier:

    def __init__(self, n = 2, smoothing="mle"):
        self.n = int(n)
        self.smoothing = smoothing
        self.models = {}


    def train(self, authors_data): # not sure if this is the best way to pass the authors data but it will be a map of author to their processed text
        print("Training LMs... (this may take a while)")
        for k,v in authors_data.items():
            train_data, vocab = padded_everygram_pipeline(self.n, v)
            if self.smoothing == "abs":
                model = AbsoluteDiscounting(self.n)
            elif self.smoothing == "sb":
                model = StupidBackoff(self.n)
            elif self.smoothing == "lp":
                model = Laplace(self.n)
            elif self.smoothing == "kn":
                model = KneserNey(self.n)
            else:
                if self.smoothing != "mle":
                    print(f"The {self.smoothing} smoothing is not supported. Defaulting to MLE")
                model = MLE(self.n)

            model.fit(train_data,vocab)
            self.models[k] = model


    def classify(self, sample):
        ngrams_lst = list(everygrams(sample, self.n))
        perplexities = {author: model.perplexity(ngrams_lst) for author,model in self.models.items()}
        return min(perplexities, key=perplexities.get)

    def evaluate_devset(self, dev_data):
        print("Results on dev set:")
        for author, samples in dev_data.items():
            correct = 0
            total = len(samples)
            if total == 0: 
                print(f"Skipping {author} (no test samples available)")
                continue
            for sample in samples:
                if self.classify(sample) == author:
                    correct+=1
            print(f"{author} \t {correct/total:.2f} correct")

    def predict(self, test_data,save_predictions = True):
        predictions = []
        for sample in test_data:
            predictions.append(self.classify(sample))
        if save_predictions:
            self.__save_predictions(predictions)

    
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
    