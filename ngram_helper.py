from collections import Counter, defaultdict
import random
import math


def generate_ngrams(sequence, n):
    """Generates n-grams from a given sequence with padding."""
    sequence = ["<s>"] * (n - 1) + sequence + ["</s>"]
    return [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


class MLE:
    def __init__(self, order):
        self.order = order
        self.ngram_counts = Counter()
        self.context_counts = Counter()

    def fit(self, train_data, vocab):
        for sentence in train_data:
            for ngram in generate_ngrams(sentence, self.order):
                self.ngram_counts[ngram] += 1
                self.context_counts[ngram[:-1]] += 1

    def score(self, word, context):
        return self.ngram_counts.get(tuple(context) + (word,), 0) / self.context_counts.get(tuple(context), 1)

    def perplexity(self, ngrams_lst):
        log_prob_sum = 0
        for ngram in ngrams_lst:
            prob = self.score(ngram[-1], ngram[:-1])
            log_prob_sum += math.log(prob or 1e-10)
        return math.exp(-log_prob_sum / len(ngrams_lst))

    def generate(self, num_words, text_seed=None, random_seed=None):
        random.seed(random_seed)
        text = text_seed[:] if text_seed else []
        for _ in range(num_words):
            context = text[-(self.order - 1):] if self.order > 1 else []
            candidates = [ngram[-1] for ngram in self.ngram_counts if ngram[:-1] == tuple(context)]
            text.append(random.choice(candidates) if candidates else "<UNK>")
        return text


class Laplace(MLE):
    def score(self, word, context):
        return (self.ngram_counts.get(tuple(context) + (word,), 0) + 1) / (self.context_counts.get(tuple(context), 0) + len(self.ngram_counts))


class KneserNeyInterpolated(MLE):
    def fit(self, train_data, vocab):
        super().fit(train_data, vocab)
        self.continuation_counts = Counter()
        self.lower_order_counts = Counter()
        for ngram in self.ngram_counts:
            self.lower_order_counts[ngram[1:]] += 1
            self.continuation_counts[ngram[-1]] += 1

    def score(self, word, context):
        d = 0.75
        context_tuple = tuple(context)
        count = self.ngram_counts.get(context_tuple + (word,), 0)
        lambda_factor = (d / self.context_counts.get(context_tuple, 1)) * len([w for w in self.ngram_counts if w[:-1] == context_tuple])
        lower_order_prob = self.continuation_counts.get(word, 0) / sum(self.continuation_counts.values())
        return max(count - d, 0) / self.context_counts.get(context_tuple, 1) + lambda_factor * lower_order_prob
      
class StupidBackoff(MLE):
    def __init__(self, order, alpha=0.4):
        super().__init__(order)
        self.alpha = alpha  # Discount factor

    def score(self, word, context):
        context_tuple = tuple(context)
        
        ngram_count = self.ngram_counts.get(context_tuple + (word,), 0)
        context_count = self.context_counts.get(context_tuple, 0)
        
        # If the n-gram is found, use it, otherwise back off
        if ngram_count > 0:
            return ngram_count / context_count
        else:
            if len(context_tuple) > 0:
                return self.alpha * super().score(word, context_tuple[1:])
            else:
                return self.alpha * self.ngram_counts.get((word,), 0) / sum(self.ngram_counts.values())
