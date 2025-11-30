import csv
import collections
import math
import argparse


class TFIDFSystem:

    def __init__(self, f):

        self.invertedIndex = collections.defaultdict(lambda: collections.defaultdict(int))
        self.docs = set()
        self.X_test = []
        self.y_test = []
        self.train_labels = {}
        self.test_ids = []

        reader = csv.DictReader(f)

        for row in reader:
            doc_id = row["rewire_id"]
            text = row["text"]
            label = row["label"]
            split = row["split"].strip()

            if split == "train":
                self.docs.add(doc_id)

                frequencies = {}
                tokens = text.lower().split()

                for token in tokens:
                    token = token.strip('"')
                    frequencies[token] = frequencies.get(token, 1) if token not in frequencies else frequencies[token] + 1

                for token, tf in frequencies.items():
                    self.invertedIndex[token][doc_id] = tf

                self.train_labels[doc_id] = label

            else:
                self.X_test.append(text)
                self.y_test.append(label)
                self.test_ids.append(doc_id)

        self.docs = sorted(self.docs)


def main(corpus):
    tfidf = TFIDFSystem(open(corpus))

    for i, prompt in enumerate(tfidf.X_test):
        p_vector = {}
        p_terms = prompt.lower().split()
        for term in p_terms:
            term = term.strip('"')
            p_vector[term] = p_vector.get(term,0) + 1
            
        p_weights = {}
        p_length = 0.0
        for term in p_vector:
            if term in tfidf.invertedIndex:
                tf = p_vector[term]
                p_log = 1 + math.log(tf, 10)
                df = len(tfidf.invertedIndex[term])
                idf= math.log(len(tfidf.docs)/ df, 10)
                weight = p_log * idf
                p_weights[term] = weight
                p_length += weight**2
        p_norm = math.sqrt(p_length) if p_length > 0 else 1.0

        for term in p_weights:
            p_weights[term] /= p_norm
        
        scores = collections.defaultdict(float)
        tweet_lengths = collections.defaultdict(float)

        for term, p_weight in p_weights.items():
            if term in tfidf.invertedIndex:
                for id, tf in tfidf.invertedIndex[term].items():
                    log_tf = 1 + math.log(tf, 10)
                    scores[id] += p_weight * log_tf
                    tweet_lengths[id] += log_tf**2
        
        for id in scores:
            norm = math.sqrt(tweet_lengths[id]) if tweet_lengths[id] > 0 else 1.0
            scores[id] /= norm

        all_ids= set(tfidf.docs)
        scored = list(scores.items())
        missing_tweets = [(id, 0.0) for id in all_ids if id not in scores]
        combined = scored + missing_tweets
        combined.sort(key=lambda x:(-x[1], x[0]))
        top_id, _ = combined[0]
        label = tfidf.train_labels.get(top_id, "0")
        label_str = "sexist" if label == "1" else "not sexist"
        print(tfidf.test_ids[i], label_str)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("CORPUS",
                        help="Path to file with the corpus")
    args = parser.parse_args()
    main(args.CORPUS)
