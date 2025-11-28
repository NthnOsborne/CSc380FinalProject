import collections
import math
import argparse


class IRSystem:

    def __init__(self, f):
        # Use lnc to weight terms in the documents:
        #   l: logarithmic tf
        #   n: no df
        #   c: cosine normalization

        # Store the vecorized representation for each document
        #   and whatever information you need to vectorize queries in _run_query(...)

        # YOUR CODE GOES HERE
        self.invertedIndex = collections.defaultdict(lambda: collections.defaultdict(int))
        self.docs = set()

        for line in f:
            tokens = line.lower().split()
            doc_id = int(tokens[0])
            self.docs.add(doc_id)

            frequencies = {}

            for token in tokens[1:]:
                if token != '-':
                    if token not in frequencies:
                        frequencies[token] = 1
                    else:
                        frequencies[token] += 1
                    
            for token, f in frequencies.items():
                self.invertedIndex[token][doc_id] = f

        self.docs = sorted(self.docs)

        

        #norm_weights.sort(key=lambda x: -x[2])
        #top = norm_weights[:20]
        #for term, doc_id, weight in top:
            #print(term, doc_id, weight)

                    

    def run_query(self, query):
        terms = query.lower().split()
        return self._run_query(terms)

    def _run_query(self, terms):
        query_vector = {}
        for term in terms:
            if term != '-':
                query_vector[term] = query_vector.get(term, 0) + 1

        # Compute the query weights and length
        query_weights = {}
        query_length = 0.0
        for term in query_vector:
            if term in self.invertedIndex:
                tf = query_vector[term]
                query_log = 1 + math.log(tf, 10)
                df = len(self.invertedIndex[term])
                idf = math.log(len(self.docs) / df, 10)
                weight = query_log * idf
                query_weights[term] = weight
                query_length += weight ** 2

        query_norm = math.sqrt(query_length) if query_length > 0 else 1.0

        # Normalize query weights
        for term in query_weights:
            query_weights[term] /= query_norm

        # Score documents
        scores = collections.defaultdict(float)
        doc_lengths = collections.defaultdict(float)

        for term, query_weight in query_weights.items():
            if term in self.invertedIndex:
                for doc_id, tf in self.invertedIndex[term].items():
                    doc_log_tf = 1 + math.log(tf, 10)  # l: log tf
                    scores[doc_id] += query_weight * doc_log_tf
                    doc_lengths[doc_id] += doc_log_tf ** 2

        # Normalize document vectors
        for doc_id in scores:
            norm = math.sqrt(doc_lengths[doc_id]) if doc_lengths[doc_id] > 0 else 1.0
            scores[doc_id] /= norm

        # Rank and return top 10 documents
        all_doc_ids = set(self.docs)
        scored_docs = list(scores.items())
        missing_docs = [(doc_id, 0.0) for doc_id in all_doc_ids if doc_id not in scores]
        combined = scored_docs + missing_docs
        combined.sort(key=lambda x: (-x[1], x[0]))
        return [doc_id for doc_id, _ in combined[:10]]


def main(corpus):
    ir = IRSystem(open(corpus))

    while True:
        query = input('Query: ').strip()
        if query == 'exit':
            break
        results = ir.run_query(query)
        print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("CORPUS",
                        help="Path to file with the corpus")
    args = parser.parse_args()
    main(args.CORPUS)
