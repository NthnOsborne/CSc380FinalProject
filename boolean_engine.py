# DO NOT modify code except "YOUR CODE GOES HERE" blocks

import collections
import argparse
import re

from query import Query, QuerySyntaxError
from node import Node

# Add support for phrase queries using a biword index
# Add support for wildcard queries using a permuterm index
#   Implement your search tree for the permuterm index in Node.py

class IRSystem:
    def __init__(self, f):
        # inverted index as discussed in class
        self.inverted_index = collections.defaultdict(lambda: [])

        # YOUR CODE GOES HERE
        #   Create whichever indixes you need

        # partially copied from hw 1b code provided,

        # set of documents to implement the NOT operator
        self.docs = set()

        # head of the binary searcg tree for permuterm index
        self.tree_head = None
        # biword index
        self.biword_index = collections.defaultdict(lambda: [])

        # dictionary of file for matching phrase queries
        self.file_docs = collections.defaultdict(list)

        for line in f:
            tokens = line.lower().split()
            doc_id = int(tokens[0])
            self.docs.add(doc_id)
            curr_word = []

            for token in tokens[1:]:
                if token != '-':
                    self.file_docs[doc_id].append(token)
                    # add the biword indices to the dictionary
                    curr_word.append(token)
                    if(len(curr_word) == 2):
                        new_token = curr_word[0] + " " + curr_word[1]
                        self.biword_index[new_token].append(doc_id)
                        curr_word = [curr_word[1]]
            if(len(curr_word )== 2):
                new_token = curr_word[0] + " " + curr_word[1]
                self.biword_index[new_token].append(doc_id)

            for token in set(tokens[1:]):
                if token != '-':
                    # add the permuterm indices to the binary search tree for the token
                    self.permuterm(token)
                    self.inverted_index[token].append(doc_id)

        self.docs = sorted(self.docs)
        #print(self.biword_index)
        # biword index


    def pri(self, head):
        if(head == None):
            return "None"
        #print("key: " + head.key)
        
        self.pri(head.left)
        self.pri(head.right)

    # calculates all the permuterm indices and adds them to the tree
    def permuterm(self, token):
        # number of shifts = the length of the token
        new_token = token + "$" # token$
        new_node = Node(new_token, token)
        if(self.tree_head == None):
            self.tree_head = new_node
        else:
            self.add_node(new_node)
        # shifting the token, then adding to the BST
        i = 0
        while(i < (len(token) + 1)):
            letter = new_token[0]
            new_token = new_token[1:]
            new_token = new_token + letter
            new_node = Node(new_token, token)
            self.add_node(new_node)
            i += 1



    # adds a node to the binary search tree used for the permuterm index
    def add_node(self, node):
        curr = self.tree_head
        while(curr != None):
            if(node.key < curr.key):
                if(curr.left != None):
                    curr = curr.left
                else:
                    # node found its palce in the tree
                    curr.left = node
                    return
            elif(node.key > curr.key):
                if(curr.right != None):
                    curr = curr.right
                else:
                    # node found its place in the tree
                    curr.right = node
                    return
            # else, it is already in the tree
            else:
                return




    def q_term(self, term):
        term = term.lower()
        self.matches = []
        # not a wildcard query
        if '*' not in term:
            return self.inverted_index[term]

        results = []

        # YOUR CODE GOES HERE
        #   Add support for wildcard queries
        if '*' in term:
            term = term + "$"
            # push * to the end of the term
            while(term[len(term) - 1] != '*'):
                char = term[0]
                term = term[1:]
                term = term + char
            term = term.replace('*', '')
            if(term == "$"):
                return self.docs
            match = []
            matched_tokens = self.find_prefix(term, self.tree_head, match)
            # add every doc in the matched tokens
            for token in matched_tokens:
                for doc in self.inverted_index[token]:
                        results.append(doc)
            results.sort()
        
        new_results = []
        for item in results:
            if item not in new_results:
                new_results.append(item)
        return new_results
    
    # takes a term in "term*" format and finds each token with a matching prefix
    def find_prefix(self, term, curr_head, match):
        if (curr_head == None):
            return match
        if(curr_head.key[:len(term)] == term):
            # must search left and right branches
            match.append(curr_head.value)
            self.find_prefix(term, curr_head.left, match)
            self.find_prefix(term, curr_head.right, match)
        else:
            if(curr_head.key[:len(term)] < term):
                self.find_prefix(term, curr_head.right, match)
            else:
                self.find_prefix(term, curr_head.left, match)

        return match



    def q_not(self, docs):
        results = []

        # YOUR CODE GOES HERE
        results = [i for i in self.docs if i not in docs]
        return results

    @staticmethod
    def q_and(docs1, docs2):
        results = []

        # YOUR CODE GOES HERE
        i1 = 0
        i2 = 0
        while(i1 < len(docs1) and i2 < len(docs2)):
            if(docs1[i1] == docs2[i2]):
                results.append(docs1[i1])
                i1 += 1
                i2 += 1
            elif(docs1[i1] < docs2[i2]):
                i1 += 1
            else:
                i2 += 1

        return results

    @staticmethod
    def q_or(docs1, docs2):
        results = []

        # YOUR CODE GOES HERE
        i1 = 0
        i2 = 0
        while(i1 < len(docs1) and i2 < len(docs2)):
            if(docs1[i1] < docs2[i2]):
                results.append(docs1[i1])
                i1 += 1
            elif(docs2[i2] < docs1[i1]):
                results.append(docs2[i2])
                i2 += 1
            else:
                results.append(docs1[i1])
                i1 += 1
                i2 += 1
        if(i1 < len(docs1)):
            while(i1 < len(docs1)):
                results.append(docs1[i1])
                i1 += 1
        else:
            while(i2 < len(docs2)):
                results.append(docs2[i2])
                i2 += 1
        return results



    def q_phrase(self, terms):
        results = []

        # YOUR CODE GOES HERE
        #   Add support for phrase queries
        terms_list = terms
        curr_terms = []
        curr_docs = []
        i = 0
        while(i < len(terms_list)):
            curr_terms.append(terms_list[i].lower())
            if(len(curr_terms) == 2):
                new_term = curr_terms[0] + " " + curr_terms[1]
                print(new_term)
                for doc in self.biword_index[new_term]:
                    #print(doc)
                    curr_docs.append(doc)
                if(len(results) == 0):
                    results = curr_docs
                else:
                    results = self.q_and(results, curr_docs)
                curr_docs = []
                curr_terms = [curr_terms[1]]
            i += 1

        # ensuring the phrase occurs exactly in each document returned
        phrase = ""
        for term in terms_list:
            phrase += term
            phrase += " "
        phrase = phrase[:len(phrase) - 1]

        new_results = []
        for item in results:
            if item not in new_results:
                text = " ".join(self.file_docs[item]).lower()
                if phrase in text:
                    
                    new_results.append(item)
                

        

        return new_results

    def run_query(self, query, method=None):
        # Tokens are reversed to facilitate running arbitrary boolean queries (see below)
        #tokens = reversed(list(Query.tokenize(query)))
        tokens = list(Query.tokenize(query))

        results = []

        # YOUR CODE GOES HERE
        #   Substitute the code above to run:
        #     - single boolean operators (undergraduate students)
        #     - arbitrary boolean queries (graduate students)
        #tokens = [token.lower() if token not in ["NOT", "AND", "OR"] else token for token in tokens]
        count = 0
        for token in tokens:
            #print(token)
            count += 1
        results = []
        if count == 1:
            if(isinstance(tokens[0], str)):
                assert tokens[0] not in ["NOT", "AND", "OR"], "Invalid query"
            # if a query
            if(isinstance(tokens[0], list)):
                if(len(tokens[0]) > 1):
                    results = self.q_phrase(tokens[0])
                else:
                    results = self.q_term(tokens[0][0])
            else:
                results = self.q_term(tokens[0])
        elif count == 2:
            assert tokens[0] == "NOT", "Unary operator is not NOT"
            if(isinstance(tokens[1], list)):
                if(len(tokens[1]) > 1):
                    results = self.q_not(self.q_phrase(tokens[1]))
                else:
                    results = self.q_not(self.q_term(tokens[1][0]))
            else:
                results = self.q_not(self.q_term(tokens[1]))
        elif count == 3:
            if(isinstance(tokens[0], str)):
                if tokens[0] == "AND":
                    if(isinstance(tokens[1], list)):
                        if(len(tokens[1]) > 1):
                            results1 = self.q_phrase(tokens[1])
                        else: 
                            results1 = self.q_term(tokens[1][0])
                    else:
                        results1 = self.q_term(tokens[1])
                    if(isinstance(tokens[2], list)):
                        if(len(tokens[2]) > 1):
                            results2 = self.q_phrase(tokens[2])
                        else:
                            results2 = self.q_term(tokens[2][0])
                    else:
                        results2 = self.q_term(tokens[2])
                    results = self.q_and(results1, results2)
                elif tokens[0] == "OR":
                    if(isinstance(tokens[1], list)):
                        if(len(tokens[1]) > 1):
                            results1 = self.q_phrase(tokens[1])
                        else:
                            results1 = self.q_term(tokens[1][0])
                    else:
                        results1 = self.q_term(tokens[1])
                    if(isinstance(tokens[2], list)):
                        if(len(tokens[2]) > 1):
                            results2 = self.q_phrase(tokens[2])
                        else:
                            results2 = self.q_term(tokens[2][0])
                    else:
                        results2 = self.q_term(tokens[2])
                    results = self.q_or(results1, results2)
                else:
                    assert False, "Binary operator is neither AND nor NOT"



        return results


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
