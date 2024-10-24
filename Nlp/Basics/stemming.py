#!/home/akugyo/Programs/Python/PyTorch/bin/python

# import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# nltk.download("punkt_tab")
stemmer = PorterStemmer()

words = ["program", "programs", "programmer", "programming", "programmers"]

for w in words:
    print(w, ":", stemmer.stem(w))

sentence = "Programmers program with programming languages"
words = word_tokenize(sentence)

for w in words:
    print(w, ":", stemmer.stem(w))
