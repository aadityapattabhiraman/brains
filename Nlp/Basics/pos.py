#!/home/akugyo/Programs/Python/PyTorch/bin/python

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


stop_words = set(stopwords.words("english"))

text = "Baby this is what you came for " \
    "Lightning strikes every times you move" \
    "And everybody's watching now. "

tokenized = sent_tokenize(text)

for i in tokenized:
    words_list = word_tokenize(i)
    words_list = [w for w in words_list if not w in stop_words]

    tagged = nltk.pos_tag(words_list)
    print(tagged)
