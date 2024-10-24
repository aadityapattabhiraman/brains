#!/home/akugyo/Programs/Python/PyTorch/bin/python

# import nltk
from nltk.stem import WordNetLemmatizer


# nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("corpora"))
