## NLTK

### Tokenize 

break down the text into smaller units  

```python
sent = "This is the blast off. time to blow this up. so best you run for cover."
print(word_tokenize(sent))
print(sent_tokenize(sent))
```

### Stopwords

A stop word is a commonly used word (such as “the”, “a”, “an”, or “in”) that a search engine has been programmed to ignore  

```python3
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



example_sent = """This is a sample sentence,
                  showing off the stop words filtration."""
 
stop_words = set(stopwords.words('english'))
 
word_tokens = word_tokenize(example_sent)
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

print(word_tokens)
print(filtered_sentence)
```

### Punctuations

Removes punctuations

```python3
from nltk.tokenize import RegexpTokenizer



tokenizer = RegexpTokenizer(r"\w+")

text = "This is another example! Notice: it removes punctuation."
tokens = tokenizer.tokenize(text)
print(tokens)
```

### Parts of speech

bla
```
CC coordinating conjunction 
CD cardinal digit 
DT determiner 
EX existential there (like: “there is” … think of it like “there exists”) 
FW foreign word 
IN preposition/subordinating conjunction 
JJ adjective – ‘big’ 
JJR adjective, comparative – ‘bigger’ 
JJS adjective, superlative – ‘biggest’ 
LS list marker 1) 
MD modal – could, will 
NN noun, singular ‘- desk’ 
NNS noun plural – ‘desks’ 
NNP proper noun, singular – ‘Harrison’ 
NNPS proper noun, plural – ‘Americans’ 
PDT predeterminer – ‘all the kids’ 
POS possessive ending parent’s 
PRP personal pronoun –  I, he, she 
PRP$ possessive pronoun – my, his, hers 
RB adverb – very, silently, 
RBR adverb, comparative – better 
RBS adverb, superlative – best 
RP particle – give up 
TO – to go ‘to’ the store. 
UH interjection – errrrrrrrm 
VB verb, base form – take 
VBD verb, past tense – took 
VBG verb, gerund/present participle – taking 
VBN verb, past participle – taken 
VBP verb, sing. present, non-3d – take 
VBZ verb, 3rd person sing. present – takes 
WDT wh-determiner – which 
WP wh-pronoun – who, what 
WP$ possessive wh-pronoun, eg- whose 
WRB wh-adverb, eg- where, when
```

```python
tokenized = sent_tokenize(txt)

for i in tokenized:

    word_list = nltk.word_tokenize(i)
    word_list = [w for w in word_list if not w in stop_words]

    tagged = nltk.pos_tag(word_list)
    print(tagged)
```

### Stemming

```python
ps = PorterStemmer()

sentence = "Programmers program with programming languages"
words = word_tokenize(sentence)

stemmed_sentence = reduce(lambda x, y: x + " " + ps.stem(y), words, "")
print(stemmed_sentence)
``` 

### N-grams

splits the input into a group of n words  

```python
from nltk import ngrams



sentence = "this is a foo bar sentences and i want to ngramize it"

n = 6
grams = ngrams(sentence.split(), n)

print(grams)
for gram in grams:
    print(gram)
```


### Word Embeddings

Word Embedding or Word Vector is a numeric vector input that represents a word in a lower-dimensional space. It allows words with similar meanings to have a similar representation  

### Bag of words

text representation technique that represents a document as an unordered set of words and their respective frequencies. It discards the word order and captures the frequency of each word in the document, creating a vector representation  

```python
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()
```

### TF-IDF

term frequency inverse document freuency  

```python3
string = ["hey there nigga", "sucks to be you", "well les see"]

tfidf = TfidfVectorizer()
result = tfidf.fit_transform(string)
```

### Word2vec

```python
sample = "Word embeddings are dense vector representations of words."
tokenized_corpus = word_tokenize(sample.lower())  # Lowercasing for consistency

skipgram_model = Word2Vec(sentences=[tokenized_corpus],
                          vector_size=100,
                          window=5,
                          sg=1,
                          min_count=1,
                          workers=4)   
```
