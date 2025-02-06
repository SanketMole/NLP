# NLP
A repo made to learn all concepts of Natural Language Processing, with the help of a small project. 

## Natural Language Processing
Natural Language Processing (NLP) is a branch of artificial intelligence (AI) that enables computers to understand, interpret, generate, and manipulate human language. It combines computational linguistics, machine learning, and deep learning techniques to process text and speech.
Aim of NLP is to process text and speech to understand Human Language. 
#### Applications
* Sentiment Analysis (Product Review, Social Media Monitoring)
* Text Classification (Spam detection, Fake news detection)
* Named Entity Recognition (Extracting news, spaces, dates from text)
* Language Translation 
* Question Answering
* Speech Recognition
* Chatbots & Virtual Assistants (Siri, Alexa)
#### Techniques 
* [Tokenization](#tokenization)
* Stopword removal
* [Parts of Speech Tagging (POS)](#part-of-speech-tagging-pos)
* [Stemming](#stemming)
* [Lemmatization](#lemmatization)
* Spelling Correction
* Bag of Words
* TF/IDF
* Word Embeddings
* Named Entity Recognition
* Data Augmentation
#### Tools and Libraries
* NLTK
* spaCy
* gensim (Word2vec)
* scikit-learn

## Vectorization
Vectorization is a process of converting text data into numerical vectors that can be processed by machine learning algorithms. The raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.

In order to address this issue, early researchers came up with some methods to extract numerical features from text content. All these methods have the following steps in common:
1. **Tokenizing** string, for instance, by using white-spaces and punctuation as token separators. And then giving an integer-id for each possible token.
2. **Counting** the occurance of tokens in each string/sentence/document.
3. **Normalizing and Weighting** with diminishing importance tokens that occur in the majority of samples/documents. 

A set of documents can thus be **represented by a matrix** with one row per document and one column per token (e.g. word) occurring in the corpus. *corpus* is nothing but collection of documents. 

Documents are described by **word occurrences** while completely *ignoring the relative position information* of the words in the document.

We want an **algebraic model** representing textual information as a vector, the components of this vector could represent the *absence or presence* (Bag of Words) of it in a document or even the *importance of a term* (tf–idf) in the document.

## CountVectorizer
The first step in modeling the document into a vector is to create a **dictionary of terms present in documents**. To do that, you can simple tokenize the complete document & select all the unique terms from the document. Also we are supposed to remove stop words from them. 
```python
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(train_set)
vectorizer.vocabulary_
```
The `fit()` method is used to create a dictionary of terms present in the documents. 
The `vocabulary_` attribute is a dictionary mapping terms to their integer indices in the feature matrix. The key is *token* and the value is *index*. 

The **CountVectorizer** already uses as default *analyzer* call ***Word***, which is responsible to convert the text to lowercase, accents removal, token extraction, filter stop words, etc...

#### Drawbacks
1. Tokens which we get is not in the order. That information is lost in the transformation process. 
2. Document size is not taken into consideration.
3. Count Vectorization *scales up frequent terms and scales down rare terms*, which is empirically more informative than the high frequency terms. (i.e. document frequency of words are also ignored (weighting))



## Tokenization
Tokenization is the process of breaking down the given text in natural language processing into the smallest unit in a sentence called a token. Punctuation marks, words, and numbers can be considered tokens.

```python
from nltk import sent_tokenize, word_tokenize
```
There are two types of tokenization: **Sentence Tokenization** and **Word Tokenization**. They mean as their name suggests. 

## Stemming
Stemming is the process of finding the root of words. A word stem need not be the same root as a dictionary-based morphological root, it just is an equal to or smaller form of the word.
```python
from nltk.stem import PorterStemmer, SnowballStemmer
```

## Lemmatization
Lemmatization is the process of finding the form of the related word in the dictionary. It is different from Stemming. It involves longer processes to calculate than Stemming.
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('workers')
```

## Part of Speech Tagging (POS)
Part of Speech Tagging is a process of converting a sentence to forms — list of words, list of tuples (where each tuple is having a form (word, tag)). The tag in case of is a part-of-speech tag, and signifies whether the word is a noun, adjective, verb, and so on.

[POS Tag](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

```python
from nltk import pos_tag
pos_tag(['fighting'])
#Output: [('fighting', 'VBG')]
```


