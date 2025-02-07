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
* [Feature Extraction from Text Data](#feature-extraction-from-text-data)
* Bag of Words
* TF/IDF
* Spelling Correction
* Word Embeddings
* Named Entity Recognition
* Data Augmentation
#### Tools and Libraries
* NLTK
* spaCy
* gensim (Word2vec)
* scikit-learn

#### NLP Project
[Twitter Semtiments](#twitter-sentiments)

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

## Twitter Sentiments
Twitter Sentiment analysis is a process of determining whether a tweet is positive or negative.

Steps involved: 
- Import Dataset.
- Alter Dataset to keep only one Columns (i.e tweet)
- Convert to lowercase
- Removal of Punctuations
- Removal of Stopwords
- Removal of Frequent Words
- Removal of Rare Words
- Removal of Special characters
- Stemming
- Lemmatization & POS Tagging
- Remove URLs 
- Remove HTML Tags
- Spelling Correction
- Feature Extraction from Text Data
    * Bag of Words
    * TF-IDF
    * Word Embeddings (Word2Vec, GloVe, FastText)
- Named Entity Recognition
- Data Augmentation for Text 
    - Synonym Replacement
    - Random Substitution
    - Random Deletion
    - Random Swap
    - Back Translation
    


## Feature Extraction from Text Data
Feature extraction from text data is a crucial step in text classification. The goal is to transform the text 
into a numerical representation that can be used by machine learning algorithms. There are several techniques
for feature extraction from text data, including:

1.  `Bag-of-Words (BoW)`: This is a simple and widely used
technique for feature extraction from text data. It represents each document as a bag, or a set
of its word occurrences, without considering the order or context of the words. The BoW model is
often used in conjunction with term frequency-inverse document frequency (TF-IDF) to reduce the
dimensionality of the feature space and to down-weight the importance of common words.
```python
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(stop_words='english')
bow.fit(text_data)
```
2.  `Term Frequency-Inverse Document Frequency (TF-IDF)`: This is a techniqu
e that takes into account the importance of each word in a document and its rarity across the entire corpus
. It is often used in conjunction with the BoW model to reduce the dimensionality of the featur
space and to down-weight the importance of common words.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(text_data)
tfidf.vocabulary_
```
3.  `Word Embeddings`: This is a technique that represents words as vectors in a high
dimensional space, such that semantically similar words are mapped to nearby points. Word embeddings
can be used to capture the nuances of language and to improve the performance of text classification models.

`Word Embedding using Glove`: GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.
Download link: [Stanford's GloVe 100d word embeddings](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt)

4. `Word2vec`: The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence.
```python
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
model = Word2Vec(common_texts, size=100, min_count=1)
model.wv['graph']
model.wv.most_similar('graph')
```

5. `Named Entity Recognition`: 
Named Entity Recognition (NER) is a subtask of information extraction that involves identifying named entities in un
structured text and categorizing them into predefined categories such as person, organization, location, date, tim
e, money, percentage, etc.
```python
!pip install -U pip setuptools wheel
!pip install -U spacy
!pip -m spacy download en_core_web_sm

import spacy
from spacy import displacy
```

6. `Data Augmentation for Text`: 
Data augmentation is a technique used to artificially increase the size of a dataset by applying transformations to the existing
data.
Uses:-
- Increase the dataset size by creating more samples.
- Reduce overfitting.
- Imporve model generalization.
- Handling imbalance dataset. 

```python
!pip install nlpaug
!pip install sacremoses
import nlpaug.augmenter.word as naw
```

7.  `Convolutional Neural Networks (CNNs)`: This is a type of neural
network that can be used for feature extraction from text data. CNNs use convolutional and pooling layers
to extract features from text data, and can be used to capture local patterns and relationships in the data.

8.  `Recurrent Neural Networks (RNNs)`: This is a type of neural
network that can be used for feature extraction from text data. RNNs use recurrent connections to captur
e the temporal relationships in the data, and can be used to capture long-range dependencies in the data.

9.  `Transformers`: This is a type of neural network that can be used for featur
e extraction from text data. Transformers use self-attention mechanisms to capture the relationships between
words in a sentence, and can be used to capture complex patterns and relationships in the data.

10.  `Pre-trained Language Models`: This is a type of model that has been pre-trained
on a large corpus of text data, and can be fine-tuned for specific text classification tasks.


