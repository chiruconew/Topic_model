import spacy  
from typing import List

import re
import pandas as pd
import gensim
from typing import Generator, List
import spacy

from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from spacy.lang.en import English

import nltk
nltk.download()
from nltk.corpus import stopwords   

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])


def tokenize(documents: List[str]) -> List[List[str]]:
    document_words = clean_up_text(documents)
    document_words = list(sentence_to_words(document_words))
    document_words = remove_stopwords(document_words)
    document_words = build_bigrams(document_words)
    document_words = lemmatization(nlp, document_words)
    return document_words


def sentence_to_words(sentences: List[str]) -> List[List[str]]:
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))    

def remove_stopwords(texts: List[List[str]]) -> List[List[str]]:
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def lemmatization(nlp: English, texts:List[List[str]], allowed_postags:List[str]=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def core_bigram(data_words: List[List[str]], min_count:int=5,threshold:int=10):
    bigram = gensim.models.Phrases(data_words, min_count=min_count, threshold=threshold)
    return(bigram)

def build_bigrams(data_words: List[List[str]], min_count:int=5,threshold:int=10) -> List[List[str]]:
    bigram = core_bigram(data_words,min_count,threshold)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in data_words]
    
def build_trigrams(data_words:List[List[str]],min_count:int=5,threshold:int=10) -> List[List[str]]:
    bigram = core_bigram(data_words, min_count, threshold)
    trigram = gensim.models.Phrases(bigram[data_words], threshold)  
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return[trigram_mod[ data_words ] for doc in data_words]
