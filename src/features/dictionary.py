  
from typing import List

from gensim.corpora import Dictionary
import gensim.corpora as corpora


def create_dictionary(documents: List[List[str]]):
    id2word = corpora.Dictionary(documents)
    texts   = documents
    corpus  = [id2word.doc2bow(text) for text in texts ]
    return (id2word,  corpus)


