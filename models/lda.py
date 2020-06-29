import pandas as pd
import gensim 

from src.features import nlp
from src.features.tokenize import tokenize
from src.features.dictionary import create_dictionary
from src.data.prepare_data import read_sample
from src.features.utils import clean_up_text
from gensim.models import CoherenceModel



def load_doc() -> pd.DataFrame:
    return read_sample()

def lda_model(raw_file: pd.DataFrame):
    doc       = clean_up_text(raw_file)
    lemma     = tokenize(doc)
    id2word, corpus = create_dictionary(lemma)
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=id2word,num_topics=20, random_state=100,
                                    update_every=1,chunksize=100,passes=10,alpha='auto',per_word_topics=True)
    return(lda_model)
    
    
    
    