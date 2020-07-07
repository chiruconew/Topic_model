import pandas as pd 
import pickle
import gensim.corpora as corpora

from src.features.tokenize import tokenize
from src.features.dictionary import create_dictionary
from src.features.clean import clean_up_text


def load_model():
    with open(r"models/model.pkl", "rb") as input_file:
        model = pickle.load(input_file)
    return(model)

def test(model,text:str):
    text                 = pd.DataFrame(data={'content':[text]}, columns=['content'])
    doc                  = clean_up_text(text)
    lemma                = tokenize(doc)
    id2word,  corpus     = create_dictionary(lemma)
    prediction = model[corpus]

    topics = list()
    for prob in prediction[0][1]:
        for topic in prob[1]:
            topics.append(topic)
    moda = max(set(topics), key=topics.count)
    topics = model.print_topics()[moda]
    return topics