# Topic_model
# En esta tarea, aprendimos a modularizar los resultados de entrenamiento y prediccion de un Modelo de Topicos
# Utilizando un ambiente python, deben de cargarse con las siguientes instrucciones

$ from src.models.train import *
$ from src.models.predict import *

Luego ya se pueden correr tanto el entrenamiento como la prediccion

$ var1 = load_doc()
$ lda_model(var1)

# El modelo se guardo como pickel en en directorio /Topic_Model/models/model.pkl
# Ahora ya se puede ejecutar la prediccion de la siguiente forma:

$ var2 = load_model()
$ test(var3,'if would be better if I understand what is going on in these prediction, I want to see cars all over the prediction, I hope I could see something like that')

# Obtendran este resultado
(3, '0.026*"people" + 0.014*"say" + 0.014*"may" + 0.013*"reason" + 0.012*"believe" + 0.011*"evidence" + 0.010*"mean" + 0.009*"fact" + 0.009*"state" + 0.009*"claim"')

# Entiendo es el tercer topico, en el cual tiene esas probabilidades de las palabras que contiene

