import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pandas as pd
from google.colab import files
from google.colab import drive

drive.mount('/content/drive')
path_docs = "/content/drive/My Drive/Colab Notebooks/PruebaLUNES/"

file_path = path_docs + 'SalMay25.txt'
fw=open(file_path, "w+")

module_url = "https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2" 
model = hub.load(module_url)
print ("Modulo %s  cargado." % module_url)
def embed(input):
  return model(input)

def cosine_similarity(v1, v2):
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if (not mag1) or (not mag2):
        return 0
    return np.dot(v1, v2) / (mag1 * mag2)

Set_Tweets = [
'rt codechocoprensa en la jurisdiccioncodechoco conmemoramos el diamundialdelosanimales  y al ser una region tan biodiversa  invi',
'diamundialdelosanimales  en colombia se han registrado 509 especies de aranas estos heroes son los grandes contr httpstco9bxovndzfh',
'rt corpochivor hoy diamundialdelosanimales es importante recalcar la conservacion y proteccion de todas las especies teniendo en cuenta',
'rt proteccionponal hoy festejamos el dia mundial de los animales  en conmemoracion a san francisco de asis quien no solo servia a s',
'sina25anos hoy celebramos a las entidades e instituciones que hacen parte del sistema nacional ambiental sina y httpstcozzbwk2udsg',
'rt infopresidencia hoy desde minambienteco conmemoramos el diamundialdelosanimales nuestro trabajo dentro de las 59 areas protegidas d',
'crautonoma ivanduque infopresidencia ungrd corantioquia ideamcolombia minhacienda mincomercioco felicitamos httpstcoidyhwqhf4t',
'rt minvivienda hoy conmemoramos el diamundialdelosanimales nuestro trabajo dentro de las 59 areas protegidas del sistema nacional de pa',
'rt ideamcolombia diamundialdelosanimales  colombia es el segundo pais en el mundo con mas diversidad de especies de murcielagos alrede',
'diamundialdelosanimales  las hormigas son titanes de la naturaleza que aportan equilibrio a nuestro ecosistema  httpstconkp8pa8rh1'
]

fw.write('Modulo nnlm-es-dim128-with-normalization-2'+ '\n')
mens_emb = embed(Set_Tweets)
for i, sentencia1 in enumerate(Set_Tweets):
    for j in range(i+1, len(Set_Tweets) ):         
        fw.write(sentencia1[:10]+' i ')
        fw.write(str(i))
        fw.write(' j ')
        fw.write(str(j))
        fw.write(' Sim : ')
        fw.write(str(cosine_similarity(mens_emb[i],mens_emb[j])))
        fw.write('\n')
fw.close()
