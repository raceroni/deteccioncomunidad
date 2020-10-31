import pandas as pd
import numpy as np
from collections import defaultdict
import gensim
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import string
import re
import nltk
from nltk.corpus import stopwords
from pattern.text.es import lemma
from nltk.stem import WordNetLemmatizer
import csv

spanish_stopwords = stopwords.words('spanish')

# fr=open("Ini25May.txt", "r")                  Ejercicio4.csv
fr=open("Data_B0.csv", "r")      #  Data_U02.csv   Data_U03.csv
if fr.mode == 'r':
	contents =fr.readlines()
	# print(contents)

fw=open("Med5Sept.csv", "w+") # si cambia tambien para contenidofiltrado
fE=open("Err5sept.txt", "w+")
res = []
'''
texto = 'crautonoma ivanduque infopresidencia ungrd corantioquia ideamcolombia minhacienda mincomercioco felicitamos httpstcoidyhwqhf4t'
texto = re.sub(r'http\S+', 'labelurl', texto)
# texto= re.sub('httpstco[a-z0-9]{10}', 'labelurl', texto)
print(texto)
'''
wordnet_lemmatizer = WordNetLemmatizer()

for linea in contents:
    try:
        # linea = linea.replace("\n", " ")
        #   r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        new_text = ''.join(c for c in linea if c not in string.punctuation)
        new_text = new_text.lower()
        # new_text = new_text.replace('\n',' ')
        # new_text = new_text.rstrip('\n')
        # new_text = ' '.join(word0 for word0 in word_tokenize(new_text) if length(word0) > 2)
        new_text = ' '.join(word0 for word0 in word_tokenize(new_text) if word0 not in stopwords.words('spanish'))
        new_text = ' '.join(wordnet_lemmatizer.lemmatize(word1, pos="v") for word1 in word_tokenize(new_text))
        # new_text = re.sub("\n"," borrar" ,new_text)  comm 27 jun
        # new_text = re.sub('httpstco[a-z0-9]{10}', 'labelurl', new_text)
        new_text = re.sub(r'http\S+', 'labelurl', new_text)
        # new_text = [word for word in word_tokenize(new_text) if word not in stopwords.words('spanish')]
        fw.write(new_text)
        # print(new_text)
        fw.write('\n')
        res.append(new_text)
    except:
        fE.write(linea)
        continue
# es = [''.join(c for c in text if c not in string.punctuation) for text in textList]
fw.close()
fr.close()
fE.close()

# reviews_datasets = pd.read_csv(fw)
# reviews_datasets.dropna()  #  Remover los vlres nulos del DataSet
# fw.readable()
fr=open("Med5Sept.csv", "r")
if fr.mode == 'r':
    contenidofiltrado =fr.readlines()
    print(contenidofiltrado)

count_vect = CountVectorizer(max_df=0.1, min_df=0, stop_words=spanish_stopwords ) # 'spanish')
# doc_term_matrix = count_vect.fit_transform(reviews_datasets['description'].values.astype('U'))
# doc_term_matrix = count_vect.fit_transform(reviews_datasets['text'].values.astype('U'))
doc_term_matrix = count_vect.fit_transform(contenidofiltrado)

NUM_TOPICS = 6

LDA = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
lda_Z = LDA.fit_transform(doc_term_matrix)

sentencia = []
def print_topics(model, vectorizer, top_n=11):                  #  11 Sintagmas
    for idx, topic in enumerate(model.components_):
        # print("Topic %d:" % (idx))
        oracion = ' '.join([(vectorizer.get_feature_names()[i])
               for i in topic.argsort()[:-top_n - 1:-1]])
        # print([(vectorizer.get_feature_names()[i] )
        #      for i in topic.argsort()[:-top_n - 1:-1]])
        # print(oracion)
        sentencia.append(oracion)

print("LDA Model:")
print_topics(LDA, count_vect)
print("=" * 20)
print(sentencia)

lda_output = lda_Z
topicnames =["Topico"+str(i) for i in range(LDA.n_components)]

# docnames=[reviews_datasets['text']]
docnames=[contenidofiltrado]

df_document_topic=pd.DataFrame(np.round(lda_output,5), columns=topicnames,index=docnames)
dominant_topic=np.argmax(df_document_topic.values,axis=1)
df_document_topic['TopicoDomina']=dominant_topic
df_document_topic.sort_values(by=['TopicoDomina'],ascending=[True],inplace=True)   ## Sin inplace no funciona
# print(df.loc[df['Name'] == 'Bert'])
# print(df_document_topic.loc[df_document_topic['TopicoDomina']==4])
# print('Vamos a imprimir topicos dominantes ')
# df_document_topic.to_csv('pepe.csv',sep=';',encoding='utf-8')
# return df_document_topic.loc[df_document_topic['TopicoDomina'] == 3]
# print(topicnames)
# print(df_document_topic['Topico0'])
# print(df_document_topic[topicnames[0]])
i = 0
while i < NUM_TOPICS:
    new_df = pd.DataFrame(data=df_document_topic.loc[df_document_topic['TopicoDomina'] == i])
    # new_df = df_document_topic.loc[df_document_topic['TopicoDomina'] == 0]
    new_df.sort_values(by=[topicnames[i]], ascending=[False], inplace=True)
    new_df = new_df[topicnames[i]]
    print(topicnames[i])
    # print('El cero ')
    # print(new_df.to_string())
    # print('Ahora el 10')
    print(sentencia[i])
    # print(len(new_df))  937
    # print(new_df.shape) (937, )
    # print(new_df.dtype)  float64
    # new_df.append(sentencia[i])
    # tmp_df = pd.DataFrame({topicnames[i]:[sentencia[i]+';0.0']})
    # tmp_df = pd.DataFrame({sentencia[i]+';0.0': [ ]})  Add TOPICO
    # new_df.append(tmp_df,ignore_index=True,verify_integrity=False)  AL FINAL , No acepto el sort tampoco
    # print(new_df.to_string(1,100)) imprime mas de 100 !!!
    new_df.to_csv(topicnames[i] + '.csv', sep=';', encoding='utf-8')
    i += 1

# Cada topico .csv a un .txt , eliminando repetidos, encabezado el TOPICO, en ingles ; palabras cortads
i = 0
while i < NUM_TOPICS :
    with open(topicnames[i] + '.csv') as csv_file:
        filewri = open(topicnames[i]+'.txt', "w+")
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                # filewri.write("'")
                filewri.write(sentencia[i])
                repetida0 = sentencia[i]
                filewri.write(" ; ")
                # line_count += 1
            else:
                # print(f'\t{row[0]} Sentencia {row[1]} Score.')
                repetida1 = row[0]
                if repetida0 == repetida1:
                    continue
                else:
                    repetida0 = row[0]
                # filewri.write(" ; ")
                row[0] = re.sub("\n", " ", row[0])
                filewri.write(row[0])
                filewri.write(" ; ")
            line_count += 1
        filewri.close()
        print(f'Se procesaron  {line_count} lineas, para {topicnames[i]}.')
    i += 1
