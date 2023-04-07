# importation package---------------------------------------------------------------------------------------------------
import nltk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

#data preprocessing----------------------------------------------------------------------------
#lire json----------------------------------------------------------------
with open("intents.json") as file:
    data = json.load(file)
try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    # make a list of the tokenize words in our dataset---------------
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
# on n ajoute pas  les mot dupliquer pour connaitre le nombre de mot qui connait le chatbot
            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)
#phase d'entainement
tensorflow.compat.v1.reset_default_graph()
#creation du reseux neuronne
net = tflearn.input_data(shape=[None, len(training[0])])

# pour chaque entree de net dans le reseau de neuronne il va creer 23 neuronne pour chqaue couche cacher on a 2 couche
net = tflearn.fully_connected(net, 23)
net = tflearn.fully_connected(net, 23)

# pour nous permettre davoir une probabiliter pour chaque sortie
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")

# applique une regression au reseau de neuronne
net = tflearn.regression(net)

# entrenemment du model DNN est un type de reseaux de neurone il prend le net et lutilise
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")

except:
    # fit le modele adapter au modele
    model.fit(training, output, n_epoch=1000, batch_size=23, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)

    return numpy.array(bag)


def chat():
    print("COMMENCE DE PARLER AVEC LE CHATBOT tape quit pour arreter la conversation")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit" :
            break
        # donne probabilité de chaque tag
        results = model.predict([bag_of_words(inp, words)])[0]
        # donne le tag avec la plus grande proba ou la reponse se trouve
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if results[results_index]> 0.7:

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("je n'ai pas compris ta phrase, essaye encore")
chat()
