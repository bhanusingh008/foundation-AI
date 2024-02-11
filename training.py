from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

import random
import json
import pickle
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('wordnet')

# That's all we need to start training

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_these = ["?", "!", ".", ","]

for intent in intents["intents"]:
    for pattern in intent["pattern"]:
        words_list = nltk.tokenize.word_tokenize(pattern)
        words.extend(words_list)
        documents.append((words_list, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])


# Got classes for each tag + words list having patterns to be recognized
# tokenizer just separated words and makes a list out of it

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_these]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    words_pattern = document[0]
    words_pattern = [lemmatizer.lemmatize(word.lower()) for word in words_pattern]
    for word in words:
        bag.append(1) if word in words_pattern else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)

for train_data in training:
    print(train_data)

training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_dim=(len(train_x[0])), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('foundation_ai.model', hist)

print("done")