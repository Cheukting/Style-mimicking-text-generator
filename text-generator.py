

import keras
import numpy as np
import re

# this is used to remove stage direction if we don't want them
def remove_stage_dir(text):
    text = re.sub("[\<].*?[\>]", "", text)
    text = re.sub("\\s+", " ", text)
    return text

# chop it up in sentances (for Tokenizer)
text = ''.join(open("hamlet.txt", encoding = "UTF-8").readlines())
in_sentences = re.split(r' *[\.\?!][\'"\)\]]* *', remove_stage_dir(text))


print(in_sentences[0:10])
print('Corpus length:', len(text))



# Length of extracted sample
maxlen = 20

# Stride of sampling
step = 1

# This holds our samples sequences
sentences = []

# This holds the next word (as training label)
next_word = []

#use Kears Tokenizer
from keras.preprocessing.text import Tokenizer

max_num_word = 10000 #max size of library
tokenizer = Tokenizer(num_words=max_num_word)
tokenizer.fit_on_texts(list(in_sentences))
list_tokenized_train = tokenizer.texts_to_sequences(list(in_sentences))

#if the library ends up smaller then the max size, update the info
if len(tokenizer.word_index) < max_num_word:
    max_num_word = len(tokenizer.word_index)
    
print('Number of words:', max_num_word)

#stick the encoded words back together is a big sequence
token_word = []
for line in range (0,len(in_sentences)):
    that_sentences = list_tokenized_train[line]
    for i in range(0,len(that_sentences)):
        token_word.append(that_sentences[i])

#sample the sequence
for i in range(0, len(token_word) - maxlen, step):
    sentences.append(token_word[i: i + maxlen])
    next_word.append(token_word[i + maxlen])
print('Number of sequences:', len(sentences))

#nomalized x
x = np.asarray(sentences).astype('float32')/max_num_word
#one-hot encode y
y = np.zeros((len(sentences), max_num_word), dtype=np.bool)
for i in range (0,len(sentences)):
    for j in range (0,maxlen):
        y[i, next_word[j]] = 1

#build Keras model, using word embedding layer and LSTM then 
#output via softmax layer to give a prediction distribution
from keras import layers

model = keras.models.Sequential()
model.add(layers.Embedding(max_num_word, 128, input_length=maxlen))
model.add(layers.LSTM(128,return_sequences=True))
model.add(layers.Flatten())
model.add(layers.Dense(max_num_word, activation='softmax'))

model.summary()

# Since our prediction are one-hot encoded, use `categorical_crossentropy` as the loss
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='mse', optimizer=optimizer)

#this is for sampling the next work with a prediction distribution
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    exp_preds = preds - np.exp(temperature)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

model.fit(x, y, batch_size=128, epochs=1)

#to change back to word
reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

#randomize a seed
import random

start_index = random.randint(0, len(token_word) - maxlen - 1)
generated_seed = token_word[start_index: start_index + maxlen]

generated_text = ' '.join([reverse_word_map.get(i) for i in generated_seed])
print('--- Generating with seed ---')
print(generated_text)
print('--- --- --- --- ---')

for i in range(20): #generate 20 words

    array_seed = np.zeros((maxlen,1))
    array_seed[:,0] = np.asarray(generated_seed).astype('float32')/max_num_word
    
    preds = model.predict(array_seed.transpose(), verbose=0)[0]
    next_index = sample(preds)
    next_word = reverse_word_map.get(next_index)

    generated_seed.append(next_index)       
    generated_seed = generated_seed[1:]
    generated_text = generated_text + ' ' + next_word

print('--- Generated text ---')
print(generated_text)
print('--- --- --- --- ---')
