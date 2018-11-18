import glob
import pickle
import time

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Merge, Activation, Flatten
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import image
from keras.preprocessing import sequence
from tqdm import tqdm


class ImageCaption:

    def __init__(self):

        # image encoding model
        model = InceptionV3(weights='imagenet')
        from keras.models import Model
        new_input = model.input
        hidden_layer = model.layers[-2].output

        self.model_new = Model(new_input, hidden_layer)

        # word embedding
        unique = pickle.load(open('unique_words.p', 'rb'))
        self.word2idx = {val:index for index, val in enumerate(unique)}
        print(self.word2idx['<start>'])
        self.idx2word = {index:val for index, val in enumerate(unique)}

        # cationing model
        self.max_len = 40
        vocab_size = len(unique)
        embedding_size = 300
        image_model = Sequential([
                Dense(embedding_size, input_shape=(2048,), activation='relu'),
                RepeatVector(self.max_len)
            ])
        caption_model = Sequential([
                Embedding(vocab_size, embedding_size, input_length=self.max_len),
                LSTM(256, return_sequences=True),
                TimeDistributed(Dense(300))
            ])
        self.final_model = Sequential([
                Merge([image_model, caption_model], mode='concat', concat_axis=1),
                Bidirectional(LSTM(256, return_sequences=False)),
                Dense(vocab_size),
                Activation('softmax')
            ])
        self.final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        #final_model.summary()
        self.final_model.load_weights('./weights/time_inceptionV3_1.5987_loss.h5')
        print('Image Captioning model loaded.')


    def preprocess_input(self, x):
        x /= 255.
        #x -= 0.5
        #x *= 2.
        return x


    def preprocess(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = self.preprocess_input(x)
        return x


    def encode(self, image):
        image = self.preprocess(image)
        temp_enc = self.model_new.predict(image)
        temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
        return temp_enc


    def predict_captions(self, image):
        start_word = ["<start>"]
        while True:
            par_caps = [self.word2idx[i] for i in start_word]
            par_caps = sequence.pad_sequences([par_caps], maxlen=self.max_len, padding='post')
            e = self.encode(image)#encoding_test[image[len(images):]]
            preds = self.final_model.predict([np.array([e]), np.array(par_caps)])
            word_pred = self.idx2word[np.argmax(preds[0])]
            start_word.append(word_pred)
            
            if word_pred == "<end>" or len(start_word) > self.max_len:
                break
                
        return ' '.join(start_word[1:-1])


if __name__ == '__main__':

    caption = ImageCaption();

    start_time = time.time()
    print ('Normal Max search:', caption.predict_captions('images/bike.png'))
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    print ('Normal Max search:', caption.predict_captions('images/obstacle.png'))
    print("--- %s seconds ---" % (time.time() - start_time))

