
# coding: utf-8

# In[2]:


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

# In[3]:


#token = '/usr/local/data/bonniehu/flickr_data/Flickr8k.token.txt'


# In[4]:


#captions = open(token, 'r').read().strip().split('\n')


# ## Creating a dictionary containing all the captions of the images

# In[5]:

'''
d = {}
for i, row in enumerate(captions):
    row = row.split('\t')
    row[0] = row[0][:len(row[0])-2]
    if row[0] in d:
        d[row[0]].append(row[1])
    else:
        d[row[0]] = [row[1]]


# In[6]:


d['1000268201_693b08cb0e.jpg']


# In[7]:


images = '/usr/local/data/bonniehu/flickr_data/Flicker8k_Dataset/'


# In[8]:


# Contains all the images
img = glob.glob(images+'*.jpg')


# In[9]:


img[:5]


# In[10]:


train_images_file = '/usr/local/data/bonniehu/flickr_data/Flickr_8k.trainImages.txt'


# In[11]:


train_images = set(open(train_images_file, 'r').read().strip().split('\n'))


# In[12]:


def split_data(l):
    temp = []
    for i in img:
        if i[len(images):] in l:
            temp.append(i)
    return temp


# In[13]:


# Getting the training images from all the images
train_img = split_data(train_images)
len(train_img)


# In[14]:


val_images_file = '/usr/local/data/bonniehu/flickr_data/Flickr_8k.devImages.txt'
val_images = set(open(val_images_file, 'r').read().strip().split('\n'))


# In[15]:


# Getting the validation images from all the images
val_img = split_data(val_images)
len(val_img)


# In[16]:


test_images_file = '/usr/local/data/bonniehu/flickr_data/Flickr_8k.testImages.txt'
test_images = set(open(test_images_file, 'r').read().strip().split('\n'))


# In[17]:


# Getting the testing images from all the images
test_img = split_data(test_images)
len(test_img)


# In[18]:


Image.open(train_img[0])
'''

# We will feed these images to VGG-16 to get the encoded images. Hence we need to preprocess the images as the authors of VGG-16 did. The last layer of VGG-16 is the softmax classifier(FC layer with 1000 hidden neurons) which returns the probability of a class. This layer should be removed so as to get a feature representation of an image. We will use the last Dense layer(4096 hidden neurons) after popping the classifier layer. Hence the shape of the encoded image will be (1, 4096)

# In[19]:


def preprocess_input(x):
    x /= 255.
    #x -= 0.5
    #x *= 2.
    return x


# In[20]:


def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)
    return x



# In[23]:


model = InceptionV3(weights='imagenet')


# In[24]:


from keras.models import Model

new_input = model.input
hidden_layer = model.layers[-2].output

model_new = Model(new_input, hidden_layer)


# In[25]:


#tryi = model_new.predict(preprocess(train_img[0]))


# In[28]:


def encode(image):
    image = preprocess(image)
    temp_enc = model_new.predict(image)
    temp_enc = np.reshape(temp_enc, temp_enc.shape[1])
    return temp_enc


# In[33]:

'''
encoding_train = {}
for img in tqdm(train_img):
    encoding_train[img[len(images):]] = encode(img)


# In[36]:


with open("encoded_images_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_train, encoded_pickle) 


# In[34]:


encoding_train = pickle.load(open('encoded_images_inceptionV3.p', 'rb'))


# In[18]:


encoding_train['3556792157_d09d42bef7.jpg'].shape


# In[82]:


encoding_test = {}
for img in tqdm(test_img):
    encoding_test[img[len(images):]] = encode(img)


# In[100]:


with open("encoded_images_test_inceptionV3.p", "wb") as encoded_pickle:
    pickle.dump(encoding_test, encoded_pickle) 


# In[19]:


encoding_test = pickle.load(open('encoded_images_test_inceptionV3.p', 'rb'))


# In[20]:


encoding_test[test_img[0][len(images):]].shape


# In[29]:


train_d = {}
for i in train_img:
    if i[len(images):] in d:
        train_d[i] = d[i[len(images):]]


# In[30]:


len(train_d)


# In[31]:


train_d[images+'3556792157_d09d42bef7.jpg']


# In[32]:


val_d = {}
for i in val_img:
    if i[len(images):] in d:
        val_d[i] = d[i[len(images):]]


# In[33]:


len(val_d)


# In[34]:


test_d = {}
for i in test_img:
    if i[len(images):] in d:
        test_d[i] = d[i[len(images):]]


# In[35]:


len(test_d)


# Calculating the unique words in the vocabulary.

# In[36]:


caps = []
for key, val in train_d.items():
    for i in val:
        caps.append('<start> ' + i + ' <end>')


# In[37]:


words = [i.split() for i in caps]


# In[38]:


unique = []
for i in words:
    unique.extend(i)


# In[39]:


unique = list(set(unique))


# In[40]:


# with open("unique.p", "wb") as pickle_d:
#     pickle.dump(unique, pickle_d) 
'''

# In[41]:


unique = pickle.load(open('unique_words.p', 'rb'))


# In[42]:


len(unique)


# Mapping the unique words to indices and vice-versa

# In[43]:


word2idx = {val:index for index, val in enumerate(unique)}


# In[44]:


word2idx['<start>']


# In[45]:


idx2word = {index:val for index, val in enumerate(unique)}


# In[47]:


idx2word[5553]

'''
# Calculating the maximum length among all the captions

# In[48]:


max_len = 0
for c in caps:
    c = c.split()
    if len(c) > max_len:
        max_len = len(c)
'''
max_len = 40

# In[50]:


vocab_size = len(unique)


# Adding <start> and <end> to all the captions to indicate the starting and ending of a sentence. This will be used while we predict the caption of an image

# In[52]:

'''
f = open('flickr8k_training_dataset.txt', 'w')
f.write("image_id\tcaptions\n")


# In[53]:


for key, val in train_d.items():
    for i in val:
        f.write(key[len(images):] + "\t" + "<start> " + i +" <end>" + "\n")

f.close()


# In[54]:


df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')


# In[56]:


c = [i for i in df['captions']]
len(c)


# In[57]:


imgs = [i for i in df['image_id']]


# In[58]:


a = c[-1]
a, imgs[-1]


# In[59]:


for i in a.split():
    print (i, "=>", word2idx[i])


# In[60]:


samples_per_epoch = 0
for ca in caps:
    samples_per_epoch += len(ca.split())-1


# In[61]:


samples_per_epoch
'''

'''
# ## Generator 
# 
# We will use the encoding of an image and use a start word to predict the next word.
# After that, we will again use the same image and use the predicted word 
# to predict the next word.
# So, the image will be used at every iteration for the entire caption. 
# This is how we will generate the caption for an image. Hence, we need to create 
# a custom generator for that.
# 
# The CS231n lecture by Andrej Karpathy explains this concept very clearly and beautifully.
# Link for the lecture:- https://youtu.be/cO0a0QYmFm8?t=32m25s

# In[62]:


def data_generator(batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        
        df = pd.read_csv('flickr8k_training_dataset.txt', delimiter='\t')
        df = df.sample(frac=1)
        iter = df.iterrows()
        c = []
        imgs = []
        for i in range(df.shape[0]):
            x = next(iter)
            c.append(x[1][1])
            imgs.append(x[1][0])


        count = 0
        while True:
            for j, text in enumerate(c):
                current_image = encoding_train[imgs[j]]
                for i in range(len(text.split())-1):
                    count+=1
                    
                    partial = [word2idx[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    
                    # Initializing with zeros to create a one-hot encoding matrix
                    # This is what we have to predict
                    # Hence initializing it with vocab_size length
                    n = np.zeros(vocab_size)
                    # Setting the next word to 1 in the one-hot encoded matrix
                    n[word2idx[text.split()[i+1]]] = 1
                    next_words.append(n)
                    
                    images.append(current_image)

                    if count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
                        count = 0
'''

# ## Let's create the model

# In[63]:


embedding_size = 300


# Input dimension is 4096 since we will feed it the encoded version of the image.

# In[64]:


image_model = Sequential([
        Dense(embedding_size, input_shape=(2048,), activation='relu'),
        RepeatVector(max_len)
    ])


# Since we are going to predict the next word using the previous words(length of previous words changes with every iteration over the caption), we have to set return_sequences = True.

# In[65]:


caption_model = Sequential([
        Embedding(vocab_size, embedding_size, input_length=max_len),
        LSTM(256, return_sequences=True),
        TimeDistributed(Dense(300))
    ])


# Merging the models and creating a softmax classifier

# In[66]:


final_model = Sequential([
        Merge([image_model, caption_model], mode='concat', concat_axis=1),
        Bidirectional(LSTM(256, return_sequences=False)),
        Dense(vocab_size),
        Activation('softmax')
    ])


# In[67]:


final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])


# In[68]:


final_model.summary()

'''

# In[101]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[102]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[103]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[111]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[119]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[144]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[161]:


final_model.optimizer.lr = 1e-4
final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[174]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[175]:


final_model.save_weights('time_inceptionV3_7_loss_3.2604.h5')


# In[57]:


final_model.load_weights('time_inceptionV3_7_loss_3.2604.h5')


# In[58]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[69]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[78]:


final_model.save_weights('time_inceptionV3_3.21_loss.h5')


# In[79]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[88]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[101]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)


# In[110]:


final_model.save_weights('time_inceptionV3_3.15_loss.h5')


# In[ ]:


final_model.fit_generator(data_generator(batch_size=128), samples_per_epoch=samples_per_epoch, nb_epoch=1, 
                          verbose=2)

'''
# In[69]:


final_model.load_weights('./weights/time_inceptionV3_1.5987_loss.h5')


# ## Predict funtion

# In[79]:


def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        e = encode(image)#encoding_test[image[len(images):]]
        preds = final_model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])


# In[80]:


def beam_search_predictions(image, beam_index = 3):
    start = [word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            e = encode(image)#encoding_test[image[len(images):]]
            preds = final_model.predict([np.array([e]), np.array(par_caps)])
            
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption



# In[78]:

start_time = time.time()
try_image4 = 'images/bike.png'
print ('Normal Max search:', predict_captions(try_image4))
#print ('Beam Search, k=3:', beam_search_predictions(try_image4, beam_index=3))
#print ('Beam Search, k=5:', beam_search_predictions(try_image4, beam_index=5))
#print ('Beam Search, k=7:', beam_search_predictions(try_image4, beam_index=7))
#Image.open(try_image4)
print("--- %s seconds ---" % (time.time() - start_time))


# In[82]:

start_time = time.time()
im = 'images/obstacle.png'
print ('Normal Max search:', predict_captions(im))
#print ('Beam Search, k=3:', beam_search_predictions(im, beam_index=3))
#print ('Beam Search, k=5:', beam_search_predictions(im, beam_index=5))
#print ('Beam Search, k=7:', beam_search_predictions(im, beam_index=7))
#Image.open(im)
print("--- %s seconds ---" % (time.time() - start_time))

