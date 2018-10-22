import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from os import listdir
from helper import *


def define_model(vocab_size, max_length):  # the sequence model 
	inputs1 = Input(shape=(4096,)) 
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256)(se2)
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	return model

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos):
	X1, X2, y = list(), list(), list()
	for key in descriptions.keys():
		seq = tokenizer.texts_to_sequences([descriptions[key]])[0]
		for i in range(1, len(seq)):
			in_seq, out_seq = seq[:i], seq[i]
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
		X2.append(in_seq)
		y.append(out_seq)
		X1.append(photos[key])
	return np.array(X1), np.array(X2), np.array(y)

def extract_features(directory):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	features = dict()
	for name in listdir(directory):
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		image = img_to_array(image)
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		image = preprocess_input(image)
		feature = model.predict(image, verbose=0)
		image_id = name.split('.')[0]
		features[image_id] = feature
		if len(features) % 1000 == 0 : 
			print(len(features), end = " ")
	return features



directory = 'dataset'
features = extract_features(directory)
descriptions = load_clean_descriptions('descriptions.txt')

train_desc, test_desc, train_feat, test_feat = train_test(descriptions , features)
tokenizer = create_tokenizer(train_desc)
vocab_size = len(tokenizer.word_index) + 1
max_length = get_max(train_desc)

X1train , X2train, ytrain = create_sequences(tokenizer , max_length, train_desc , train_feat)
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_desc, test_feat)
X1test = X1test.reshape( X1test.shape[0],X1test.shape[2])
X1train = X1train.reshape(X1train.shape[0],X1train.shape[2])
model = define_model(vocab_size, max_length)

filepath = 'bestmodel.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.fit([X1train, X2train], ytrain, epochs=20, verbose=1, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))



