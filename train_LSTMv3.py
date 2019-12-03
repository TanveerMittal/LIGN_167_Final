import pickle
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import torch
from modeling.LSTMv3 import *

training_data = pickle.load(open("data/train-v2.0.pkl", "rb"))
print("%d questions in our training set" % len(training_data))

words_to_index, index_to_words, word_to_vec_map = pickle.load(open("embeddings/glove.6B.50d.pkl", "rb"))
print("%d words in our model's vocabulary" % len(word_to_vec_map))

n_h = 256
n_x = word_to_vec_map["a"].shape[0]
n_y = len(index_to_words)
model_name = "LSTMv1"

q_encoder_params = lstm_encoder_init(n_h, n_x)
c_encoder_params = lstm_encoder_init(n_h, n_x)
decoder_params = lstm_decoder_init(n_h * 4, n_x, n_y)

#encoder_params, decoder_params = pickle.load(open("modeling/saves/%s.pkl" % model_name, 'rb'))

q_encoder_params, c_encoder_params, decoder_params = train(training_data, q_encoder_params, c_encoder_params, decoder_params,
                                                           word_to_vec_map, words_to_index, index_to_words, model_name,
                                                           learning_rate=0.001, batch_size=32, epochs=1, sample_size=len(training_data)//10)

pickle.dump((q_encoder_params, c_encoder_params, decoder_params), open("modeling/saves/%s.pkl" % model_name, "wb"))
