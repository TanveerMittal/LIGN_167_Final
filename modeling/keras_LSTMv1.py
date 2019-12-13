import warnings
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed

training_data = pickle.load(open("../data/train-v1.1.pkl", "rb"))
print("%d questions in our training set" % len(training_data))


word_to_index, index_to_words, word_to_vec_map = pickle.load(open("../embeddings/glove.6B.50d.pkl", "rb"))
vocab_len = len(index_to_words)
print("%d words in our model's vocabulary" % vocab_len)


max_qc_len = max([len(qac["question"] + ["<sep>"] + qac["context"]) for qac in training_data])
max_ans_len = max([len(qac["answer"]) + 1 for qac in training_data])

def preprocess(data):
    encoder_input_data = np.zeros((len(data), max_qc_len))
    decoder_input_data = np.zeros((len(data), max_ans_len))
    decoder_target_data = np.zeros((len(data), max_ans_len), dtype=np.uint8)

    for i in range(encoder_input_data.shape[0]):
        qc = data[i]["question"] + ["<sep>"] + data[i]["context"]
        encoder_input_data[i,:] = [word_to_index[word] if word in word_to_index else word_to_index["<unk>"] for word in qc] + [0 for j in range(max_qc_len - len(qc))]

        ans_input = ["<start>"] + data[i]["answer"]
        decoder_input_data[i,:] = [word_to_index[word] if word in word_to_index else word_to_index["<unk>"] for word in ans_input] + [0 for j in range(max_ans_len - len(ans_input))]

        ans_output = data[i]["answer"] + ["<end>"]
        decoder_target_data[i,:] = [word_to_index[word] if word in word_to_index else word_to_index["<unk>"] for word in ans_output] + [0 for j in range(max_ans_len - len(ans_input))]
    
    return encoder_input_data, decoder_input_data, decoder_target_data

encoder_input_data, decoder_input_data, decoder_target_data = preprocess(training_data)


emb_dim = len(word_to_vec_map['a'])
state_dim = 512
batch_size = 16
epochs = 1
learning_rate = 0.01
path = "saves/keras_LSTM.h5"
load = True

encoder_inputs = Input(shape=(max_qc_len,), dtype='int32')

emb_matrix = np.zeros((vocab_len, emb_dim))
for word, index in word_to_index.items():
    if index != 0 :
        emb_matrix[index, :] = word_to_vec_map[word]
embedding_layer = Embedding(vocab_len, emb_dim, trainable=False, mask_zero=True)
embedding_layer.build((None,))
embedding_layer.set_weights([emb_matrix])

encoder_embeddings = embedding_layer(encoder_inputs)

encoder = LSTM(state_dim, return_state=True)(encoder_embeddings)

encoder_outputs, state_h, state_c = encoder

encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(max_ans_len,))
decoder_embeddings = embedding_layer(decoder_inputs)
decoder_lstm = LSTM(state_dim, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(decoder_embeddings, initial_state=encoder_states)
outputs = TimeDistributed(Dense(vocab_len, activation='softmax'))(decoder_outputs)


model = Model([encoder_inputs, decoder_inputs], outputs)

if load:
    loaded_model = load_model(path)
    model.set_weights(loaded_model.get_weights())
model.summary()

opt = Adam(lr=0.000001)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size, epochs=epochs)
model.save(path)

val_data = pickle.load(open("../data/dev-v1.1.pkl", "rb"))
print("%d questions in our validation set" % len(val_data))

encoder_input_val, decoder_input_val, decoder_target_val = preprocess(val_data)

model.evaluate([encoder_input_val, decoder_input_val], decoder_target_val)
