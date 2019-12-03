import torch
import torch.nn as nn
from torch.nn.modules import TransformerModel

training_data = pickle.load(open("data/train.pkl", "rb"))
print("%d questions in our training set" % len(training_data))

words_to_index, index_to_words, word_to_vec_map = pickle.load(open("embeddings/glove.6B.200d.pkl", "rb"))
print("%d words in our model's vocabulary" % len(word_to_vec_map))

ntokens = len(words_to_index) # the size of vocabulary
emsize = word_to_vec_map['a'].shape[0] # embedding dimension
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 5 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
