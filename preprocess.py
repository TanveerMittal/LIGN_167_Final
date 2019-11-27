import re
import json
import torch
import pickle
from utils import *

split = lambda x: re.findall(r"[\w']+|[.,!?;]", x.lower())
def json_to_dct(raw):
    qas = []
    for i in range(len(raw["data"])):
        for j in range(len(raw["data"][i]["paragraphs"])):
            for k in range(len(raw["data"][i]["paragraphs"][j]["qas"])):
                qa = raw["data"][i]["paragraphs"][j]["qas"][k]
                question = qa["question"]
                if(qa["is_impossible"]):
                    answer = ""
                else:
                    answer = qa["answers"][0]["text"]

                is_impossible = qa["is_impossible"]
                qas.append({"question": split(question),
                            "answer": split(answer),
                            "context": split(raw["data"][i]["paragraphs"][j]['context'])})
    return qas

pickle.dump(json_to_dct(json.load(open("data/train.json"))), open("data/train.pkl", "wb"))
pickle.dump(json_to_dct(json.load(open("data/validation.json"))), open("data/validation.pkl", "wb"))

def refactor_embeddings(words_to_index, index_to_words, word_to_vec_map):
    dim = len(word_to_vec_map['a'])
    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = torch.from_numpy(word_to_vec_map[key].reshape((dim, 1))).type(torch.float32)

    def vocab_append(word, embedding, index=None):
        if index is None:
            index = len(index_to_words)
        word_to_vec_map[word] = embedding
        index_to_words[index] = word
        words_to_index[word] = index

    # Unknown token embedding representation
    unk = torch.zeros((dim, 1))
    for emb in word_to_vec_map.values():
        unk = torch.add(unk, emb)
    unk = unk / len(word_to_vec_map)
    vocab_append("<unk>", unk, index=0)

    # Seed numpy for random embedding generation
    np.random.seed(50)

    # Seperation of query and context token
    vocab_append("<sep>", torch.from_numpy(np.random.rand(dim, 1)).type(torch.float32))
    # Start of answer token
    vocab_append("<start>", torch.from_numpy(np.random.rand(dim, 1)).type(torch.float32))
    # End of answer token
    vocab_append("<end>", torch.from_numpy(np.random.rand(dim, 1)).type(torch.float32))

    return words_to_index, index_to_words, word_to_vec_map


embeddings = "glove.6B.50d.txt"
words_to_index, index_to_words, word_to_vec_map = read_glove_vecs("embeddings/" + embeddings)
words_to_index, index_to_words, word_to_vec_map = refactor_embeddings(words_to_index, index_to_words, word_to_vec_map)
pickle.dump((words_to_index, index_to_words, word_to_vec_map), open("embeddings/%s.pkl" % embeddings[:-4], 'wb'))
