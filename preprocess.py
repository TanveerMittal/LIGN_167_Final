import re
import json
import torch
import pickle
import numpy as np
from utils import *

split = lambda x: re.findall(r"[\w']+|[.,!?;]", x.lower()) if x is not None else []
def json_to_dctv1(raw):
    qas = []
    for i in range(len(raw["data"])):
        for j in range(len(raw["data"][i]["paragraphs"])):
            for k in range(len(raw["data"][i]["paragraphs"][j]["qas"])):
                qa = raw["data"][i]["paragraphs"][j]["qas"][k]
                question = qa["question"]
                answer = qa["answers"][0]["text"]
                qas.append({"question": split(question),
                            "answer": split(answer),
                            "context": split(raw["data"][i]["paragraphs"][j]['context'])})
    return qas

def json_to_dctv2(raw):
    qas = []
    for i in range(len(raw["data"])):
        for j in range(len(raw["data"][i]["paragraphs"])):
            for k in range(len(raw["data"][i]["paragraphs"][j]["qas"])):
                qa = raw["data"][i]["paragraphs"][j]["qas"][k]
                question = qa["question"]
                if(qa["is_impossible"]):
                    continue
                else:
                    answer = qa["answers"][0]["text"]

                is_impossible = qa["is_impossible"]
                qas.append({"question": split(question),
                            "answer": split(answer),
                            "context": split(raw["data"][i]["paragraphs"][j]['context'])})
    return qas

training_data = json_to_dctv1(json.load(open("data/train-v1.1.json")))
pickle.dump(training_data, open("data/train-v1.1.pkl", "wb"))
val_data = json_to_dctv1(json.load(open("data/dev-v1.1.json")))
pickle.dump(val_data, open("data/dev-v1.1.pkl", "wb"))

def torch_embeddings(words_to_index, index_to_words, word_to_vec_map):
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

def np_embeddings(words_to_index, index_to_words, word_to_vec_map):
    dim = len(word_to_vec_map['a'])

    vocab =  {word for qac in training_data + val_data for word in qac["question"] + qac["context"] + qac["answer"]}
    overwrite = 1
    for idx in range(1, len(index_to_words) + 1):
        if index_to_words[idx] in vocab:
            index_to_words[overwrite] = index_to_words[idx]
            if idx != overwrite:
                index_to_words.pop(idx)
            words_to_index[index_to_words[overwrite]] = overwrite
            overwrite += 1
        else:
            word_to_vec_map.pop(index_to_words[idx])
            words_to_index.pop(index_to_words[idx])
            index_to_words.pop(idx)

    for qac in training_data + val_data:
        for word in qac["question"] + qac["context"] + qac["answer"]:
            vocab.add(word)

    unneeded = []
    for word in word_to_vec_map:
        if not word in vocab:
            unneeded.append(word)

    def vocab_append(word, embedding, index=None):
        if index is None:
            index = len(index_to_words)
        word_to_vec_map[word] = embedding
        index_to_words[index] = word
        words_to_index[word] = index

    # Padding token
    vocab_append("<pad>", np.random.rand(dim), index=0)

    # Unknown token embedding representation
    unk = np.zeros((dim))
    for emb in word_to_vec_map.values():
        unk += emb
    unk = unk / len(word_to_vec_map)
    vocab_append("<unk>", unk)

    # Seed numpy for random embedding generation
    np.random.seed(50)

    # Seperation of query and context token
    vocab_append("<sep>", np.random.rand(dim))
    # Start of answer token
    vocab_append("<start>", np.random.rand(dim))
    # End of answer token
    vocab_append("<end>", np.random.rand(dim))


    return words_to_index, index_to_words, word_to_vec_map


embeddings = "glove.6B.50d.txt"
words_to_index, index_to_words, word_to_vec_map = read_glove_vecs("embeddings/" + embeddings)
words_to_index, index_to_words, word_to_vec_map = np_embeddings(words_to_index, index_to_words, word_to_vec_map)
pickle.dump((words_to_index, index_to_words, word_to_vec_map), open("embeddings/%s.pkl" % embeddings[:-4], 'wb'))
