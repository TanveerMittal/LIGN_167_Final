import torch
from torch.nn.modules.normalization import LayerNorm
import numpy as np

def encoder_block_init(n_a, n_f, n_h, n_x):
    '''
    Arguements:
    n_a: the attention dimension
    n_f: the feed-forward network dimension
    n_h: the number of heads for multi-head attention
    n_x: the embedding/input dimension
    '''
    Wq = [torch.rand((n_a, n_x), dtype=torch.float32, requires_grad=True) for i in range(n_h)]
    Wk = [torch.rand((n_a, n_x), dtype=torch.float32, requires_grad=True) for i in range(n_h)]
    Wv = [torch.rand((n_a, n_x), dtype=torch.float32, requires_grad=True) for i in range(n_h)]
    Wo = torch.rand((n_a, n_a * n_h), dtype=torch.float32, requires_grad=True)
    Wf = torch.rand((n_f, n_a), dtype=torch.float32, requires_grad=True)
    norm1 = LayerNorm(n_a)
    norm2 = LayerNorm(n_f)
    return {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wf": Wf, "Wo": Wo, "norm1": norm1, "norm2": norm2}

def encoder_init(n_b, n_a, n_f, n_h, n_x):
    '''
    Arguements:
    n_b: the number of encoder blocks
    n_a: the attention dimension
    n_f: the feed-forward network dimension
    n_h: the number of heads for multi-head attention
    n_x: the embedding dimension
    '''
    return [encoder_block_init(n_a, n_f, n_h, n_x) for i in range(n_b)]

def encode(qac, word_to_vec_map, layers):
    '''
    Arguments:
    qac: python dictionary containing:
        question: A question in the form of a list of words and punctuation
        answer: An answer to the question in the form of a list of words and punctuation
        context: A paragraph of context in the form of a list of words and punctuation
    word_to_vec_map: python dictionary that maps an english word its corresponding GloVe embedding
    layers: list of dictionaries that contain the parameters for each encoder block
    '''
    # Create the sequence to computer encoder on
    sequence = qac["question"] + ["<sep>"] + qac["context"]
    sequence = [word_to_vec_map[word] if word in word_to_vec_map else word_to_vec_map["<unk>"] for word in sequence]

    # TODO: Add positional encodings


    # Concatenate embeddings into single matrix
    X = torch.cat(sequence, 1)

    # Compute first encoding block
    # Compute multi-headed self attention
    Q = [torch.matmul(Wq, X) for Wq in layers[0]["Wq"]]
    K = [torch.matmul(Wk, X) for Wk in layers[0]["Wk"]]
    V = [torch.matmul(Wv, X) for Wv in layers[0]["Wv"]]
    Z = [torch.matmul(V[i], torch.softmax(torch.matmul(torch.transpose(K[i], 0, 1), Q[i])/np.sqrt(K[i].shape[0]), dim=0)) for i in range(len(Q))]
    Z = torch.cat(Z, 0)
    Z = layers[0]["norm1"](Z)
    Z = torch.matmul(layers[0]["Wo"], Z)

    # Compute feedforward network
    R = torch.matmul(layers[0]["Wf"], Z)
    R = layers[0]["norm2"](R)

    # Propogate through the rest of the encoding blocks
    for layer in layers[1:]:
        Q = [torch.matmul(Wq, R) for Wq in layer["Wq"]]
        K = [torch.matmul(Wk, R) for Wk in layer["Wk"]]
        V = [torch.matmul(Wv, R) for Wv in layer["Wv"]]
        Z = [torch.matmul(V[i], torch.softmax(torch.matmul(torch.transpose(K[i], 0, 1), Q[i])/np.sqrt(K[i].shape[0]), dim=0)) for i in range(len(Q))]
        Z = torch.cat(Z, 0)
        Z = torch.matmul(layer["Wo"], Z)
        Z = layer["norm1"](Z)

        # Compute feedforward network
        R = torch.matmul(layers[0]["Wf"], Z)
        R = layers[0]["norm2"](R)

    return K, V, R
