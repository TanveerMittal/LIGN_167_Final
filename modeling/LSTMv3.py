import torch
import random
import pickle
import numpy as np
from utils import *

# Retrieve torch device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def lstm_encoder_init(n_h, n_x):
    '''
    Arguements:
    n_h: the hidden state dimension
    n_x: the embedding dimension
    Returns:
    parameters: python dictionary containing:
        Wf: Weight matrix of the forget gate, numpy array of shape (n_h, n_h + n_x)
        bf: Bias of the forget gate, numpy array of shape (n_h, 1)
        Wu: Weight matrix of the update gate, numpy array of shape (n_h, n_h + n_x)
        bu: Bias of the update gate, numpy array of shape (n_h, 1)
        Wc: Weight matrix of the first "tanh", numpy array of shape (n_h, n_h + n_x)
        bc:  Bias of the first "tanh", numpy array of shape (n_h, 1)
        Wo: Weight matrix of the output gate, numpy array of shape (n_h, n_h + n_x)
        bo:  Bias of the output gate, numpy array of shape (n_h, 1)
    '''
    Wf = torch.rand((n_h, n_h + n_x), dtype=torch.float32, requires_grad=True)
    bf = torch.rand((n_h, 1), dtype=torch.float32, requires_grad=True)
    Wu = torch.rand((n_h, n_h + n_x), dtype=torch.float32, requires_grad=True)
    bu = torch.rand((n_h, 1), dtype=torch.float32, requires_grad=True)
    Wc = torch.rand((n_h, n_h + n_x), dtype=torch.float32, requires_grad=True)
    bc = torch.rand((n_h, 1), dtype=torch.float32, requires_grad=True)
    Wo = torch.rand((n_h, n_h + n_x), dtype=torch.float32, requires_grad=True)
    bo = torch.rand((n_h, 1), dtype=torch.float32, requires_grad=True)
    return {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "bf": bf, "bu": bu, "bo": bo, "bc": bc}

def lstm_encoder_step(xt, h_prev, c_prev, parameters):
    '''
    Computes the output of the LSTM cell at given timestep

    Arguments:
    xt: your input data at timestep "t", numpy array of shape (n_x, m).
    h_prev: Hidden state at timestep "t-1", numpy array of shape (n_h, m)
    c_prev: Memory state at timestep "t-1", numpy array of shape (n_h, m)
    parameters: python dictionary containing:
        Wf: Weight matrix of the forget gate, numpy array of shape (n_h, n_h + n_x)
        bf: Bias of the forget gate, numpy array of shape (n_h, 1)
        Wu: Weight matrix of the update gate, numpy array of shape (n_h, n_h + n_x)
        bu: Bias of the update gate, numpy array of shape (n_h, 1)
        Wc: Weight matrix of the first "tanh", numpy array of shape (n_h, n_h + n_x)
        bc:  Bias of the first "tanh", numpy array of shape (n_h, 1)
        Wo: Weight matrix of the output gate, numpy array of shape (n_h, n_h + n_x)
        bo:  Bias of the output gate, numpy array of shape (n_h, 1)
        Wy: Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_h)
        by: Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    h_next: next hidden state, of shape (n_h, m)
    c_next: next memory state, of shape (n_h, m)
    '''
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wu = parameters["Wu"]
    bu = parameters["bu"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]

    # Concatenate hidden state and input for matrix mulitplication
    hx = torch.cat((h_prev, xt), 0)

    # Compute the first tanh for the memory state using the previous hidden state and input
    ct = torch.tanh(torch.matmul(Wc, hx) + bc)

    # Compute the update gate
    update = torch.sigmoid(torch.matmul(Wu, hx) + bu)

    # Compute the forget gate
    forget = torch.sigmoid(torch.matmul(Wf, hx) + bf)

    # Compute the output gate
    output = torch.sigmoid(torch.matmul(Wo, hx) + bo)

    # Compute the memory state using the tanh computation and the appropriate gates
    c_next = torch.mul(update, ct) + torch.mul(forget, c_prev)

    # Compute the hidden state using the memory state and output gate
    h_next = torch.mul(output, torch.tanh(c_next))

    return h_next, c_next

def lstm_encode(sequence, word_to_vec_map, parameters):
    """
    Computes the forward propogation process of the LSTM cell to encode a question and its context

    Arguments:
    sequence: the input sequence to encoder using the LSTM
    word_to_vec_map: python dictionary that maps an english word its corresponding GloVe embedding
    parameters: python dictionary containing:
        Wf: Weight matrix of the forget gate, numpy array of shape (n_h, n_h + n_x)
        bf: Bias of the forget gate, numpy array of shape (n_h, 1)
        Wu: Weight matrix of the update gate, numpy array of shape (n_h, n_h + n_x)
        bu: Bias of the update gate, numpy array of shape (n_h, 1)
        Wc: Weight matrix of the first "tanh", numpy array of shape (n_h, n_h + n_x)
        bc: Bias of the first "tanh", numpy array of shape (n_h, 1)
        Wo: Weight matrix of the output gate, numpy array of shape (n_h, n_h + n_x)
        bo: Bias of the output gate, numpy array of shape (n_h, 1)
        Wy: Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_h)
        by: Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    encoding: the final LSTM hidden state that encodes the question and answer
    """
    # Compute LSTM encoding for the sequence in its original order
    sequence = ["<start>"] + sequence

    # Hidden and cell state initialization
    h = torch.zeros((parameters["Wf"].shape[0], 1), dtype=torch.float32, requires_grad=False)
    c = torch.zeros(h.shape, dtype=torch.float32, requires_grad=False)

    # Compute LSTM output on each word of the sentence
    for word in sequence:
        # Retrieve the embedding for the current word
        if word in word_to_vec_map:
            emb = word_to_vec_map[word]
        else:
            emb = word_to_vec_map["<unk>"]

        # LSTM computation
        h, c = lstm_encoder_step(emb, h, c, parameters)

    # Save the hidden and cell states for the forward encoding
    forward = (h, c)

    # Compute LSTM encoding for the sequence in its reverse order
    sequence = sequence[::-1]

    # Hidden and cell state initialization
    h = torch.zeros((parameters["Wf"].shape[0], 1), dtype=torch.float32, requires_grad=False)
    c = torch.zeros(h.shape, dtype=torch.float32, requires_grad=False)

    # Compute LSTM output on each word of the sentence
    for word in sequence:
        # Retrieve the embedding for the current word
        if word in word_to_vec_map:
            emb = word_to_vec_map[word]
        else:
            emb = word_to_vec_map["<unk>"]

        # LSTM computation
        h, c = lstm_encoder_step(emb, h, c, parameters)

    # Save the hidden and cell states for the backward encoding
    backward = (h, c)

    return (forward, backward)

# TODO change dimensions in docstring
def lstm_decoder_init(n_h, n_x, n_y):
    '''
    Arguements:
    n_h: the hidden state dimension
    n_x: the embedding dimension
    words_to_index: python dictionary that maps an english word its corresponding one_hot index
    Returns:
    parameters: python dictionary containing:
        Wf: Weight matrix of the forget gate, numpy array of shape (n_h, n_h + n_x)
        bf: Bias of the forget gate, numpy array of shape (n_h, 1)
        Wu: Weight matrix of the update gate, numpy array of shape (n_h, n_h + n_x)
        bu: Bias of the update gate, numpy array of shape (n_h, 1)
        Wc: Weight matrix of the first "tanh", numpy array of shape (n_h, n_h + n_x)
        bc: Bias of the first "tanh", numpy array of shape (n_h, 1)
        Wo: Weight matrix of the output gate, numpy array of shape (n_h, n_h + n_x)
        bo: Bias of the output gate, numpy array of shape (n_h, 1)
        Wy: Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_h)
        by: Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    '''
    Wf = torch.rand((n_h, n_h + n_x), dtype=torch.float32, requires_grad=True)
    bf = torch.rand((n_h, 1), dtype=torch.float32, requires_grad=True)
    Wu = torch.rand((n_h, n_h + n_x), dtype=torch.float32, requires_grad=True)
    bu = torch.rand((n_h, 1), dtype=torch.float32, requires_grad=True)
    Wc = torch.rand((n_h, n_h + n_x), dtype=torch.float32, requires_grad=True)
    bc = torch.rand((n_h, 1), dtype=torch.float32, requires_grad=True)
    Wo = torch.rand((n_h, n_h + n_x), dtype=torch.float32, requires_grad=True)
    bo = torch.rand((n_h, 1), dtype=torch.float32, requires_grad=True)
    Wy = torch.rand((n_y, n_h), dtype=torch.float32, requires_grad=True)
    by = torch.rand((n_y, 1), dtype=torch.float32, requires_grad=True)

    return {"Wf": Wf, "Wu": Wu, "Wo": Wo, "Wc": Wc, "Wy": Wy,
            "bf": bf, "bu": bu, "bo": bo, "bc": bc, "by": by}

def lstm_decoder_step(y_prev, h_prev, c_prev,  parameters):
    '''
    Computes the output of the LSTM cell at given timestep

    Arguments:
    y_prev: your output data at timestep "t-1", numpy array of shape (n_y, m).
    h_prev: Hidden state at timestep "t-1", numpy array of shape (n_h, m)
    c_prev: Memory state at timestep "t-1", numpy array of shape (n_h, m)
    parameters: python dictionary containing:
        Wf: Weight matrix of the forget gate, numpy array of shape (n_h, n_h + n_x)
        bf: Bias of the forget gate, numpy array of shape (n_h, 1)
        Wu: Weight matrix of the update gate, numpy array of shape (n_h, n_h + n_x)
        bu: Bias of the update gate, numpy array of shape (n_h, 1)
        Wc: Weight matrix of the first "tanh", numpy array of shape (n_h, n_h + n_x)
        bc: Bias of the first "tanh", numpy array of shape (n_h, 1)
        Wo: Weight matrix of the output gate, numpy array of shape (n_h, n_h + n_x)
        bo: Bias of the output gate, numpy array of shape (n_h, 1)
        Wy: Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_h)
        by: Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    h_next: next hidden state, of shape (n_h, m)
    c_next: next memory state, of shape (n_h, m)
    '''
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wu = parameters["Wu"]
    bu = parameters["bu"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # Concatenate hidden state and input for matrix mulitplication
    hx = torch.cat((h_prev, y_prev), 0)

    # Compute the first tanh for the memory state using the previous hidden state and input
    ct = torch.tanh(torch.matmul(Wc, hx) + bc)

    # Compute the update gate
    update = torch.sigmoid(torch.matmul(Wu, hx) + bu)

    # Compute the forget gate
    forget = torch.sigmoid(torch.matmul(Wf, hx) + bf)

    # Compute the output gate
    output = torch.sigmoid(torch.matmul(Wo, hx) + bo)

    # Compute the memory state using the tanh computation and the appropriate gates
    c_next = torch.mul(update, ct) + torch.mul(forget, c_prev)

    # Compute the hidden state using the memory state and output gate
    h_next = torch.mul(output, torch.tanh(c_next))

    # Compute softmax probability distribution
    y_t = torch.softmax(torch.matmul(Wy, h_next) + by, dim=0)

    return h_next, c_next, y_t

def lstm_decode(qac, encoding, word_to_vec_map, index_to_words, parameters):
    """
    Computes the forward propogation process of the LSTM cell to decode an answer from an encoding

    Arguments:
    qac: python dictionary containing:
        question: A question in the form of a list of words and punctuation
        answer: An answer to the question in the form of a list of words and punctuation
        context: A paragraph of context in the form of a list of words and punctuation
    encoding: lstm encoder output that encodes a query and its context as hidden and cell states
    parameters: python dictionary containing:
        Wf: Weight matrix of the forget gate, numpy array of shape (n_h, n_h + n_x)
        bf: Bias of the forget gate, numpy array of shape (n_h, 1)
        Wu: Weight matrix of the update gate, numpy array of shape (n_h, n_h + n_x)
        bu: Bias of the update gate, numpy array of shape (n_h, 1)
        Wc: Weight matrix of the first "tanh", numpy array of shape (n_h, n_h + n_x)
        bc: Bias of the first "tanh", numpy array of shape (n_h, 1)
        Wo: Weight matrix of the output gate, numpy array of shape (n_h, n_h + n_x)
        bo: Bias of the output gate, numpy array of shape (n_h, 1)
        Wy: Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_h)
        by: Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    y_preds: list of softmax probability outputs from each timestep
    """

    #TODO Double check cell state initialization
    h, c = encoding
    y_hat = []
    y_t = word_to_vec_map["<start>"]


    # Compute LSTM output  sequence until the answer length has been reached
    while len(y_hat) < len(qac["answer"]) + 1:
        # LSTM computation
        h, c, y_t = lstm_decoder_step(y_t, h, c, parameters)
        y_hat.append(y_t)

        # Convert softmax output into embedding for the highest probability word
        y_t = word_to_vec_map[index_to_words[torch.max(y_t, 0)[1][0].item()]]

    # Combine list of tensors into single tensor
    y_hat = torch.cat(y_hat, 1)

    return y_hat



def train(training_data, q_encoder_params, c_encoder_params, decoder_params, word_to_vec_map, words_to_index, index_to_words, name, learning_rate=0.01, batch_size=64, epochs=3, sample_size=80000):
    # Categorical crossentropy loss function
    categorical_crossentropy = lambda y, y_hat: -torch.sum(torch.mul(y, torch.log(y_hat)))/y.shape[1]

    # List to store loss at each training step
    losses = []

    # Define the adam optimizer to be used on the encoder and decoder parameters
    optimizer = torch.optim.Adam(list(q_encoder_params.values()) + list(c_encoder_params.values()) + list(decoder_params.values()), lr=learning_rate)

    # Initialize logging
    log = open("modeling/logs/%s.txt" % name, "a+")

    for i in range(epochs):
        print("Starting epoch #%d" % (i+1))
        sample = random.sample(training_data, sample_size)
        for index, qac in enumerate(sample):
            # Retrieve one_hot representation of answer
            y = []
            for word in qac["answer"] + ["<end>"]:
                if not word in words_to_index:
                    word = "<unk>"
                y.append(one_hot(words_to_index[word], len(words_to_index)))
            y = torch.cat(y, 1)

            # Create encoding of query
            q_forward, q_backward = lstm_encode(qac["question"], word_to_vec_map, q_encoder_params)

            # Concatenate the forward and backward encodings of the question
            q_encoding = (torch.cat((q_forward[0], q_backward[0]), 0), torch.cat((q_forward[1], q_backward[1]), 0))

            # Create encoding of context
            c_forward, c_backward = lstm_encode(qac["context"], word_to_vec_map, c_encoder_params)

            # Concatenate the forward and backward encodings of the question
            c_encoding = (torch.cat((c_forward[0], c_backward[0]), 0), torch.cat((c_forward[1], c_backward[1]), 0))

            # Concatenate the hidden and cell states for the question and context
            encoding = (torch.cat((q_encoding[0], c_encoding[0]), 0), torch.cat((q_encoding[1], c_encoding[1]), 0))

            # Decode the encoding into probability distributions
            y_hat = lstm_decode(qac, encoding, word_to_vec_map, index_to_words, decoder_params)

            # Compute categorical crossentropy loss
            loss = categorical_crossentropy(y, y_hat)

            # Print current training step
            print("Example #%d, Loss: %f" % (index + 1, loss.item()), end="\r")

            # Record the loss value
            losses.append(loss)

            # After every batch save the parametes and log
            if (index + 1) % batch_size == 0:
                # pickle.dump((encoder_params, decoder_params), path + ".pkl")

                # Average loss for batch
                cost = torch.sum(torch.stack(losses, dim=0))/len(losses)

                # Print the cost
                print("Batch %d/%d, Cost: %f" % (((index + 1) // batch_size), (len(training_data) // batch_size) + 1, cost))

                # Write average loss to logs
                log.write(str(cost) + "\n")

                # Reset list of losses
                losses = []

                # Gradient descent
                cost.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(list(q_encoder_params.values()) + list(c_encoder_params.values()) + list(decoder_params.values()), 10)

                # Parameter update
                optimizer.step()


                if ((index + 1) // batch_size) % 25 == 0:
                    print("Saving...", end='\r')
                    pickle.dump((q_encoder_params, c_encoder_params, decoder_params), open("modeling/saves/%s.pkl" % name, 'wb'))
                    print("Model has been saved")

        # Average loss for last batch
        cost = torch.sum(torch.stack(losses, dim=0))/len(losses)

        # Write average loss to logs
        log.write(str(cost) + "\n")

        # Gradient descent
        cost.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(list(q_encoder_params.values()) + list(c_encoder_params.values()) + list(decoder_params.values()), 10)

        # Parameter update
        optimizer.step()

    # Close log file
    log.close()

    return q_encoder_params, c_encoder_params, decoder_params
