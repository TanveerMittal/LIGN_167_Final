import torch
import random
import pickle
import numpy as np


def one_hot(index, size):
    o = np.zeros((size, 1))
    o[index,:] = 1
    return torch.from_numpy(o).type(torch.float32)

def rnn_encoder_init(n_h, n_x):
    '''
    Arguments:
    n_h: the hidden state dimension
    n_x: the embedding dimension
    Returns:
    parameters: python dictionary containing:
        Wx: Weight matrix that relates the input to the hidden state, tensor of shape (n_h, n_x)
        Wh: Weight matrix that relates the previous hidden state to the current hidden state, tensor of shape (n_h, n_h)
        bh: Bias vector for the hidden statae calculation, tensor of shape (n_h, 1)
    '''
    Wx = torch.rand((n_h, n_x), dtype=torch.float32, requires_grad=True)
    Wh = torch.rand((n_h, n_h), dtype=torch.float32, requires_grad=True)
    bh = torch.rand((n_h, 1), dtype=torch.float32, requires_grad=True)
    return {"Wx": Wx, "Wh": Wh, "bh": bh}

def rnn_encoder_step(xt, h_prev, parameters):
    '''
    Computes the output of the LSTM cell at given timestep

    Arguments:
    xt: your input data at timestep "t", numpy array of shape (n_x, m).
    h_prev: Hidden state at timestep "t-1", numpy array of shape (n_h, m)
    parameters: python dictionary containing:
        Wx: Weight matrix that relates the input to the hidden state, tensor of shape (n_h, n_x)
        Wh: Weight matrix that relates the previous hidden state to the current hidden state, tensor of shape (n_h, n_h)
        bh: Bias vector for the hidden statae calculation, tensor of shape (n_h, 1)

    Returns:
    h_next: next hidden state, of shape (n_h, 1)
    '''
    Wx = parameters["Wx"]
    Wh = parameters["Wh"]
    bh = parameters["bh"]

    # Compute and return the hidden state at this timestep
    return torch.tanh(torch.matmul(Wx, xt) + torch.matmul(Wh, h_prev) + bh)

def rnn_encode(qac, word_to_vec_map, parameters):
    """
    Computes the forward propogation process of the RNN cell to encode a question and its context

    Arguments:
    qac: python dictionary containing:
        question: A question in the form of a list of words and punctuation
        answer: An answer to the question in the form of a list of words and punctuation
        context: A paragraph of context in the form of a list of words and punctuation
    word_to_vec_map: python dictionary that maps an english word its corresponding GloVe embedding
    parameters: python dictionary containing:
        Wx: Weight matrix that relates the input to the hidden state, tensor of shape (n_h, n_x)
        Wh: Weight matrix that relates the previous hidden state to the current hidden state, tensor of shape (n_h, n_h)
        bh: Bias vector for the hidden statae calculation, tensor of shape (n_h, 1)

    Returns:
    encoding: the final LSTM hidden state that encodes the question and answer
    """

    # Create the sequence to compute on
    sequence = qac["question"] + ["<sep>"] + qac["context"]

    #TODO Double check hidden state initialization
    h = torch.zeros((parameters["Wh"].shape[0], 1), dtype=torch.float32, requires_grad=False)

    # Compute LSTM output on each word of the sentence
    for word in sequence:
        # Retrieve the embedding for the current word
        if word in word_to_vec_map:
            emb = word_to_vec_map[word]
        else:
            emb = word_to_vec_map["<unk>"]

        # LSTM computation
        h = rnn_encoder_step(emb, h, parameters)

    return h

# TODO change dimensions in docstring
def rnn_decoder_init(n_h, n_x, n_y):
    '''
    Arguments:
    n_h: the hidden state dimension
    n_x: the embedding dimension
    Returns:
    parameters: python dictionary containing:
        Wx: Weight matrix that relates the input to the hidden state, tensor of shape (n_h, n_x)
        Wh: Weight matrix that relates the previous hidden state to the current hidden state, tensor of shape (n_h, n_h)
        Wy: Weight matrix that relates the current hidden state to the softmax output, tensor of shape (n_y, n_h)
        bh: Bias vector for the hidden state calculation, tensor of shape (n_h, 1)
        by: Bias vector for softmax output calculation, tensor of shape (n_y, 1)
    '''
    Wx = torch.rand((n_h, n_x), dtype=torch.float32, requires_grad=True)
    Wh = torch.rand((n_h, n_h), dtype=torch.float32, requires_grad=True)
    Wy = torch.rand((n_y, n_h), dtype=torch.float32, requires_grad=True)
    bh = torch.rand((n_h, 1), dtype=torch.float32, requires_grad=True)
    by = torch.rand((n_y, 1), dtype=torch.float32, requires_grad=True)
    return {"Wx": Wx, "Wh": Wh, "Wy": Wy, "bh": bh, "by":by}


def rnn_decoder_step(y_prev, h_prev,  parameters):
    '''
    Computes the output of the RNN cell at given timestep

    Arguments:
    y_prev: Embedding of output at timestep "t-1", numpy array of shape (n_x, 1).
    h_prev: Hidden state at timestep "t-1", numpy array of shape (n_h, m)
    parameters: python dictionary containing:
        Wx: Weight matrix that relates the input to the hidden state, tensor of shape (n_h, n_x)
        Wh: Weight matrix that relates the previous hidden state to the current hidden state, tensor of shape (n_h, n_h)
        Wy: Weight matrix that relates the current hidden state to the softmax output, tensor of shape (n_y, n_h)
        bh: Bias vector for the hidden state calculation, tensor of shape (n_h, 1)
        by: Bias vector for softmax output calculation, tensor of shape (n_y, 1)

    Returns:
    h_next: next hidden state, of shape (n_h, 1)
    y: sotmax output for current time step, of shape(n_y, 1)
    '''
    Wx = parameters["Wx"]
    Wh = parameters["Wh"]
    Wy = parameters["Wy"]
    bh = parameters["bh"]
    by = parameters["by"]

    # Compute the current hidden state
    h = torch.tanh(torch.matmul(Wx, y_prev) + torch.matmul(Wh, h_prev) + bh)

    # Compute the current softmax output
    y = torch.softmax(torch.matmul(Wy, h) + by, 0)

    return h, y

def rnn_decode(qac, encoding, word_to_vec_map, index_to_words, parameters):
    """
    Computes the forward propogation process of the RNN model to decode an answer from an encoding

    Arguments:
    qac: python dictionary containing:
        question: A question in the form of a list of words and punctuation
        answer: An answer to the question in the form of a list of words and punctuation
        context: A paragraph of context in the form of a list of words and punctuation
    encoding: lstm encoder output that encodes a query and its context as hidden and cell states
    parameters: python dictionary containing:
        Wx: Weight matrix that relates the input to the hidden state, tensor of shape (n_h, n_x)
        Wh: Weight matrix that relates the previous hidden state to the current hidden state, tensor of shape (n_h, n_h)
        Wy: Weight matrix that relates the current hidden state to the softmax output, tensor of shape (n_y, n_h)
        bh: Bias vector for the hidden state calculation, tensor of shape (n_h, 1)
        by: Bias vector for softmax output calculation, tensor of shape (n_y, 1)

    Returns:
    y_preds: list of softmax probability outputs from each timestep
    """

    #TODO Double check cell state initialization
    h = encoding
    y_hat = []
    y_t = word_to_vec_map["<start>"]


    # Compute LSTM output  sequence until the answer length has been reached
    while len(y_hat) < len(qac["answer"]) + 1:
        # LSTM computation
        h, y_t = rnn_decoder_step(y_t, h, parameters)
        y_hat.append(y_t)

        # Convert softmax output into embedding for the highest probability word
        y_t = word_to_vec_map[index_to_words[torch.max(y_t, 0)[1][0].item()]]

    # Combine list of tensors into single tensor
    y_hat = torch.cat(y_hat, 1)

    return y_hat



def train(training_data, encoder_params, decoder_params, word_to_vec_map, words_to_index, index_to_words, name, learning_rate=0.01, batch_size=64, epochs=3, sample_size=80000):
    # Categorical crossentropy loss function
    categorical_crossentropy = lambda y, y_hat: -torch.sum(torch.mul(y, torch.log(y_hat)))/y.shape[1]

    # List to store loss at each training step
    losses = []

    # Define the adam optimizer to be used on the encoder and decoder parameters
    optimizer = torch.optim.Adam(list(encoder_params.values()) + list(decoder_params.values()), lr=learning_rate)

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
            
            # Create encoding of query and context
            encoding = rnn_encode(qac, word_to_vec_map, encoder_params)

            # Decode the encoding into probability distributions
            y_hat = rnn_decode(qac, encoding, word_to_vec_map, index_to_words, decoder_params)

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
                torch.nn.utils.clip_grad_norm_(list(encoder_params.values()) + list(decoder_params.values()), 10)

                # Parameter update
                optimizer.step()

                if ((index + 1) // batch_size) % 25 == 0:
                    print("Saving...", end='\r')
                    pickle.dump((encoder_params, decoder_params), open("modeling/saves/%s.pkl" % name, 'wb'))
                    print("Model has been saved")

        # Average loss for last batch
        cost = torch.sum(torch.stack(losses, dim=0))/len(losses)

        # Write average loss to logs
        log.write(str(cost) + "\n")

        # Gradient descent
        cost.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(list(encoder_params.values()) + list(decoder_params.values()), 10)

        # Parameter update
        optimizer.step()

    # Close log file
    log.close()

    return encoder_params, decoder_params
