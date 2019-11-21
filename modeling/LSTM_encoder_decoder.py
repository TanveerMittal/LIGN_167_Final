import torch
import numpy as np

def one_hot(index, size):
    o = np.zeros((size, 1))
    o[index,:] = 1
    return torch.from_numpy(o).type(torch.float32)

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

def lstm_encode(qac, parameters, word_to_vec_map):
    """
    Computes the forward propogation process of the LSTM cell to encode a question and its context

    Arguments:
    qac: python dictionary containing:
        question: A question in the form of a list of words and punctuation
        answer: An answer to the question in the form of a list of words and punctuation
        context: A paragraph of context in the form of a list of words and punctuation
    word_to_vec_map: python dictionary that maps an english word its corresponding GloVe embedding
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
    encoding: the final LSTM hidden state that encodes the question and answer
    """

    # Create the sequence to compute on
    sequence = qac["question"] + ["<sep>"] + qac["context"]

    #TODO Double check hidden state initialization
    h = torch.rand((parameters["Wf"].shape[0], 1), dtype=torch.float32, requires_grad=False)
    c = torch.rand(h.shape, dtype=torch.float32, requires_grad=False)

    # Compute LSTM output on each word of the sentence
    for word in sequence:
        # Retrieve the embedding for the current word
        if word in word_to_vec_map:
            emb = word_to_vec_map[word]
        else:
            emb = word_to_vec_map["<unk>"]

        # LSTM computation
        h, c = lstm_encoder_step(emb, h, c, parameters)

    return h

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
        bc:  Bias of the first "tanh", numpy array of shape (n_h, 1)
        Wo: Weight matrix of the output gate, numpy array of shape (n_h, n_h + n_x)
        bo:  Bias of the output gate, numpy array of shape (n_h, 1)
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
    y_t = torch.softmax(torch.matmul(Wy, h_next) + by)

    return h_next, c_next, y_t

def lstm_decode(qac, encoding, parameters, word_to_vec_map):
    """
    Computes the forward propogation process of the LSTM cell to decode an answer from an encoding

    Arguments:
    qac: python dictionary containing:
        question: A question in the form of a list of words and punctuation
        answer: An answer to the question in the form of a list of words and punctuation
        context: A paragraph of context in the form of a list of words and punctuation
    encoding: lstm encoder output that encodes a query and its context
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
    y: list of softmax probability outputs from each timestep
    """

    #TODO Double check cell state initialization
    h = encoding
    c = torch.rand(h.shape, dtype=torch.float32, requires_grad=False)
    y = []
    y_t = one_hot(words_to_index["<start>"], len(words_to_index))


    # Compute LSTM output  sequence until the answer length has been reached
    while len(y) < len(qac["answer"]) + 1:
        # LSTM computation
        h, c, y_t = lstm_decoder_step(y_t, h, c, parameters)
        y.append(y_t)

    return y



def train(training_data, encoder_params, decoder_params, word_to_vec_map, words_to_index, path, learning_rate=0.01, epochs=3):
    # Categorical crossentropy loss function
    single_categorical_crossentropy = lambda y, y_hat: -torch.sum(torch.mul(y, torch.log(y_hat)))

    # List to store loss at each training step
    losses = []

    # Define the adam optimizer to be used on the encoder and decoder parameters
    optimizer = torch.optim.Adam(list(encoder_params.values()) + list(decoder_params.values()), lr=learning_rate)

    for i in range(epochs):
        print("Starting epoch #%d" % (i+1))
        for qac in training_data:
            # Retrieve one_hot representation of answer
            y = torch.tensor([one_hot(words_to_index[word], len(words_to_index)) for word in qac["answer"]], dtype=torch.float32)

            # Create encoding of query and context
            encoding = lstm_encode(qac, encoder_params)

            # Decode the encoding into probability distributions
            y_hat = lstm_decode(qac, encoding, decoder_params)

            # Compute categorical crossentropy loss
            loss = single_categorical_crossentropy(y, y_hat)

            # Gradient descent
            loss.backward()

            # Parameter update
            optimizer.step()

            # Record the loss value
            losses.append(loss)

    # TODO write losses to file
    f = open(path, "a+")
    for l in losses:
        f.write(str(l) + "\n")
    f.close()

    return encoder_params, decoder_params
