#! /usr/bin/env python

"""
@authors: Brian Hutchinson (Brian.Hutchinson@wwu.edu)

This script trains and evaluates a recurent neural network for
language modeling on the Penn Treebank corpus.

Demo only supports BPTT, *not* truncated BPTT.

For usage, run with the -h flag.

Example command:

python lab5_demo.py simple-examples/data vocab.txt

"""

import tensorflow as tf
import argparse
import sys
import numpy as np
import pprint

def build_graph(args,vocab_size):
    """
    Constructs the RNN graph.

    :param args: the parsed argument object
    :param vocab_size: (integer) the number of tokens in the vocabulary

    :return: (op) the variable inititalizer operation
    """

    # define the RNN cell whose weights will be reused at each timestep
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(args.L)

    # define placeholders
    h0     = tf.placeholder(dtype=tf.float32,shape=(None,args.L),name="h0")
    lens   = tf.placeholder(dtype=tf.int64,shape=(None,),name="lens")
    x_ints = tf.placeholder(dtype=tf.int64,shape=(None,None),name="x_ints")
    y_true = tf.placeholder(dtype=tf.int64,shape=(None,None),name="y_true")
    # embedding matrix for words
    W = tf.get_variable(name="W",
                        shape=(vocab_size,args.emb),
                        dtype=tf.float32,
                        initializer=tf.glorot_uniform_initializer())

    # the input to our RNN are learned embedding vectors
    # x should be mb x time x args.emb
    x = tf.nn.embedding_lookup(W,x_ints,name="x")

    # dynamic_rnn produces
    #    outputs is [mb,num_steps,args.L] (per time outputs for each seq)
    #    state is   [mb,args.L]               (the last hidden state per seq)
    outputs,state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                      inputs=x,
                                      sequence_length=lens,
                                      initial_state=h0,
                                      dtype=tf.float32)

    tf.identity(state,name="state") # naming it only so we can fetch by name

    # hidden to output layer weights
    softmax_W = tf.get_variable(name="softmax_W",
                                shape=(args.L,vocab_size),
                                dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())
    softmax_b = tf.get_variable(name="softmax_b",
                                shape=(vocab_size),
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())

    # apply hidden-to-output layer to each seq in mb and each timestep
    # (maps [mb,num_steps,args.L] to [mb,num_steps,vocab_size])
    logits = tf.tensordot(outputs,softmax_W,[[2],[0]]) + softmax_b

    # compute loss over all num_steps and all sequences in the minibatch
    # produces a [mb,num_steps] tensor
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=logits)

    # produce a [mb,num_steps] binary mask with 1s for valid y_true
    # tokens and 0s for padding y_true tokens
    mask = tf.sequence_mask(lens,maxlen=tf.shape(x_ints)[1],
                            dtype=tf.float32,name="mask")

    # mask our losses so we're not penalized for padding token losses
    unnorm_obj = tf.reduce_sum(losses * mask,name="unnorm_obj")

    # compute the denominator (so that we can compute average loss
    # averaged only over the valid loss tokens
    denom      = tf.reduce_sum(mask,name="denom")

    # useful to minimize average loss because the gradient magnitude
    # isn't affected by how many valid tokens happened to be in this minibatch
    norm_obj = unnorm_obj / denom
    train_step = tf.train.AdamOptimizer(args.lr).minimize(norm_obj,
                                                          name="train_step")

    # perplexity (ppl) is a useful metric to report for language modeling
    # (a uniform prediction distribution gives vocab_size perplexity)
    ppl = tf.exp(norm_obj,name="ppl")

    # initializer
    init = tf.global_variables_initializer()

    return init

def parse_all_args():
    """
    Parses arguments

    :return: the parsed arguments object
    """
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("train",help="The training set file (space delimited)")
    parser.add_argument("dev",help="The development set file (space delimited)")
    parser.add_argument("vocab",help="A file containing the vocabulary")
    parser.add_argument("-lr", type=float,
      help="The learning rate (a float) [default=0.001]",
      default=0.001)
    parser.add_argument("-mb",type=int,
      help="The minibatch size (an int) [default=128]",
      default=128)
    parser.add_argument("-num-steps", metavar = "NUM_STEPS", help="The number of steps to unroll for Truncated BPTT (an int) [default=20]", type = int, default = 20)
    parser.add_argument("-max-len",type=int,
      help="The maximum length of a sequence (an int) [default=101]",
      default=101)
    parser.add_argument("-L",type=int,
      help="The hidden layer dimension (an int) [default=100]",
      default=100)
    parser.add_argument("-emb",type=int,
      help="The word embedding dimension (an int) [default=50]",
      default=50)
    parser.add_argument("-epochs",type=int,
      help="The number of epochs to train for (an int) [default=20]",
      default=20)

    return parser.parse_args()

def load_vocab(vocab_fn):
    """
    Reads the vocab file, producing a list that maps for integer to token
    and a dictionary that maps from token to integer.

    :param vocab_fn: (string) the filename of the vocab file

    :return: (list,dictionary) the idx2word list and word2idx dictionary
    """

    # idx2word is a just a list of the vocab tokens
    with open(vocab_fn) as f:
        idx2word = [line.strip() for line in f]

    # word2idx is the inverse mapping
    word2idx = {}
    for i,word in enumerate(idx2word):
        word2idx[word] = i

    return idx2word,word2idx

def load_set(fn,max_len,word2idx):
    """
    Reads a dataset file (e.g. train), producing 1) a 2d ndarray where
    each line is a sentence and each field in a line is the integer
    representation of that token in the setence, and 2) an 1d ndarray
    listing the "length" of each sentence (includes <s>, and will be
    up to max_len).  Does padding and truncating as needed.

    :param fn: (string) the filename of the dataset file
    :param max_len: (integer) the maximum length permissable for a sentence
                              larger sentences will be truncated
    :param word2idx: (dictionary) the dictionary mapping words to integers

    :return: (2d array,1d array) the two return values described above
    """

    seq_lengths = []
    sequences   = []

    with open(fn) as f:
        # process file
        for line in f:
            # process line

            # first prepend <s>'s integer, then map remaining tokens to
            # integers and append
            token_line = [word2idx["<s>"]] + \
                    [word2idx[word] for word in line.strip().split(' ')]
            orig_len = len(token_line) # includes <s> but not yet </s>

            if( orig_len > max_len ):
                # need to truncate (won't have an </s>)
                seq_lengths.append(max_len)
                sequences.append(token_line[0:max_len])
            else:
                # if we're padding, pad with </s> (the first one "counts")
                seq_lengths.append(orig_len+1) # since we will add </s>
                sequences.append(token_line + \
                        ([word2idx["</s>"]] * (max_len-orig_len)))

    return np.array(sequences),np.array(seq_lengths)

def score_set(args,seqs,seq_lens,sess,name):
    """
    Feeds an entire dataset through the model and computes perplexity.
    Does *not* do any MB over sequences: may not fit in GPU memory.

    :param args: the parsed arguments object
    :param seqs: (2d ndarray) the padded/truncated dataset as integers
    :param seq_lens: (1d ndarray) the sequence lengths for each sentence
    :param sess: (session object) the current tensorflow session
    :param name: (string) the name of the dataset (e.g. "dev" or "train")

    """

    hidden_states = np.zeros(shape=(seqs.shape[0],args.L),dtype=np.float32)

    my_obj,my_denom = sess.run(
            fetches=["unnorm_obj:0","denom:0"],
            feed_dict={"x_ints:0":seqs[:,0:(args.max_len-1)],
                "y_true:0":seqs[:,1:args.max_len],
                "h0:0":hidden_states,
                "lens:0":(seq_lens)})

    print(name + " ppl = %.5f" % np.exp((my_obj/my_denom)))

def main():
    """
    Parse arguments, build RNN, run training loop, report dev ppl each epoch.

    """
    # parse arguments
    args = parse_all_args()

    # we only feed in <s> but do not make a prediction on it
    # the length of our unrolled graph, num_steps, should be max_len-1
    if ( (args.max_len-1) % args.num_steps) != 0:
        sys.exit("Error: max_len-1 is not a multiple of num_steps. max_len is " + str(args.max_len) + ", num_steps is " + str(args.num_steps))

    # load vocab
    idx2word,word2idx= load_vocab(args.vocab)

    # load data
    train_seqs,train_seq_lens = load_set(args.train,args.max_len,word2idx)
    dev_seqs,dev_seq_lens     = load_set(args.dev,args.max_len,word2idx)
    max_train_seq_len         = max(train_seq_lens)

    N = train_seq_lens.shape[0]  # number of training sequences
    V = len(idx2word)            # vocab size

    # build unrolled graph
    init = build_graph(args,V)

    # train
    with tf.Session() as sess:
        # init our models
        sess.run(fetches=[init])

        batches_high = int(np.floor(N / args.mb))
        batches_wide = int((max_train_seq_len-1) / args.num_steps)

        print("minibatches per epoch: %d * %d = %d" % (batches_high, batches_wide, (batches_high * batches_wide)))

        for epoch in range(args.epochs):
            # shuffle sequencing of sentences (without reordering words within)
            idx = np.random.permutation(N)
            train_seqs = train_seqs[idx,:]
            train_seq_lens = train_seq_lens[idx]

            print("epoch %d update... " % epoch)

            for mb_row in range(batches_high):
                row_start = mb_row * args.mb # Top of minibatch
                row_end   = np.min([(mb_row+1)*args.mb, N]) # Bottom of minibatch
                                                            # last minibatch
                                                            # might be partial
                if( mb_row % 100 == 0 ):
                        print("%d... " % (mb_row))
                # If we are not in our first column, we will pass our previous outputs through to be the initial
                # inputs to the next minibatch. Otherwise the initial inputs are all 0.
                mb_h0 = np.zeros(shape=((row_end-row_start),args.L),dtype=np.float32)

                for mb_col in range(batches_wide):
                    col_start = mb_col * args.num_steps # Left of minibatch
                    col_end   = (mb_col + 1) * args.num_steps # Right of minibatch

                    # pull our minibatch of data
                    #print("Row: " + str(row_start) + ":" + str(row_end));
                    #print("Col: " + str(col_start) + ":" + str(col_end));

                    mb_x    = train_seqs[row_start:row_end,col_start:col_end]
                    mb_y    = train_seqs[row_start:row_end,col_start+1:col_end+1]
                    mb_lens = train_seq_lens[row_start:row_end]

                    # We fetch state and pass it to mb_h0 so that the resulting state from each minibatch is passed
                    # through to become the initial state of the subsequent minibatch in the same row.
                    [train_step] = sess.run(
                            fetches=["train_step"],
                            feed_dict={"x_ints:0":mb_x,
                                       "y_true:0":mb_y,
                                       "h0:0":mb_h0,
                                       "lens:0":mb_lens})

                    # return last hidden weights/biases for next minibatch
                    [mb_h0] = sess.run(
                            fetches=["state:0"],
                            feed_dict={"x_ints:0":mb_x,
                                       "y_true:0":mb_y,
                                       "h0:0":mb_h0,
                                       "lens:0":mb_lens})


                    

            print("training loop finished.")


            # training loop done for epoch, now score on dev
            # (uncomment train line to see how its ppl evolves)
            #score_set(args,train_seqs,train_seq_lens,sess,"train")
            score_set(args,dev_seqs,dev_seq_lens,sess,"dev")


if __name__ == "__main__":
    main()