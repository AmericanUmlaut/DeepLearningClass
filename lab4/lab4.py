#! /usr/bin/env python

"""
@authors: Benjamin Stuermer, Romarie Morales, Sarah Reehl

This script trains and evaluates a convolutional neural network for
digit recognition on MNIST.

For usage, run with the -h flag.

Example command:

python lab4.py data/train.csv.gz data/dev.csv.gz

"""

import tensorflow as tf
import argparse
import sys
import numpy as np

def conv2d(x,k,L,Lprime,S,P,f):
    """
    Creates the weights for a 2d convolution layer and adds the 
    corresponding convolutional layer to the graph.

    :param x: (4d tensor) tensor of inputs with dim (MB x W x W x L)
    :param k: (integer) the receptive fields will be k x k, spatially
    :param L: (integer) the input channels
    :param Lprime: (integer) the number of kernels to apply
    :param S: (integer) the stride (will be used in both spatial dims)
    :param P: (string) either "SAME" or "VALID" (specifies padding strategy)
    :param f: (string) the hidden activation (relu, tanh or linear)

    :return: (4d tensor) the result of the convolutional layer
                         (will be MB x Wprime x Wprime x Lprime)
    """

    # convolution weights (a k x k x L x Lprime 4d tensor)
    W = tf.get_variable(name="W",\
                        shape=(k,k,L,Lprime),\
                        dtype=tf.float32,\
                        initializer=tf.glorot_uniform_initializer()) 

    # pick activation and modify bias constant, if needed
    b_const = 0.0
    if f == "relu":
        b_const = 0.1
        act = tf.nn.relu
    elif f == "tanh":
        act = tf.nn.tanh
    elif f == "identity":
        act = tf.identity
    else:
        sys.exit("Error: Invalid f (%s)" % f)

    # bias weights (a Lprime dim vector)
    b = tf.get_variable(name="b",
                        shape=(Lprime),
                        initializer=tf.constant_initializer(b_const))

    # tf.nn.conv2d does the heavy lifting for us
    z = tf.nn.conv2d(x,W,strides=[1,S,S,1],padding=P)
    z = z + b # don't forget the bias!

    a = act(z)

    return a

def max_pool(x,k,L,Lprime,S,P,f):
    """
    Adds a max pooling layer to the graph.

    :param x: (4d tensor) tensor of inputs with dim (MB x W x W x L)
    :param k: (integer) will pool over k x k spatial regions
    :param S: (integer) the stride (will be used in both spatial dims)

    Other parameters are ignored and are used to give max_pool the same signature as conv2d

    :return: (4d tensor) the result of the max pooling layer
                         (will be MB x Wprime x Wprime x L)
    """

    # tf.nn.max_pool does the heavy lifting for us
    # note: using SAME padding makes the dimensionality reduction
    #       easier to compute (e.g. if k=2 and S=2, Wprime = W/2)
    return tf.nn.max_pool(x,[1,k,k,1],[1,S,S,1],padding="SAME")

def add_layer(layer_name, layer_method, prev_layer, k, L=None, Lprime=None, S=None, P=None, f=None):
    with tf.variable_scope(layer_name):
        return layer_method(prev_layer, k=k, L=L, Lprime=Lprime, S=S, P=P, f=f)

def build_graph(lr, model):
    """
    Adds a CNN to the graph.

    :param args: (string) the parsed argument object

    """

    # placeholders
    y_true = tf.placeholder(dtype=tf.int64,shape=(None),name="y_true")
    x = tf.placeholder(dtype=tf.float32,shape=(None,28,28,1),name="x")
    a0 = x # for notational simplicity/consistency

    if (model == 1 or model == 2):
        a1 = add_layer("layer_1_conv",    conv2d,   a0, k=5, L=1,   Lprime=32,  S=1, P="SAME",  f="relu")
        a2 = add_layer("layer_2_maxpool", max_pool, a1, k=2, S=2)
        a3 = add_layer("layer_3_conv",    conv2d,   a2, k=5, L=32,  Lprime=64,  S=1, P="SAME",  f="relu")
        a4 = add_layer("layer_4_maxpool", max_pool, a3, k=2, S=2)

        if (model == 1):
            a5 = add_layer("layer_5_fc",  conv2d,   a4, k=7, L=64,  Lprime=200, S=1, P="VALID", f="relu")
            z  = add_layer("layer_6_fc",  conv2d,   a5, k=1, L=200, Lprime=10,  S=1, P="VALID", f="identity")
        else:
            z  = add_layer("layer_6_fc",  conv2d,   a4, k=7, L=64,  Lprime=10,  S=1, P="VALID", f="identity")

    elif (model==3):
        a1 = add_layer("layer_1_conv",    conv2d,   a0, k=3, L=1,   Lprime=64,  S=1, P="SAME",  f="relu")
        a2 = add_layer("layer_2_maxpool", max_pool, a1, k=4, S=4)
        a3 = add_layer("layer_3_conv",    conv2d,   a2, k=7, L=64,  Lprime=200, S=1, P="VALID", f="relu")
        z  = add_layer("layer_4_fc",      conv2d,   a3, k=1, L=200, Lprime=10,  S=1, P="VALID", f="identity")

    elif (model == 4):
    	a1 = add_layer("layer_1_conv",    conv2d,   a0, k=3, L=1,   Lprime=32,  S=1, P="SAME",  f="relu")
    	a2 = add_layer("layer_2_max_pool",max_pool, a1, k=2, S=2)
    	a3 = add_layer("layer_3_conv",    conv2d,   a2, k=3, L=32,  Lprime=64,  S=1, P="SAME",  f="relu")
    	a4 = add_layer("layer_4_max_pool",max_pool, a3, k=2, S=2)
    	a5 = add_layer("layer_5_conv",    conv2d,   a4, k=3, L=64,  Lprime=64,  S=1, P="SAME",  f="relu")
    	a6 = add_layer("layer_6_conv",    conv2d,   a5, k=7, L=64,  Lprime=200, S=1, P="VALID", f="relu")
    	z  = add_layer("layer_7_fc",      conv2d,   a6, k=1, L=200, Lprime=10,  S=1, P="VALID", f="identity")

    elif (model == 5):
    	a1 = add_layer("layer_1_conv",  conv2d,   a0, k=28, L=1,   Lprime=20, S=1, P="VALID",  f="relu")
    	z  = add_layer("layer_3_fc",    conv2d,   a1, k=1,  L=20,  Lprime=10, S=1, P="VALID",  f="identity")

    # z is MB x 1 x 1 x 10, now we squeeze it to MB x 10
    z = tf.squeeze(z)

    # define loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z,labels=y_true)
    obj = tf.reduce_mean(cross_entropy,name="obj")

    # optimizer
    train_step = tf.train.AdamOptimizer(lr).minimize(obj,name="train_step")

    # define accuracy
    acc = tf.reduce_mean(tf.cast(tf.equal(y_true,tf.argmax(z,axis=1)),tf.float32),name="acc")

    # define init op last
    init = tf.global_variables_initializer()

    return init

def parse_all_args():
    """
    Parses arguments

    :return: the parsed arguments object
    """
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("train",help="The training set file (a csv)")
    parser.add_argument("dev",help="The development set file (a csv)")
    parser.add_argument("model",help="The model specifier (an int) [default = 1]",type=int,default=1)
    parser.add_argument("-lr",type=float,\
            help="The learning rate (a float) [default = 0.001]",default=0.001)
    parser.add_argument("-mb",type=int,\
            help="The minibatch size (an int) [default = 32]",default=32)
    parser.add_argument("-epochs",type=int,\
            help="The number of epochs to train for [default = 5]",default=5)

    return parser.parse_args()

def load_and_normalize(filename, savefile_prefix):
    """
        Load and normalize data from a file. If the special file name "saved" is passed, the previously loaded file
        (which is saved using Numpy's save() function) will be reloaded, which is much more performant than loading from
        a CSV file.

        Returns two values - the first is the X and the second the Y values from the file
    """

    savefile = savefile_prefix + '_savey_mc_saversave.npy'
    if filename == "saved":
        values = np.load(savefile)
        print "Loaded values from " + savefile
    else:
        values = np.genfromtxt(filename,delimiter=",")
        np.save(savefile, values)
        print "Saved values to " + savefile

    x = values[:,1:] / 255.0 # Division maps range 0-255 values to range 0-1
    y = values[:,0]

    return x, y

def main():
    """
    Parse arguments, build CNN, run training loop, report dev each epoch.

    """
    # parse arguments
    args = parse_all_args()

    train_x, train_y = load_and_normalize(args.train, 'train')
    N,D = train_x.shape

    dev_x, dev_y = load_and_normalize(args.dev, 'dev')

    # reshape dev once (train will be reshaped each MB)
    # (our graph assumes tensor-shaped input: N x W x W x L)
    dev_x = np.reshape(dev_x,(-1,28,28,1))

    # build graph
    init = build_graph(args.lr, args.model)

    # train
    with tf.Session() as sess:
        sess.run(fetches=[init]) # passing as tensor variable rather than name
                                 # just to mix things up

        for epoch in range(args.epochs):
            # shuffle data once per epoch
            idx = np.random.permutation(N)
            train_x = train_x[idx,:]
            train_y = train_y[idx]

            # train on each minibatch
            for update in range(int(np.floor(N/args.mb))):
                mb_x = train_x[(update*args.mb):((update+1)*args.mb),:]
                mb_x = np.reshape(mb_x,(args.mb,28,28,1)) # reshape vector into tensor

                mb_y = train_y[(update*args.mb):((update+1)*args.mb)]

                # note: using strings for fetches and feed_dict
                _,my_acc = sess.run(fetches=["train_step","acc:0"],\
                        feed_dict={"x:0":mb_x,"y_true:0":mb_y}) 

            # evaluate once per epoch on dev set
            [my_dev_acc] = sess.run(fetches=["acc:0"],\
                    feed_dict={"x:0":dev_x,"y_true:0":dev_y})
            
            print("Epoch %d: dev=%.5f" % (epoch,my_dev_acc))
    

if __name__ == "__main__":
    main()
