#! /usr/bin/env python

"""
@authors: Brian Hutchinson (Brian.Hutchinson@wwu.edu)

This script trains and evaluates a multinomial logistic regression model.

For usage, run with the -h flag.

Example command:

python lab2_demo.py data/train.csv.gz data/dev.csv.gz

"""

import tensorflow as tf
import argparse
import sys
import numpy as np


def construct_layer(number_of_nodes, X, f, layer_name):
    """
    :param number_of_nodes: (string) number of layer hidden units
    :X: (tensor) layer input
    :param f: (string) activation function
    """
     # variables
    D = tf.size(X) #this only works when X is one dimensional

    if f == "relu":
        b_initializer = tf.constant_initializer(0.1)
    else:
        b_initializer = tf.zeros_initializer()
    
  
    with tf.variable_scope(layer_name):
        W = tf.get_variable(name="W",shape=(D,number_of_nodes),dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name="b",shape=(number_of_nodes),dtype=tf.float32, initializer=b_initializer)
    
    # producing logits
    # note: x is NxD (for some # datapoints N), W is DxC
    #       python automatically *broadcasts* the C dim b to NxC
    #       the result is an NxC matrix z (logits for the N inputs)
    # linear transformation    
    z = tf.matmul(X,W) + b
    # non-linear transformation
    if f == "sigmod":
        z = tf.sigmoid(z)
    elif f == "relu":
        z = tf.nn.relu(z)
    else :
        z = tf.tanh(z) 
    return(z)
    
def build_graph(lr,D,C,f,hidden_dims,classify):
    """
    Constructs the multinomial logistic regression graph.

    :param lr: (float) the learning rate
    :param D: (integer) the input feature dimension
    :param C: (integer) the number of classes
    :param f: (string) hidden activation function 
    :hidden_dims (string) string dimension of hidden layers
    """

    # placeholders
    # note: None leaves the dimension unspecified, allowing us to
    #       feed in either minibatches (during training) or the dev set
    x      = tf.placeholder(dtype=tf.float32,shape=(None,D),name="x")
    y_true = tf.placeholder(dtype=tf.int64,shape=(None),name="y_true")
    
    # make hidden layers
    for i,dim in hidden_dims:
        #x.hide() this doesn't work :(
        x = construct_layer(number_of_nodes=dim, X=x, f=f, layer_name=str(i))
    
    
    # make output layer
    if classify:
        W2 = tf.get_variable(name="W2",shape=(hidden_layer_node_count,C),dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        b2 = tf.get_variable(name="b2",shape=(C),dtype=tf.float32, initializer=tf.zeros_initializer())
        z2 = tf.matmul(z,W2) + b2

        # loss (performs softmax implicitly for us - supports minibatches)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z2,labels=y_true)
        obj = tf.reduce_mean(cross_entropy,name="obj")
    
        # additional metrics (accuracy)
        # note: the mean is required because it produces N binary values
        #       one per datapoint fed into the graph
        acc = tf.reduce_mean(tf.cast(tf.equal(y_true,tf.argmax(z2,axis=1)),tf.float32),name="acc")
        
    else
      
    # side effect operations
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(obj,name="train_step")

    init = tf.global_variables_initializer()


def main():
    """
    Parses args, loads and normalizes data, builds graph, trains and evaluates.

    """
    # parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("train",help="The training set file (a csv)")
    parser.add_argument("dev",help="The development set file (a csv)")
    group = parser.add_mutually_exclusive_group(required = True) # one of the arguments need to be true
    group.add_argument('--classify', action = 'store_true')# this is the mutually exclusive statement
    group.add_argument('--autoencode', action = 'store_true')
    parser.add_argument("-f",metavar="F", type=str,default="relu",help="Hidden activation (in the set {sigmoid, tanh, relu})")    
    parser.add_argument("-lr",type=float,metavar="LR",help="The learning rate (a float)",default=0.1)
    parser.add_argument("-mb",metavar = "MB",type=int,help="The minibatch size (an int)",default=32)
    parser.add_argument("hidden-dims", metavar = "HIDDEN_DIMS", help = "A string of with the dimension of each hidden layer,comma-delimited (string)", default = "50, 50", type = str)    
    #parser.add_argument("-L",metavar = "L",type=int, default=50, help="The number of hidden units (int)")
    parser.add_argument("-epochs",type=int,help="The number of epochs to train for",default=100)
    parser.add_argument("-patience", metavar = 'PATIENCE',type = int,help ="How many epochs to continue training without improving dev accuracy",default = 10)
    parser.add_argument("-model",metavar="MODEL",type=str,help="Save the best model with this prefix (string)",default = "/tmp/model.ckpt")

    args = parser.parse_args()
    
    # change hidden_dims to an array
    args.hidden_dims = args.hidden_dims.split(",")
    # check arguments
    if args.f not in ["sigmoid", "tanh", "relu"]:
        sys.exit("Error: invalid activation function")

    # load and normalize data
    train = np.genfromtxt(args.train,delimiter=",")
    train_x = train[:,1:]
    train_y = train[:,0]
    train_x /= 255.0 # map from [0,255] to [0,1]

    dev   = np.genfromtxt(args.dev,delimiter=",")
    dev_x = dev[:,1:]
    dev_y = dev[:,0]
    dev_x /= 255.0 # map from [0,255] to [0,1]
    
    # compute relevant dimensions
    C = np.max(train_y)+1 # warning: would fail if highest class number
                          #          didn't appear in train
    N,D = train_x.shape

    # build graph
    build_graph(args.lr,D,C,args.f,args.hidden_dims, args.classify)

    # run graph
    with tf.Session() as sess:
        # storage for accuracy
        bst_dev_acc = 0.0
        bad_count = 0
        savey_mc_saversave = tf.train.Saver()

        sess.run(fetches=["init"]) # note: can specify fetches by tensor/op name

        for epoch in xrange(args.epochs):
            # shuffle data once per epoch
            idx = np.random.permutation(N)
            train_x = train_x[idx,:]
            train_y = train_y[idx]

            # train on each minibatch
            for update in xrange(int(np.floor(N/args.mb))):
                mb_x = train_x[(update*args.mb):((update+1)*args.mb),:]
                mb_y = train_y[(update*args.mb):((update+1)*args.mb)]
                # note: you can use tensor names in feed_dict
                _,my_acc = sess.run(fetches=["train_step","acc:0"],\
                        feed_dict={"x:0":mb_x,"y_true:0":mb_y}) 
            # evaluate once per epoch on dev set
            [my_dev_acc] = sess.run(fetches=["acc:0"],\
                    feed_dict={"x:0":dev_x,"y_true:0":dev_y})
            if my_dev_acc > bst_dev_acc :
                bst_dev_acc = my_dev_acc
                bad_count = 0
                savey_mc_saversave.save(sess, args.model)
            else :
                bad_count += 1
                if bad_count > args.patience :
                    print("Converged due to early stopping...")
                    sys.exit(0)
            print "Epoch %d: dev=%.5f badcount=%d" % (epoch,my_dev_acc,bad_count)
            
        

if __name__ == "__main__":
    main()
