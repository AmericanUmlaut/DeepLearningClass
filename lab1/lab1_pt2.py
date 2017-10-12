import tensorflow as tf
import numpy as num
import sys

file_name = sys.argv[1]
k = int(sys.argv[2])

with open(file_name) as f:
    vals = f.read().splitlines()

sess = tf.Session()

for val in vals:
    explist = list(range(1, k + 1))
    exponents = tf.cast(tf.constant(explist), dtype=tf.float32)
    tensor = tf.pow(float(val), exponents)
    results = tensor.eval(session=sess)
    print(" ".join(["%.5f" % r for r in results]))
