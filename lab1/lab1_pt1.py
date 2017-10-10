import tensorflow as tf
import numpy as num
import sys

a = float(sys.argv[1])
b = float(sys.argv[2])
learn_rate = float(sys.argv[3])
optimizer = sys.argv[4]

# TODO: change w1, w2 into a single constant [0, 0]^T
w = tf.get_variable(name='w', shape=[2, 1], dtype=tf.float32, initializer=tf.zeros_initializer)

# f(w) = (a - w1)^2 + b(w2 - w1^2)^2
objective = tf.add( tf.square(a-w[0]), b*tf.square(w[1] - w[0]*w[0]), name='objective' )

train_optimizer = tf.train.GradientDescentOptimizer(learn_rate) if optimizer == "gd" \
             else tf.train.MomentumOptimizer(learn_rate, 0.9) if optimizer == "gdm"           \
             else tf.train.AdamOptimizer(learn_rate)
train_step = train_optimizer.minimize(objective, name="train_step")

sess = tf.Session()
initializer = tf.global_variables_initializer()
initializerValue = sess.run(fetches=[initializer])

i = 0
while True:
    results = [ts_res, w_res, obj_res] = sess.run(fetches=[train_step, w, objective])
    print (str(i) + ': ' + str(obj_res))
    i = i + 1
    if (num.isinf(obj_res) or num.isnan(obj_res) or obj_res < 0.0001):
        break
