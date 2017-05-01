#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 10:53:45 2017

@author: joey
"""

import numpy as np
import tensorflow as tf
X = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.placeholder(tf.float32, shape=[None, 1])
tf.InteractiveSession()

W = tf.get_variable("weights", (1, 1), initializer=tf.random_normal_initializer())
b = tf.get_variable("bias", (1, ), initializer=tf.constant_initializer(0.0))
y_pred = tf.matmul(X, W) + b

loss = tf.reduce_sum((y_pred - y)**2 / 5)
opt = tf.train.AdamOptimizer(0.01)
opt_operation = opt.minimize(loss)
X_batch = [2, 7, 5, 11, 14]
y_batch = [19, 62, 37, 94, 120]
X_batch = np.array(X_batch)
y_batch = np.array(y_batch)
X_batch = np.reshape(X_batch, [5, 1])
y_batch = np.reshape(y_batch, [5, 1])


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(20000):
    _, loss_value = sess.run([opt_operation, loss], {X:X_batch, y:y_batch})

sess.run(W)
sess.run(b)