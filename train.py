from __future__ import division
import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from time import time

from model import CNN
from util import load_image, chunker
from params import TrainingParams, HyperParams, CNNParams  

tparam = TrainingParams(verbose=True)  
hyper  = HyperParams(verbose=True)
cparam = CNNParams(verbose=True)


data_train    = pd.read_pickle(tparam.data_train_path)
data_test     = pd.read_pickle(tparam.data_test_path)
len_train     = len(data_train)
len_test      = len(data_train)
num_batches   = int(math.ceil(len_train/tparam.batch_size))
images_tf     = tf.placeholder(tf.float32, [None, hyper.image_h, hyper.image_w, hyper.image_c], name = "images")
if hyper.sparse:
    labels_tf = tf.placeholder(tf.int64,   [None], name = 'labels')
else:
    labels_tf = tf.placeholder(tf.int64, [None, hyper.n_labels], name = 'labels')

cnn           = CNN()
if hyper.fine_tuning: 
    cnn.load_vgg_weights()

_,_,prob_tf   = cnn.build(images_tf)
if hyper.sparse:
    loss_tf   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prob_tf, labels_tf))
else:
    loss_tf   = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prob_tf, labels_tf))
loss_tf_write = tf.scalar_summary("training loss", loss_tf)
optimizer     = tf.train.AdamOptimizer(tparam.learning_rate)
train_op      = optimizer.minimize(loss_tf)


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    # for the pretty pretty tensorboard
    summary_writer = tf.train.SummaryWriter('tensorboards', sess.graph)

    for epoch in xrange(tparam.num_epochs):
        start = time()
        # Training
        epoch_loss = 0
        for b, train_batch in enumerate(chunker(data_train.sample(frac=1),tparam.batch_size)):
            train_images  = np.array(map(lambda i: load_image(i), train_batch['image_path'].values))
            if hyper.sparse:
                train_labels = train_batch['label'].values
            else:
                train_labels = np.zeros((len(train_batch), hyper.n_labels))
                for i,j in enumerate(train_batch['label'].values):
                    train_labels[i,j] = 1
            _, batch_loss, loss_sw = sess.run([train_op, loss_tf, loss_tf_write], feed_dict={images_tf: train_images, labels_tf: train_labels})

            average_batch_loss = np.average(batch_loss)
            epoch_loss += average_batch_loss
            summary_writer.add_summary(loss_sw, epoch*220+b)
            print("Train: epoch:{}, batch:{}/{}, loss:{}".format(epoch, b, num_batches, average_batch_loss))
        print("Train: epoch:{}, total loss:{}".format(epoch, epoch_loss/num_batches))

        # Validation
        correct_count = 0
        for b, test_batch in enumerate(chunker(data_test,tparam.batch_size)):
            test_images        = np.array(map(lambda i: load_image(i), test_batch['image_path'].values))
            test_labels        = test_batch['label'].values
            probs_val          = sess.run(prob_tf, feed_dict={images_tf:test_images})
            correct_count     += (probs_val.argmax(axis=1) == test_labels).sum()
        print("Test: epoch:{}, accuracy:{}".format(epoch, correct_count/len_test))
        print("Time for one epoch:{}".format(time()-start))
        # save the model
        saver.save(sess, tparam.model_path + '/model')
