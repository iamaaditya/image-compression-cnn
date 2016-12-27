from __future__ import division
from __future__ import print_function
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
train_b_num   = int(math.ceil(len_train/tparam.batch_size))
test_b_num    = int(math.ceil(len_train/tparam.batch_size))
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
train_loss    = tf.scalar_summary("training loss", loss_tf)
test_loss     = tf.scalar_summary("validation loss", loss_tf)

if   tparam.optimizer  == 'Adam' :
    optimizer     = tf.train.AdamOptimizer(tparam.learning_rate, epsilon=0.1)
elif tparam.optimizer  == 'Ftlr' :
    optimizer = tf.train.FtrlOptimizer(tparam.learning_rate)
elif tparam.optimizer  == 'Rmsprop' :
    optimizer     = tf.train.RMSPropOptimizer(tparam.learning_rate)
else:
    raise Exception("Unknown optimizer specified")


train_op      = optimizer.minimize(loss_tf)

def sparse_labels_or_not(batch):
    if hyper.sparse:
        return batch['label'].values
    else:
        labels = np.zeros((len(batch), hyper.n_labels))
        for i,j in enumerate(batch['label'].values):
            labels[i,j] = 1
        return labels

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    if tparam.resume_training:
        saver.restore(sess, tparam.model_path + 'model')
        if tparam.on_resume_fix_lr:
            optimizer = tf.train.FtrlOptimizer(tparam.learning_rate)
            # optimizer.learning_rate = tparam.learning_rate
            train_op      = optimizer.minimize(loss_tf)
        print("model restored...")

    # for the pretty pretty tensorboard
    summary_writer = tf.train.SummaryWriter('tensorboards', sess.graph)

    for epoch in xrange(tparam.num_epochs):

        start = time()
        # Training
        epoch_loss = 0
        for b, train_batch in enumerate(chunker(data_train.sample(frac=1),tparam.batch_size)):
            train_images  = np.array(map(lambda i: load_image(i), train_batch['image_path'].values))
            train_labels  = sparse_labels_or_not(train_batch)
            _, batch_loss, loss_sw = sess.run([train_op, loss_tf, train_loss], feed_dict={images_tf: train_images, labels_tf: train_labels})

            average_batch_loss = np.average(batch_loss)
            epoch_loss += average_batch_loss
            summary_writer.add_summary(loss_sw, epoch*train_b_num+b)
            print("Train: epoch:{}, batch:{}/{}, loss:{}".format(epoch, b, train_b_num, average_batch_loss))
        print("Train: epoch:{}, total loss:{}".format(epoch, epoch_loss/train_b_num))

        # Validation
        validation_loss = 0
        for b, test_batch in enumerate(chunker(data_test,tparam.batch_size)): # no need to randomize test batch
            test_images        = np.array(map(lambda i: load_image(i), test_batch['image_path'].values))
            # don't run the train_op by mistake ! ;-) 
            test_labels        = sparse_labels_or_not(test_batch)
            batch_loss,loss_sw = sess.run([loss_tf, test_loss], feed_dict={images_tf: test_images, labels_tf: test_labels})
            summary_writer.add_summary(loss_sw, epoch*test_b_num+b)
        print("Test: epoch:{}, total loss:{}".format(epoch, validation_loss/b))
        print("Time for one epoch:{}".format(time()-start))
        # save the model
        saver.save(sess, tparam.model_path + '/model')
