#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:24:34 2017

@author: Sivaramakrishnan Sankarapandian
"""

import tensorflow as tf
import numpy as np
import os
#import pickle
#import progressbar
#from sklearn.datasets import load_boston
#import random
#import csv

#dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/Users/siva/Documents/thesis/datasets/MNIST",one_hot = True)

#for graphs
NUMBER_OF_NEURONS = []
TEST_ACCURACY = []
TRAIN_ACCURACY = []
FILE_PATH = "/Users/siva/Documents/thesis/programs/numpy_files/"
FILE_NAME = "mnist_dynamic"

"""
#Hyperparameters for unmasked testing
LEARNING_RATE = 0.01
EPOCHS = 200
BATCH_SIZE = 50
ERROR_THRESHOLD = 0.01
"""

def model(INPUT_DIM,MAX_HIDDEN_DIM,OUTPUT_DIM,RATE_OF_NEURONS,LEARNING_RATE = 0.01):
    #LEARNING_RATE = 0.08
    #inputs
    x = tf.placeholder(tf.float32,[None,INPUT_DIM])
    y = tf.placeholder(tf.float32,[None,OUTPUT_DIM])
    
    #dynamiccaly varying parameters
    no_neurons = tf.placeholder(tf.int32)
    backpropogate_with_mask = tf.placeholder(tf.int32)
    
    #general weight matrices
    w1_original = tf.Variable(tf.random_normal([INPUT_DIM,MAX_HIDDEN_DIM]))
    w2_original = tf.Variable(tf.random_normal([MAX_HIDDEN_DIM,OUTPUT_DIM]))
    
    #general biases
    b1_original = tf.Variable(tf.random_normal([MAX_HIDDEN_DIM]))
    b2 = tf.Variable(tf.random_normal([OUTPUT_DIM]))
    
    #dynamically varying weights and biases
#    w1 = tf.slice(w1_original,[0,0],[tf.shape(w1_original)[0],no_neurons])
#    b1 = tf.slice(b1_original,[0],[no_neurons])
#    w2 = tf.slice(w2_original,[0,0],[no_neurons,tf.shape(w2_original)[1]])
    w1 = w1_original[0:tf.shape(w1_original)[0],0:no_neurons]
    b1 = b1_original[0:no_neurons]
    w2 = w2_original[0:no_neurons,0:tf.shape(w2_original)[1]]
    
    #masking the gradients
    w1_mask = tf.concat([tf.zeros([tf.shape(w1)[0],no_neurons-RATE_OF_NEURONS]),tf.ones([tf.shape(w1)[0],RATE_OF_NEURONS])],1)
    w1_mask_not = tf.concat([tf.ones([tf.shape(w1)[0],no_neurons-RATE_OF_NEURONS]),tf.zeros([tf.shape(w1)[0],RATE_OF_NEURONS])],1)
    
    w2_mask = tf.concat([tf.zeros([no_neurons-RATE_OF_NEURONS,tf.shape(w2)[1]]),tf.ones([RATE_OF_NEURONS,tf.shape(w2)[1]])],0)
    w2_mask_not = tf.concat([tf.ones([no_neurons-RATE_OF_NEURONS,tf.shape(w2)[1]]),tf.zeros([RATE_OF_NEURONS,tf.shape(w2)[1]])],0)
    
    b1_mask = tf.concat([tf.zeros([no_neurons-1]),tf.ones([RATE_OF_NEURONS])],0)
    b1_mask_not = tf.concat([tf.ones([no_neurons-1]),tf.zeros([RATE_OF_NEURONS])],0)
    
    w1_masked = tf.cond(tf.equal(backpropogate_with_mask,1),lambda:tf.stop_gradient(w1*w1_mask) + (w1*w1_mask_not),lambda:w1)
    b1_masked = tf.cond(tf.equal(backpropogate_with_mask,1),lambda:tf.stop_gradient(b1*b1_mask) + (b1*b1_mask_not),lambda:b1)
    w2_masked = tf.cond(tf.equal(backpropogate_with_mask,1),lambda:tf.stop_gradient(w2*w2_mask) + (w2*w2_mask_not),lambda:w2)
    #stopping gradients
#    if backpropogate_with_mask == 1:
#        w1_masked = tf.stop_gradients(w1*w1_mask) + w1*w1_mask_not
#        b1_masked = tf.stop_gradients(b1*b1_mask) + b1*b1_mask_not
#        w2_masked = tf.stop_gradients(w2*w2_mask) + w2*w2_mask_not
#    else:
#        w1_masked = w1
#        b1_masked = b1
#        w2_masked = w2
        
    temp = tf.nn.relu(tf.matmul(x,w1_masked) + b1_masked)
    out = tf.matmul(temp,w2_masked) + b2
    
    prediction = tf.argmax(out,axis = 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction,tf.argmax(y,axis = 1)),tf.float32))
    
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = out)
    #loss = tf.reduce_mean(tf.square(out - y))
    #optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    #loss = tf.sqrt(tf.reduce_mean(tf.square(y - out)))
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    
    return {"inputs":[x,y],"hyper":[no_neurons,backpropogate_with_mask],"weights":[w1,w2,b1,b2],\
            "loss":loss,"optimizer":optimizer,"out":out,"activated_output":temp,"accuracy":accuracy}
            
            
def train_mnist_dynamic(batch_size = 50,EPOCHS = 200,ERROR_THRESHOLD = 0.01,DEFAULT_HIDDEN_UNITS = 5,RATE_OF_NEURONS = 1):
    with tf.Session() as sess:
        model_mnist = model(784,1024,10,RATE_OF_NEURONS)
        sess.run(tf.global_variables_initializer())
        avg_prev_loss = 99999
        mask = 0
        hidden_units = DEFAULT_HIDDEN_UNITS
        for e in range(EPOCHS):
            accuracy_list = []
            test_accuracy_list = []
            current_loss_list = []
            for i in range(int(55000/batch_size)):
                data = mnist.train.next_batch(batch_size)
                to_compute = [model_mnist["optimizer"],model_mnist["loss"],model_mnist["out"],model_mnist["weights"][0],model_mnist["accuracy"]]
                feed_dict_train = {model_mnist["inputs"][0]:data[0],model_mnist["inputs"][1]:data[1],model_mnist["hyper"][0]:hidden_units,model_mnist["hyper"][1]:mask}
                _,current_loss,out_,w1_,acc_ = sess.run(to_compute,feed_dict = feed_dict_train)
                current_loss_list.append(current_loss)
                accuracy_list.append(acc_)
                if i%1000 == 0 and i != 0:
                    print("w1:shape",w1_.shape)
            NUMBER_OF_NEURONS.append(w1_.shape[1])
            for j in range(int(10000/batch_size)):
                data_test = mnist.test.next_batch(batch_size)
                to_compute_test = model_mnist["accuracy"]
                feed_dict_test = {model_mnist["inputs"][0]:data_test[0],model_mnist["inputs"][1]:data_test[1],model_mnist["hyper"][0]:hidden_units,model_mnist["hyper"][1]:mask}
                acc_test = sess.run(to_compute_test,feed_dict = feed_dict_test)
                test_accuracy_list.append(acc_test)
            avg_current_loss = np.mean(current_loss_list)
            avg_accuracy = np.mean(accuracy_list)
            avg_accuracy_test = np.mean(test_accuracy_list)
            TEST_ACCURACY.append(avg_accuracy_test)
            TRAIN_ACCURACY.append(avg_accuracy)
            print("EPOCHS:",e,"|","loss:",avg_current_loss,"|","train accuracy:",avg_accuracy,"|","test accuracy",avg_accuracy_test)
            if (avg_prev_loss - avg_current_loss >= ERROR_THRESHOLD):
                hidden_units += RATE_OF_NEURONS
                mask = 1
            elif (avg_prev_loss - avg_current_loss <= -(60*ERROR_THRESHOLD)):
                hidden_units -= RATE_OF_NEURONS
                mask = 0
            else:
                mask = 0
            avg_prev_loss = avg_current_loss
            
if __name__ == "__main__":
    train_mnist_dynamic()
    os.chdir(FILE_PATH)
    np.save(FILE_NAME,{"neurons":NUMBER_OF_NEURONS,"train":TRAIN_ACCURACY,"test":TEST_ACCURACY})