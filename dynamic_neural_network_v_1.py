#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 00:23:11 2017

@author: siva
"""
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import pandas as pd

class data_generator(object):
    def __init__(self,data_points):
        """
        @params
        datapoints = no of datapoints to be generated
        """
        self.x = np.array(pd.DataFrame(load_boston().data))[0:data_points,:]
        self.y = np.array(pd.DataFrame(load_boston().target))[0:data_points,:]
        self.x_test = np.array(pd.DataFrame(load_boston().data))[data_points:data_points+100,:]
        self.y_test = np.array(pd.DataFrame(load_boston().target))[data_points:data_points+100,:]
        #self.x = np.sin(np.linspace(0,2*np.pi,100))
        #self.y = np.random.normal(self.x,0.5)
        #self.x,self.y = make_moons(data_points,noise = 0.1)
        #self.y = np.reshape(self.y,(self.y.shape[0],1))
        #plt.scatter(self.x[:,0],self.x[:,1],s = 40,c = self.y)
        
        #plt.scatter(np.arange(len(self.x)),self.y,c = "r",edgecolor = "r")
        #plt.show()
        
    def data_set_generator(self,batch_size):
        """
        @params
        batch_size = batch size for training
        """
        i = 0
        while(i < self.x.shape[0]):
            yield self.x[i:i+batch_size,:],self.y[i:i+batch_size,:]
            i += batch_size
            
    def data_test_set_generator(self,batch_size):
        """
        @params
        batch_size = batch size for testing
        """
        i = 0
        while(i < self.x_test.shape[0]):
            yield self.x_test[i:i+batch_size,:],self.y_test[i:i+batch_size,:]
            i += batch_size
        #i = 0
        #while(i<self.x.shape[0]):
        #    #yield self.x[i:i+batch_size,:],self.y[i:i+batch_size,:]
        #    yield np.reshape(self.x[i:i+batch_size],(-1,1)),np.reshape(self.y[i:i+batch_size],(-1,1))
        #    i += batch_size
            
def plot_loss(loss_list):
    """
    @params
    loss_list = list of values to be plotted
    """
    plt.figure()
    plt.plot(np.arange(len(loss_list)),loss_list)
    plt.show()
    
class neural_network(object):
    def __init__(self, input_dim, hidden_units, output_dim, batch_size):
        """
        @params
        hidden_units = no.of hidden nodes in the hidden layer
        input_dim = no.of nodes in the input layer
        output_dim = no.of nodes in the output layer
        batch_size = no.of datapoints that are include in a single batch
        """
        self.hidden_units = hidden_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.initialize_weights()

    def initialize_weights(self, variance=1):
        """
        @params
        variance = variance of the normal distribution that is used to initialize weights
        """
        self.w1 = self.random_vector(self.input_dim, self.hidden_units, variance)
        self.w2 = self.random_vector(self.hidden_units, self.output_dim, variance)
        # self.b1 = np.zeros((self.batch_size,self.hidden_units))
        # self.b2 = np.zeros((self.batch_size,self.output_dim))
        self.b1 = self.random_vector(self.batch_size, self.hidden_units, variance)
        self.b2 = self.random_vector(self.batch_size, self.output_dim, variance)

    def increase_hidden_nodes(self, no_of_nodes_increase, variance=1):
        """
        @params
        no_of_nodes_increase = number of nodes that needs to be added to hidden layer
        """
        self.hidden_units += no_of_nodes_increase
        vector_to_append_w1 = self.random_vector(self.input_dim, 1, variance)
        vector_to_append_w2 = self.random_vector(1, self.output_dim, variance)
        self.w1 = np.append(self.w1, vector_to_append_w1, axis=1)
        self.w2 = np.append(self.w2, vector_to_append_w2, axis=0)
        vector_to_append_b1 = self.random_vector(self.batch_size, 1, variance)
        vector_to_append_b2 = self.random_vector(self.batch_size, 1, variance)
        self.b1 = np.append(self.b1, vector_to_append_b1, axis=1)
        # self.b2 = np.append(self.b2,vector_to_append_b2,axis = 1)
        # print(self.w1.shape,self.w2.shape,self.b1.shape,self.b2.shape)
        
    def reduce_hidden_nodes(self,no_of_nodes_decrease):
        """
        @params
        no_of_nodes_decrease = number of nodes which have to be decreased
        """
        self.hidden_units -= no_of_nodes_decrease
        self.w1 = self.w1[:,:-no_of_nodes_decrease]
        self.b1 = self.b1[:,:-no_of_nodes_decrease]
        self.w2 = self.w2[:-no_of_nodes_decrease,:]
        #self.b2 = self.b2[:-no_of_nodes_decrease,:]

    def feed_forward(self, input_, ground_truth, activation="sigmoid"):
        """
        @params
        input_ = input to the neural network
        ground_truth = true labels
        """
        self.input_ = input_
        self.ground_truth = ground_truth
        self.input_w1 = (input_ @ self.w1) + self.b1
        self.input_w1_activated = self.sigmoid(self.input_w1)
        self.input_w2 = (self.input_w1_activated @ self.w2) + self.b2
        #self.output = self.sigmoid(self.input_w2)
        self.output = self.input_w2
        self.loss = self.loss_fn(self.output, ground_truth)
        return self.loss

    def back_prop(self):
        self.d_error = self.loss_fn_derivative(self.output,self.ground_truth);
        #self.dw2 = (self.input_w1_activated.T @ (self.d_error * self.sigmoid_prime(self.input_w2)) )
        #self.db2 = self.d_error * self.sigmoid_prime(self.input_w2)
        self.dw2 = (self.input_w1_activated.T @ (self.d_error))
        self.db2 = self.d_error
        # print(self.input_.T.shape,"--",self.output.shape,self.d_error.shape,self.w2.T.shape,self.input_w1.shape)
        self.dw1 = self.input_.T @ ((self.d_error @ self.w2.T) * self.sigmoid_prime(self.input_w1_activated))
        #print("dw1:",self.sigmoid_prime(self.input_w1))
        self.db1 = ((self.d_error @ self.w2.T) * self.sigmoid_prime(self.input_w1_activated))
        #print("weights----",self.dw1)

    def optimize(self, learning_rate=0.01,masked = False):
        """
        @params
        learning_rate = determines how fast the network learns
        """
        if not masked:
            self.w1 -= learning_rate * self.dw1
            self.w2 -= learning_rate * self.dw2
            self.b1 -= learning_rate * self.db1
            self.b2 -= learning_rate * self.db2
        else:
            self.w1 -= learning_rate * (self.dw1 * self.create_mask(self.w1,False,True))
            self.w2 -= learning_rate * (self.dw2 * self.create_mask(self.w2,True,False))
            self.b1 -= learning_rate * (self.db1 * self.create_mask(self.b1,False,True))
            self.b2 -= learning_rate * self.db2
            #self.b2 -= learning_rate * (self.db2 * create_mask(self.b2,True,False))
            
    def create_mask(self,in_,row,col):
        """
        @params
        in_ = input weight to the function
        row,col = booleans that says whether row or col has to be masked
        """
        dummy = np.zeros(in_.shape)
        if row:
            dummy[-1,:] = np.ones(in_.shape[1])
        if col:
            dummy[:,-1] = np.ones(in_.shape[0])
        return dummy

    def loss_fn(self, output, ground_truth, loss_fn = "mean_square_error"):
        """
        @params
        output = output of the neural network
        ground_truth = labels for learning
        """
        if loss_fn == "mean_square_error":
            return (output - ground_truth) ** 2
        if loss_fn == "cross_entropy":
            return (ground_truth * np.log(output)) 
    

    def loss_fn_derivative(self,output,ground_truth,loss_fn = "mean_square_error"):
        """
        @params
        output = output of the neural network
        ground_truht = labels for learning
        """
        if loss_fn == "mean_square_error":
            return 2*(output - ground_truth)
        if loss_fn == "cross_entropy":
            return 2*(output - ground_truth)

    def sigmoid(self, inp):
        """
        finds the sigmoid activated output of the input matrix
        """
        return (1 / (1 + np.exp(-inp)))
    
    def softmax(self,inp):
        """
        finds the softmax activated output of the input matrix
        """
        return (np.exp(inp)/np.sum(np.exp(inp),axis = 0))

    def sigmoid_prime(self, inp):
        """
        finds the value of the matrix substituted(elementwise) in a sigmoid function derivative
        """
        #return (np.exp(-inp)) / ((1 + np.exp(-inp))**2)
        return inp*(1 - inp)

    def random_vector(self, first_dim, second_dim, variance):
        """
        @params
        first_dim,second_dim = dimensions of the multivariate gaussian used to sample the weights from
        variance = variance of the
        """
        #np.random.seed()
        random_matrix = np.zeros((first_dim, second_dim))
        for i in range(first_dim):
            for j in range(second_dim):
                random_matrix[i][j] = np.random.normal(0,variance)
        return random_matrix
        #mu = np.zeros((first_dim, second_dim))
        #covariance = np.ones((first_dim, second_dim)) * variance
        #return np.random.normal(mu, covariance)
    
#hyperparameters
input_dim = 13
hidden_nodes = 5
output_dim = 1
batch_size = 20
learning_rate = 0.0001
train_dataset_size = 400
test_dataset_size = 100
THRESHOLD = 0.1
ERROR_THRESHOLD = 1

#class objects for the neural network and the data generator
#nn = neural_network(input_dim,hidden_nodes,output_dim,batch_size)
train_data = data_generator(train_dataset_size)
test_data = data_generator(test_dataset_size)

#variables
loss_prev = 0
train_loss_list = []
test_loss_list = []
hidden_units_list = []
result = []
ground_truth = []

#normal training
EPOCH = 500


current_error_list =  []
test_error_list= []

nn_testing = neural_network(input_dim,hidden_nodes,output_dim,batch_size)
prev_error = 9999999
optimize_with_mask = False
#testing implementation
for epoch in range(EPOCH):
    current_error = 0
    for x_test,y_test in train_data.data_set_generator(batch_size):
        nn_testing.feed_forward(x_test,y_test)
        nn_testing.back_prop();
        if epoch == 1:
            nn_testing.optimize(learning_rate) 
        else:
            #print(optimize_with_mask)
            nn_testing.optimize(learning_rate,masked = optimize_with_mask)
        current_error_list.append(np.sum(nn_testing.loss)/batch_size)
    current_error = np.mean(current_error_list)
    train_loss_list.append(current_error)
    hidden_units_list.append(nn_testing.w1.shape[1])
    print("epoch:",epoch,"|","error:",current_error,"|","hidden units:",nn_testing.w1.shape[1],"|","error difference:",(current_error - prev_error))
    if (prev_error - current_error) >= ERROR_THRESHOLD:
        nn_testing.increase_hidden_nodes(1)
        optimize_with_mask = True
    elif (prev_error - current_error) <= -ERROR_THRESHOLD:
        nn_testing.reduce_hidden_nodes(1)
        optimize_with_mask = False
    else:
        optimize_with_mask = False
    prev_error = current_error
    
    test_error = 0
    for x_,y_ in test_data.data_test_set_generator(batch_size):
        nn_testing.feed_forward(x_,y_)
        for e in nn_testing.output:
            result.append(e)
        for f in y_:
            ground_truth.append(f)
        test_error_list.append(np.sum(nn_testing.loss)/batch_size)
    test_error = np.mean(test_error_list)
    test_loss_list.append(test_error)
    #print("test error:",test_error)
plt.plot(np.arange(EPOCH),np.array(train_loss_list))
plt.xlabel("epochs")
plt.ylabel("train error")
plt.ylim(0,700)
plt.show()
plt.plot(np.arange(EPOCH),np.array(test_loss_list))
plt.xlabel("epochs")
plt.ylabel("test error")
plt.ylim(0,700)
plt.show()
plt.plot(np.arange(EPOCH),np.array(hidden_units_list))
plt.xlabel("epochs")
plt.ylabel("no of hidden units")
#print(np.mean((np.array(result) - np.array(ground_truth))**2))
#print((np.array(result) - np.array(ground_truth)))
#print(nn_testing.w1)
#print(nn_testing.w2)

            
        

#print(nn_testing.b1.shape)
#nn_testing.increase_hidden_nodes(1)
#print(nn_testing.create_mask(nn_testing.b2,True,False))
#nn_testing.reduce_hidden_nodes(1)
#print(nn_testing.b1.shape)


#for epoch in range(EPOCH):
#    train_loss = 0
#    for x_,y_ in train_data.data_set_generator(batch_size):
#        nn.feed_forward(x_,y_)
#        nn.back_prop()
#        nn.optimize(learning_rate)
#        train_loss_list.append(np.sum(nn.loss)/batch_size)
#    print("epoch:",epoch,"loss:",(np.mean(train_loss_list)))
#for x_,y_ in train_data.data_set_generator(batch_size):
#    nn.feed_forward(x_,y_)
#    for e in nn.output:
#        result.append(e)
#    for f in y_:
#        ground_truth.append(f)
        
#ground_truth = np.array(pd.DataFrame(load_boston().target))[0:train_dataset_size,:]
#plt.scatter(result,ground_truth)

#optimization
#EPOCH = 1000
#for epo in range(EPOCH):
#    train_loss = 0
#    test_loss = 0
#    #train_loop
#    for x_,y_ in train_data.data_set_generator(batch_size):
#        nn.feed_forward(x_,y_)
#        nn.back_prop()
#        nn.optimize(learning_rate)
#        train_loss += np.sum(nn.loss)
#    train_loss_list.append(train_loss/train_dataset_size)
#    for x_test,y_test in test_data.data_set_generator(batch_size):
#        nn.feed_forward(x_test,y_test)
#        test_loss += np.sum(nn.loss)
#    test_loss_list.append(test_loss/test_dataset_size)
#    #if train_loss - loss_prev > THRESHOLD:
#    if train_loss > loss_prev:
#        nn.increase_hidden_nodes(1)
#        #print("no_of_hidden_units:",nn.hidden_units)
#    hidden_units_list[epo] = nn.hidden_units
#    #print(loss)
#    loss_prev = train_loss
#plot_loss(train_loss_list)
#plot_loss(test_loss_list)
#temp = sorted(hidden_units_list.keys())
#plt.plot(temp,[hidden_units_list[a] for a in temp])
#plt.show()
#print("test_loss:",test_loss_list)