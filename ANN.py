#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/16 14:00
# @Author  : Ting
import numpy as np
from math import *
import matplotlib.pyplot as plt
import random
random.seed(123)


def generate_data(X, func):
    """
    generate sample data for given x
    """
    Y = []
    for x in X:
        Y.append(func(x))
    return np.array(X).T, np.array(Y).T


# define network
class Network:
    def __init__(self, x, y, weights, biases):
        # Hidden fully connected layer1 with ReLU
        self.z1 = weights['h1'].T.dot(x) + biases['b1']
        self.a1 = np.maximum(self.z1, 0)
        # Hidden fully connected layer2 with ReLU
        self.z2 = weights['h2'].T.dot(self.a1) + biases['b2']
        self.a2 = np.maximum(self.z2, 0)
        # Output fully connected output layer
        self.y_pred = weights['out'].T.dot(self.a2) + biases['out']
        # loss function
        self.loss = np.square(self.y_pred - y).sum() / y.shape[1]


def NN(train_x, train_y, test_x, test_y, learning_rate=0.01, batch_size=100, epochs=100,
       n_hidden_1=50, n_hidden_2=50):

    # Parameters
    num_input = train_x.shape[0]    # input dimension
    num_output = train_y.shape[0]   # output dimension
    num_steps = int(train_x.shape[1]/batch_size)

    # Store layers weight & bias
    weights = {
        'h1': np.random.normal(loc=0.0, scale=0.1, size=(num_input, n_hidden_1)),
        'h2': np.random.normal(loc=0.0, scale=0.1, size=(n_hidden_1, n_hidden_2)),
        'out': np.random.normal(loc=0.0, scale=0.1, size=(n_hidden_2, num_output))
    }
    # weights = {
    #     'h1': np.random.uniform(-sqrt(6/(num_input+n_hidden_1)), sqrt(6/(num_input+n_hidden_1)), size=(num_input, n_hidden_1)),
    #     'h2': np.random.normal(-sqrt(6 / (n_hidden_1+n_hidden_2)), sqrt(6 / (n_hidden_1+n_hidden_2)), size=(n_hidden_1, n_hidden_2)),
    #     'out': np.random.normal(-sqrt(6 / (n_hidden_2+num_output)), sqrt(6 / (n_hidden_2+num_output)), size=(n_hidden_2, num_output))
    # }
    biases = {
        'b1': np.zeros([n_hidden_1,1]),
        'b2': np.zeros([n_hidden_2,1]),
        'out': np.zeros([num_output,1])
    }

    loss_values = []
    # Start training
    for i in range(epochs):
        # shuffle the data - for SGD:
        # copy_data = np.concatenate((train_x,train_y),axis=0).T
        # np.random.shuffle(copy_data)
        # train_x = copy_data.T[:num_input]
        # train_y = copy_data.T[num_input:]
        for step in range(num_steps):
            batch_x = train_x[:, batch_size * step:batch_size * (step+1)]
            batch_y = train_y[:, batch_size * step:batch_size * (step+1)]

            # create network
            nn = Network(batch_x, batch_y, weights=weights, biases=biases)

            # BP process
            # calculate gradient for each layer
            error_out = 2*(nn.y_pred - batch_y)/batch_size
            grad_wout = error_out.dot(nn.a2.T)
            grad_bout = np.sum(error_out,axis=1, keepdims=True)

            nn.z2[nn.z2 < 0] = 0
            nn.z2[nn.z2 > 0] = 1
            error_2 = nn.z2 * (weights['out'].dot(error_out))
            grad_w2 = error_2.dot(nn.a1.T)
            grad_b2 = np.sum(error_2,axis=1, keepdims=True)

            nn.z1[nn.z1 < 0] = 0
            nn.z1[nn.z1 > 0] = 1
            error_1 = nn.z1 * (weights['h2'].dot(error_2))
            grad_w1 = error_1.dot(batch_x.T)
            grad_b1 = np.sum(error_1,axis=1, keepdims=True)

            # update parameters
            weights['h1'] = weights['h1'] - learning_rate * grad_w1.T
            weights['h2'] = weights['h2'] - learning_rate * grad_w2.T
            weights['out'] = weights['out'] - learning_rate * grad_wout.T

            biases['b1'] = biases['b1'] - learning_rate * grad_b1
            biases['b2'] = biases['b2'] - learning_rate * grad_b2
            biases['out'] = biases['out'] - learning_rate * grad_bout

        # save loss value for each epoch
        nn = Network(test_x, test_y, weights=weights, biases=biases)
        loss_values.append(nn.loss)

    nn = Network(test_x,test_y, weights=weights, biases=biases)
    plt.figure()
    plt.plot(loss_values)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    # plt.plot(test_x[0,:],test_y[0,:])
    # plt.plot(test_x[0, :], nn.y_pred[0, :])
    ax = plt.subplot(1, 1, 1)
    p1, = ax.plot(test_x[0, :], test_y[0,:], label="true")
    p2, = ax.plot(test_x[0, :], nn.y_pred[0, :], label="predict")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    plt.show()
    print(nn.loss)


# training data

# x and y is designed to be a two-dimension array
# so remember to correctly refer x values and return list of y values
# y = x**2
# function = lambda x: [x[0]**2]
# y = sin(x)
# function = lambda x: [sin(x[0])]

function = lambda x: [0.2+0.4*x[0]**2+0.3*x[0]*sin(15*x[0])+0.05*cos(50*x[0])]
trainx, trainy = generate_data([[i] for i in np.random.randn(10000)],function)
testx, testy = generate_data([[i/250-2] for i in range(1000)],function)

learning_rate = 0.1
batch_size = 20
epochs = 500
n_hidden_1 = 200
n_hidden_2 = 200
NN(trainx, trainy, testx, testy,learning_rate=learning_rate, batch_size=batch_size,
   epochs=epochs, n_hidden_1=n_hidden_1, n_hidden_2=n_hidden_2)
