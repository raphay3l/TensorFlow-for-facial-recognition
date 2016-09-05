# Author: Rafal Jankowski
# UDACITY Nanodegree Machine Learning Engineer
# Deep Learning with TensorFlow
# 15 APRIL 2016


# --------------- INFO ---------------
# In the below I implement basic TensorFlow functionality to show how quickly a
# forecast using a single-layered network with stochastic gradient descent is completed
# >80% of accuracy is achieved with just one layer and biases


from __future__ import print_function
import numpy as np
import tensorflow as tf

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA




# download ready facial data
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
num_labels = target_names.shape[0]
image_size = len(train_dataset[0])


# split into a training and testing set
train_dataset, test_dataset, train_labels, test_labels = train_test_split(X, y, test_size=0.15, random_state=42)


# explore the data a little
n = 20
pca = RandomizedPCA(n_components=n, whiten=True).fit(train_dataset)
eigenvals = pca.components_.reshape((n, h, w))

#use to plot the eigen poitns of the gallery (source: SKLEARN)
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
eigenface_titles = ["eigenface %d" % i for i in range(eigenvals.shape[0])]
plot_gallery(eigenvals, eigenface_titles, h, w)

plt.show()


#for stochastic grad descent use subsets for training
train_subset = 100
batch_size = 100

#accuracy function helper
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

graph = tf.Graph()
with graph.as_default():
        # Load the data using the minibatch series for stochastic gradient descent
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)
    
        # use truncated normal distribution to initialise the weights
    weights = tf.Variable(tf.truncated_normal([image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
                                
        # take advantage of TensorFlows softmax & cross entropy combo to define the loss function
    logits = tf.matmul(tf_train_dataset, weights) + biases
    
    # use an extra hidden layer 
    #hidden1 = tf.nn.relu(tf.matmul(image_size, weights) + biases)
    #hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    #logits = tf.matmul(hidden2, weights) + biases
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    # start off with gradient descent
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
                                            
      # report the accuracy here
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


# run the model
num_steps = 1000

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for step in range(num_steps):

        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.

        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))



