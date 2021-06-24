
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from datasets import MNISTDataset

#Load Data and check for Datatype sanity by loading image #1 in greyscale colormap
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

plt.imshow(train_images[0], cmap="Greys_r")

data = MNISTDataset(train_images.reshape([-1, 784]), train_labels, #reshapes the 28 by 28 Pixel (=784 Entrys) input image
                    test_images.reshape([-1, 784]), test_labels,
                    batch_size=128)                                 #take 128 digits per learning batch
# Setup Training parameters
train_steps = 100 # Number of iterations of the training loop. Can increase accuracy to a certain limit
learning_rate = 0.7 # Scaling factor to set the weight adjustment of each learning iteration
N_input = 784 # Number auf Inputs(flattend pixels)
n_hidden = 390 # Number of hidden perceptrons
n_output = 10
W_i = tf.Variable(np.random.uniform(-0.0,0.1,[784, n_hidden]).astype(np.float32)) # Number and type of [Input,Output] Nodes, filled with zeros
b_i = tf.Variable(np.random.uniform(-0.1,0.1,n_hidden),dtype=np.float32)         # Biase according to number of Output nodes (10 for 10 Numbers)
# do i have to iniatiate I,h,O all with random numbers?
######Initialization for Hidden Layer Network######
# dont initialize weights to zero but to a small number
W_o = tf.Variable(np.random.uniform(-0.2,0.1,[n_hidden,n_output]).astype(np.float32)) #Connect Input layer size to hidden Layer, fill with random numbers
b_o = tf.Variable(np.random.uniform(-0.1,0.1,n_output),dtype=np.float32) #Bias according to Number of hidden Perceptrons


"""
Training

The main training loop, using cross-entropy as a loss function. We regularly print the current loss and accuracy to check progress.

Note that we compute the “logits”, which is the common name for pre-softmax values. They can be interpreted as log unnormalized probabilities and represent a “score” for each class.
"""
for step in range(train_steps):
    img_batch, lbl_batch = data.next_batch()

    """ GradientTape()
    Since different operations can occur during each call, all forward-pass operations get recorded to a "tape". 
    To compute the gradient, play the tape backwards and then discard. A particular tf.GradientTape can only compute one gradient; 
    subsequent calls throw a runtime error.
    """
    with tf.GradientTape() as tape:
        logits = tf.matmul(tf.nn.relu(tf.matmul(img_batch, W_i)+b_i),W_o)+b_o   #double matrixmultiplication A*B(row*col)+ Bias
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(   #Use Cross-Entropy as Cost Function, this defines what will be considered as learning progress
            logits=logits, labels=lbl_batch))
        
    grads = tape.gradient(xent, [W_i, W_o, b_i, b_o])                 # this is the gradient function which calculates suitable weights to minimize the Cost
    W_i.assign_sub(learning_rate * grads[0])# update weights and biases adjusted with learning rate
    W_o.assign_sub(learning_rate * grads[1])
    b_i.assign_sub(learning_rate * grads[2])
    b_o.assign_sub(learning_rate * grads[3])
    
    if not step % 100:                          #Output every 100th Step
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch),        # Watch out, sometimes different datatypes inside an array(e.g. int8,float32..) are necessary aka dtypes
                             tf.float32))
        print("Loss: {} Accuracy is: {} ".format(xent, acc))
        
"""
We can use the trained model to predict labels on the test set (10000 images) and check the model’s accuracy. You should get around 0.9 (90%) here.
"""
test_preds = tf.matmul(tf.nn.relu(tf.matmul(data.test_data, W_i)+b_i),W_o)+b_o
test_preds = tf.argmax(test_preds, axis=1, output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels),tf.float32))
plt.imshow(test_images[2], cmap="Greys_r")
print("\n===============================")
print("Test Accuracy is: {0:.3f} %".format(acc*100))

#Take away: adding a hidden layer of (input+output)/2 improves the accurecy. That is because non linear seperations are now possible
#Before HL: Training(shuffeld) acc was 89 to 94 % and test was 91.94 now with Hl it is 87 to 95 with 94.1% at test depending on random seed start With more traing it gets close to 100%
#Learning rate:Train_steps ratio is crucial for steadyness of the final result
#Learning rate close to 1 but not higher works best for small training sizes(100)
# changing the initiation weights to unsymmetric values detoriated the learning for small batches
