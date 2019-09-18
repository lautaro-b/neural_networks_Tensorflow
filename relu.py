
# coding: utf-8

# ## Solving MNIST with TensorFlow

# In[1]:

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import time


# Load the MNIST data:

# In[2]:

data_dir = 'MNIST_data/'
#mnist = input_data.read_data_sets(data_dir,
#                                  one_hot=True,
#                                  fake_data=False)
mnist = tf.contrib.learn.datasets.load_dataset("mnist") 


# Let's see what we have here:

# In[3]:

print(mnist.train.images.shape)


# In[4]:

def matrix_show(A):
    for y in A:
        print(''.join(['%4d'%x for x in y]))
matrix_show(100*mnist.train.images[1].reshape((28,28)))


# In[5]:

plt.imshow(mnist.train.images[1].reshape((28,28)), cmap='gray')
plt.show()
print(mnist.train.labels[1])


# In[6]:

mnist.validation.images.shape


# In[7]:

mnist.test.images.shape


# In[8]:

def show_images(imgs, labels, highlight=None):
    if highlight is None:
        highlight = np.zeros(len(labels)) > 0
    colors = np.array([(0,0,1), (1,0,0)])
    colors = colors[highlight.astype(int)]

    N=imgs.shape[0]
    w=int(np.ceil(np.sqrt(N)))
    h=int(N/w)
    imgs.shape=(N, 28,28)

    rows = [ np.hstack(imgs[i:i+w,:,:]) for i in range(0,w*h, w)]
    tiled = np.vstack(rows)
    plt.imshow(1.0 - tiled, cmap='gray')

    for i in range(w):
        for j in range(h):
            plt.text(i*28, 14+j*28, '%s'%labels[j*w+i], color=colors[j*w+i])
    plt.show()
#%pylab inline
#pylab.rcParams['figure.figsize'] = (10,10)  
idx = np.random.randint(0, 55000, 25)
show_images(mnist.train.images[idx], mnist.train.labels[idx])


# # Define a graph to classify images

# In[9]:

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')  # x is BatchSize x 784
    y = tf.placeholder(tf.int32, [None], name='y')  # y is BatchSize x 1


# Define some useful functions:

# In[10]:

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(tf.cast(var, tf.float32))
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)


# In[11]:

from pprint import pprint
def affine_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu):
    """
    Build one simple Affine layer: y = W*x + b
    Adds non-linearity and collects statistics
    """
    input_shape = input_tensor.get_shape().as_list()
    input_dim = input_shape[-1]

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            shape = [input_dim, output_dim]
            init = tf.truncated_normal(shape, stddev=0.1)
            weights = tf.Variable(init)
            #variable_summaries(weights)
        with tf.name_scope('biases'):
            initial = tf.constant(0.1, shape=[output_dim])
            biases = tf.Variable(initial)
            #variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            #tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    #tf.summary.histogram('activations', activations)
    print("Added Affine layer %d -> %d with %s" % (input_dim, output_dim, act.__name__))
    return activations


# ### Define our simple network

# In[12]:

hidden = affine_layer(x, 500, 'hidden')
prediction = affine_layer(hidden, 10, 'pred', act=tf.identity)


# In[13]:

sess = tf.InteractiveSession()


# We built an "inference" function that receives a batch of images and returns a tensorflow object of the prediction. 
# 
# The next component is creating a loss function.

# In[14]:

def create_loss(predict, labels):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    
    #labels_1hot = tf.one_hot(labels, depth=10)
    #diff = tf.reduce_mean(tf.square(labels_1hot - predict), reduction_indices=[1])
    
    diff = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, 
                                                   logits=predict)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy
  


# In[15]:

loss = create_loss(prediction, y)


# In[16]:

BATCH_SIZE = 100
eta = 0.1


# In[17]:

with tf.name_scope('train'):
    #train_step = tf.train.AdamOptimizer(eta).minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(eta).minimize(loss)


# In[18]:

correct_prediction = tf.equal(y, tf.cast(tf.argmax(prediction, axis=1), tf.int32))
accuracy = 100.0 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()


# Merge all the summaries and start training:

# In[19]:

log_dir = '/tmp/tensorboard-mnist/'
if tf.gfile.Exists(log_dir):  # Delete previous runs
    import shutil
    shutil.rmtree(log_dir)
tf.gfile.MakeDirs(log_dir)

train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

tf.global_variables_initializer().run()


# In[20]:

train = []
test = []


acc_list = []
for i in range(1000):
    xs, ys = mnist.train.next_batch(32)
    ys = ys.astype(np.int32)
    summary, acc, _ = sess.run([merged, accuracy, train_step], 
                               feed_dict={x:xs, y:ys})
    acc_list.append(acc)
    train_writer.add_summary(summary, i)
    if i % 20 == 0:   # Measure accuracy on the test set
        summary, test_acc = sess.run([merged, accuracy], 
                                     feed_dict={x:mnist.test.images, 
                                                y:mnist.test.labels})
        test_writer.add_summary(summary, i)
        print('Accuracy at iteration %4d: %6s%%  (train is %6s%%)' % (i, test_acc, np.mean(acc_list)))
        
        train.append(np.mean(acc_list))
        test.append(test_acc)
        
        acc_list=[]
        
plt.title("Activation function = reLU")        
plt.plot(train, label="train error")
plt.plot(test, label="test error")
plt.legend()
plt.show()

train_writer.flush()
test_writer.flush()    


# ### (tensor Board)

# # Let's look at some predictions

# In[21]:

xs, ys = mnist.test.next_batch(BATCH_SIZE)
ys = ys.astype(np.float32)


# In[22]:

pred = sess.run(prediction, feed_dict={x:xs})
pred = np.argmax(pred, axis=1)
errors = pred != ys
print("Found %d/%d errors"%(np.sum(errors), len(errors)))


# In[23]:

pred = sess.run(prediction, feed_dict={x:mnist.test.images})
pred = np.argmax(pred, axis=1)


# In[24]:

show_images(xs, pred)


# In[25]:

show_images(xs, pred, errors)


# In[26]:

errors = pred != mnist.test.labels
print('Error rate is %.2f%%'%(100.0*sum(errors) / len(errors)))

pick = np.random.permutation(np.where(errors)[0])
pick = pick[:100]
show_images(mnist.test.images[pick], pred[pick], errors[pick])


# So we have a simple, nearly linear, classifier

# # LeNet

# Lenet has convolutions and max-pooling layers
# 
# Let's reset TensorFlow:

# In[27]:

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.int32, [None], name='y-input')


# Define a convolutional layer:

# In[28]:

def conv_layer(input_tensor, output_channels, layer_name, 
               ksize=5, stride=1, act=tf.nn.relu):
    input_shape = input_tensor.get_shape().as_list()
    input_channels = input_shape[-1]
    with tf.name_scope(layer_name):
        weights = tf.Variable(
            tf.truncated_normal([ksize, ksize, input_channels, 
                                 output_channels],
                                stddev=1.0 / np.sqrt(float(input_channels))),name='weights')
        biases = tf.Variable(tf.zeros([output_channels]),name='biases')
        c = tf.nn.conv2d(input_tensor, weights, 
                         strides=[1, stride, stride, 1], padding='SAME') + biases
        print("Added Conv layer %dx%dx %d --%dx%d--> %d with %s" % (input_shape[1], input_shape[2], input_shape[3], 
                                                                    ksize, ksize,
                                                                    output_channels, act.__name__))
        return act(c)

def flatten_layer(x, layer_name):
    x_shape = x.get_shape().as_list()
    new_shape = [-1, np.product(x_shape[1:])]
    print("Added Flatten layer %dx%d%d -> %d"% (x_shape[1], x_shape[2], x_shape[3], new_shape[1]))
    with tf.name_scope(layer_name):
        out = tf.reshape(x, new_shape)
    return out


# In[29]:

def inference(x):
    images = tf.reshape(x, [-1, 28, 28, 1])
    #tf.summary.image('input', images, 10)
    conv1 = conv_layer(images, 20, 'conv1', ksize=3) #, act=tf.identity)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') 
    conv2 = conv_layer(pool1, 50, 'conv2', ksize=3) # , act=tf.identity)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
    flat_pool = flatten_layer(pool2, 'flatten')
    aff = affine_layer(flat_pool, 500, 'affine')
    predict = affine_layer(aff, 10, 'predict', act=tf.identity)
    return predict


# In[30]:

prediction = inference(x)


# Redefine the loss etc:

# In[ ]:

loss = create_loss(prediction, y)
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(eta).minimize(loss)
correct_prediction = tf.equal(y, tf.cast(tf.argmax(prediction, axis=1), tf.int32))
accuracy = 100.0 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_writer = tf.summary.FileWriter(log_dir + '/trainLenet', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/testLenet')

tf.global_variables_initializer().run()
iter = 0
acc_list = []


# In[ ]:

for i in range(300):
    xs, ys = mnist.train.next_batch(BATCH_SIZE)
    ys = ys.astype(np.float32)
    acc, _ = sess.run([accuracy, train_step], feed_dict={x:xs, y:ys})
    acc_list.append(acc)
    summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value= np.asscalar(acc))])
    train_writer.add_summary(summary, i)
    if i % 50 == 0:   # Measure accuracy on the test set
        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        summary = tf.Summary(value=[tf.Summary.Value(tag='accuracy', simple_value= np.asscalar(test_acc))])
        test_writer.add_summary(summary, iter)
        print('Accuracy at iteration %3d: %6s  (train is %6s)' % (iter, test_acc, np.mean(acc_list)))
        iter = iter + 50
        acc_list = []
train_writer.flush()
test_writer.flush()


# In[ ]:

xs = mnist.test.images
pred = sess.run(prediction, feed_dict={x:xs})
pred = np.argmax(pred, axis=1)

errors = pred != mnist.test.labels
print(xs.shape)
print('Error rate is %.2f%%'%(100.0*np.sum(errors) / len(mnist.test.labels)))

pick = np.random.permutation(np.where(errors)[0])
pick = pick[:100]
show_images(mnist.test.images[pick], pred[pick], errors[pick])


# In[ ]:




# In[ ]:



