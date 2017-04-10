from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

import tensorflow as tf
import tensorflowvisu
import math
import sys

tf.set_random_seed(0)

gui = sys.argv[1] == "gui"
folder = sys.argv[2]

# Input image,  28 * 28 grey image
x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x")

# Input label
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="label")

rate = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1,28,28,1])
tf.summary.image('input', x_image, 7)

# Five layers and their number of neurons
L = 200
M = 100
N = 60
O = 30

XX = tf.reshape(x, [-1, 784])

with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1), name="W1") # 28 * 28 image reshape to 784 vector
    B1 = tf.Variable(tf.ones([L])/10, name="B1")
    Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
    tf.summary.histogram('W1', W1)
    tf.summary.histogram('B1', B1)

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1), name="W2")
    B2 = tf.Variable(tf.ones([M])/10, name="B2")
    Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
    tf.summary.histogram('W2', W2)
    tf.summary.histogram('B2', B2)

with tf.name_scope("layer3"):
    W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1), name="W3")
    B3 = tf.Variable(tf.ones([N])/10, name="B3")
    Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
    tf.summary.histogram('W3', W3)
    tf.summary.histogram('B3', B3)

with tf.name_scope("layer4"):
    W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1), name="W4")
    B4 = tf.Variable(tf.ones([O])/10, name="B4")
    Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
    tf.summary.histogram('W4', W4)
    tf.summary.histogram('B4', B4)

with tf.name_scope("softmax"):
    W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=1.0), name="W")
    B5 = tf.Variable(tf.zeros([10]), name="B")
    tf.summary.histogram('W5', W5)
    tf.summary.histogram('B5', B5)
    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits)

# Loss func
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y_)
    cross_entropy = tf.reduce_mean(cross_entropy) * 100
    tf.summary.scalar('cross_entropy',cross_entropy)

# Training
with tf.name_scope("train"):
    train = tf.train.GradientDescentOptimizer(rate).minimize(cross_entropy)
    allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])],0)
    allbiases = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])],0)

I = tensorflowvisu.tf_format_mnist_images(x, Y, y_)
It = tensorflowvisu.tf_format_mnist_images(x, Y, y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

writer = tf.summary.FileWriter("summary/relu/" + folder)
writer.add_graph(sess.graph)

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()


def training_step(i, update_test_data, update_train_data):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_data = {x:batch_xs, y_:batch_ys}
    test_data = {x:mnist.test.images, y_:mnist.test.labels}
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    r = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

    if update_train_data:
        [a, loss, im, w, b] = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict=train_data)
        print(str(i) + ": Trained accuracy: " + str(a) + " trained loss: " + str(loss))
        datavis.append_training_curves_data(i, a, loss)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)

    if update_test_data:
        [a, loss, im] = sess.run([accuracy, cross_entropy, It], feed_dict=test_data)
        print(str(i) + ": Test accuracy: " + str(a) + " test loss: " + str(loss))
        datavis.append_test_curves_data(i, a, loss)
        datavis.update_image2(im)

    s = sess.run(merged_summary, feed_dict=test_data)
    writer.add_summary(s, i)

    # Backpropagation
    sess.run(train, {x:batch_xs, y_:batch_ys, rate: r})

if gui:
    datavis.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)
else:
    for i in range(30000):
        training_step(i, i % 100 == 0, i % 20 == 0)

print("Max test accuracy: " + str(datavis.get_max_test_accuracy()))