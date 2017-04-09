from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

import tensorflow as tf
import tensorflowvisu
tf.set_random_seed(0)

# Input image,  28 * 28 grey image
x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x")

# Input label
y_ = tf.placeholder(tf.float32, shape=[None, 10], name="label")

x_image = tf.reshape(x, [-1,28,28,1])
tf.summary.image('input', x_image, 7)

# Five layers and their number of neurons
L = 200
M = 100
N = 60
O = 30

XX = tf.reshape(x, [-1, 784])

with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1), name="W1") # 28 * 28 image reshate to 784 vector
    B1 = tf.Variable(tf.zeros([L]), name="B1")
    Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)
    tf.summary.histogram('weights', W1)
    tf.summary.histogram('biases', B1)
    tf.summary.histogram('act', Y1)

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1), name="W2")
    B2 = tf.Variable(tf.zeros([M]), name="B2")
    Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)
    tf.summary.histogram('weights', W2)
    tf.summary.histogram('biases', B2)
    tf.summary.histogram('act', Y2)

with tf.name_scope("layer3"):
    W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1), name="W3")
    B3 = tf.Variable(tf.zeros([N]), name="B3")
    Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)
    tf.summary.histogram('weights', W3)
    tf.summary.histogram('biases', B3)
    tf.summary.histogram('act', Y3)

with tf.name_scope("layer4"):
    W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1), name="W4")
    B4 = tf.Variable(tf.zeros([O]), name="B4")
    Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)
    tf.summary.histogram('weights', W4)
    tf.summary.histogram('biases', B4)
    tf.summary.histogram('act', Y4)

with tf.name_scope("softmax"):
    W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=1.0), name="W")
    B5 = tf.Variable(tf.zeros([10]), name="B")
    tf.summary.histogram('weights', W5)
    tf.summary.histogram('biases', B5)
    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits)

# Loss func
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y_))
    cross_entropy = cross_entropy * 100
    tf.summary.scalar('cross_entropy',cross_entropy)

# Training
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.03).minimize(cross_entropy)

allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])],0)
allbiases = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])],0)

I = tensorflowvisu.tf_format_mnist_images(x, Y, y_)
It = tensorflowvisu.tf_format_mnist_images(x, Y, y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis()

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

#writer = tf.summary.FileWriter("summary/30")
#writer.add_graph(sess.graph)

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged_summary = tf.summary.merge_all()

def training_step(i, update_test_data, update_train_data):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_data = {x:batch_xs, y_:batch_ys}

    test_data = {x:mnist.test.images, y_:mnist.test.labels}

    if update_train_data :
        [trained_accuracy, trained_loss, im, w, b] = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict=train_data)
        print(str(i) + ": Trained accuracy: " + str(trained_accuracy) + " trained loss: " + str(trained_loss))
        datavis.append_training_curves_data(i, trained_accuracy, trained_loss)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im);

    if update_test_data:
        [test_accuracy, test_loss, im] = sess.run([accuracy, cross_entropy, It], feed_dict=test_data)
        print(str(i) + ": Test accuracy: " + str(test_accuracy))
        datavis.append_test_curves_data(i, test_accuracy, test_loss)
        datavis.update_image2(im)

    # Backpropagation
    sess.run(train_step, feed_dict=train_data)

datavis.animate(training_step, iterations=10000+1, train_data_update_freq=20, test_data_update_freq=100, more_tests_at_start=True)

print("Max test accuracy: " + str(datavis.get_max_test_accuracy()))