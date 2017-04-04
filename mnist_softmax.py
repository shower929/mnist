from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784], name="x")

W = tf.Variable(tf.zeros([784,10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32, shape=[None, 10], name="label")

#Loss func
with tf.name_scope("xent"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#Training
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

with tf.name_scope("accury"):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    accury = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accury, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

writer = tf.summary.FileWriter("graph/4")
writer.add_graph(sess.graph)