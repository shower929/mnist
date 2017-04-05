from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# Model
x = tf.place
x = tf.placeholder(tf.float32, [None, 784], name="x")
x_image = tf.reshape(x, [-1,28,28,1])
tf.summary.image('input', x_image, 3)
W = tf.Variable(tf.zeros([784,10]), name="W")
b = tf.Variable(tf.zeros([10]), name="b")

y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32, shape=[None, 10], name="label")

tf.summary.histogram('weights', W)
tf.summary.histogram('biases', b)

# Loss func
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('cross_entropy',cross_entropy)

# Training
with tf.name_scope("train"):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

writer = tf.summary.FileWriter("summary/10")
writer.add_graph(sess.graph)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

merged_summary = tf.summary.merge_all()

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if(i % 5 == 0):
        [test_accury, s] = sess.run([accuracy, merged_summary], feed_dict={x:batch_xs, y_:batch_ys})
        writer.add_summary(s,i)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})



