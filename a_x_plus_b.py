import tensorflow as tf
import numpy as np

a = 2
b = 6


x = tf.placeholder("float")
y = tf.placeholder("float")

w = tf.Variable([1.0, 2.0], name="w")

y_model = tf.multiply(x, w[0]) + w[1]

error = tf.square(y - y_model)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error)

model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    for i in range(1000):
        x_value = np.random.rand()
        y_value = x_value * a + b
        session.run(train_op, feed_dict={x: x_value, y: y_value})

    w_value = session.run(w)
    print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
