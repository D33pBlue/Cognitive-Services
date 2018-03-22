import numpy as np
import tensorflow as tf

# build and execute a computational graph
# create variables and initializer
x = tf.Variable(4,name='x')
y = tf.Variable(2,name='y')
init = tf.global_variables_initializer()
# create the function
f = x*x*y + y + 2
result = 0
# create session
with tf.Session() as sess:
    init.run()
    writer = tf.summary.FileWriter('./graphs',sess.graph)
    result = f.eval()
print(result)
# To view the generated graph type in a shell
# tensorboard --logdir="graphs"
