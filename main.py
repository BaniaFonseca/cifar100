import tensorflow as tf

print(tf.__version__)

x = tf.Variable(7, name='x')
y = tf.Variable(7, name='y')


@tf.function
def sum(z):
    return z*(x+y)

@tf.function
def forward():
    return sum(2)


out = forward()
print(out.numpy())