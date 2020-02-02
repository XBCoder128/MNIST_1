import tensorflow as tf
import numpy as np
import input_data
import matplotlib.pyplot as plt

mnist_data = input_data.read_data_sets('mnist_data', one_hot=False)

train_images = mnist_data.train.images
train_labels = mnist_data.train.labels

test_images = mnist_data.test.images
test_labels = mnist_data.test.labels

input_ = tf.keras.Input(shape=(784, ))
dense = tf.keras.layers.Dense(128, activation='tanh')(input_)
out = tf.keras.layers.Dense(10, activation='softmax')(dense)

model = tf.keras.Model(inputs=input_, outputs=out)
model.summary()
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

model.fit(x=train_images, y=train_labels, epochs=5)

for i in range(10):
    pred = model(tf.expand_dims(test_images[i], axis=0))
    img = np.reshape(test_images[i], (28, 28))
    lab = test_labels[i]
    print('真实标签: ', lab, '， 网络预测: ', np.argmax(pred.numpy()))
    plt.imshow(img)
    plt.show()
