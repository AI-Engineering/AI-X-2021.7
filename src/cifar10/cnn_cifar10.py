import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model


# data 로드. 전역으로 로드한다.
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print("학습 데이터 {0}개 로드".format(train_images.shape[0]))
print("테스트 데이터 {0}개 로드".format(test_images.shape[0]))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def show_data_sample():
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i][0]])

    plt.show()


def make_model():
    model = Sequential()
    # 1
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',
                     input_shape=(32, 32, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # 2
    model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # 3
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    # 4
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    return model


# 하이퍼 파라메터
MY_EPOCH = 1
MY_BATCHSIZE = 200
filename = f"./model/cifar_e({MY_EPOCH}).h5"


def train(model, x, y):
    x = x.astype("float32")
    x /= 255
    y = tf.keras.utils.to_categorical(y, 10)
    history = model.fit(x, y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE)
    model.save(filename)

    return history


def test_all(model, x, y):
    x = x.astype("float32")
    x /= 255
    y = tf.keras.utils.to_categorical(y, 10)
    model.evaluate(x, y)


def predict_one_sample(model, x):
    x = 255 - x
    x = x.astype("float32")
    x /= 255
    y = model.predict(x)
    result = y.argmax()
    return result


if __name__ == "__main__":
    #show_data_sample()

    cnn = make_model()
    #train(cnn, train_images, train_labels)

    model = load_model(filename)
    test_all(model, test_images, test_labels)