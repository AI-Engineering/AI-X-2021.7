import numpy
import numpy as np
import tensorflow as tf         # current ver. 2.3.0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import h5py


# 데이터 셋을 로딩한다.
# 자주 쓸 것이라서 오프라인에 다운로드 해두었다.
def load_dataset(online=False):
    if online:
        (tr_data, tr_label), (te_data, te_label) = tf.keras.datasets.mnist.load_data()

    else:
        path = "D:/Project/Mnist/dataset/mnist.npz"
        (tr_data, tr_label), (te_data, te_label) = tf.keras.datasets.mnist.load_data(path)

    print("학습 데이터 {0}개 로드".format(tr_data.shape[0]))
    print("테스트 데이터 {0}개 로드".format(te_data.shape[0]))

    return (tr_data, tr_label), (te_data, te_label)


# 이미지로 보고 싶을 때
def show_image(dataset, index):
    #plt.imshow(dataset[index])
    # 이쁘게 보이게
    plt.imshow(255-dataset[index], cmap="gray")
    plt.show()


# 데이터 분포. 히스토그램으로
def show_data_values(label):
    count_value = np.bincount(label)
    print(count_value)
    plt.bar(np.arange(0, 10), count_value)
    plt.xticks(np.arange(0, 10))
    plt.grid()
    plt.show()


# 모델 만들기
def make_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),
                     activation="relu",
                     input_shape=(28, 28, 1),
                     padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, input_dim=784, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.summary()

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    return model


# 하이퍼 파라메터
MY_EPOCH = 10
MY_BATCHSIZE = 200


def train(model, x, y):
    history = model.fit(x, y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE)
    filename = "./model/cnn_e({0}).h5".format(MY_EPOCH)
    model.save(filename)

    return history


def test_all(model, x, y):
    acc = model.evaluate(x, y, batch_size=MY_BATCHSIZE)
    print(acc)


def predict_one_sample(model, x):
    y = model.predict(x)
    print(y)
    result = y.argmax()
    return result


if __name__ == "__main__":
    (train_set, train_label), (test_set, test_label) = load_dataset()

    # 모델 생성
    cnn = make_model()

    # 데이터 모양 변경
    #print(train_set.shape)
    train_set = train_set.reshape(train_set.shape[0], 28, 28, 1)
    #print(train_set.shape)
    test_set = test_set.reshape(test_set.shape[0], 28, 28, 1)

    # 정답은 one-hot encoding
    train_label = tf.keras.utils.to_categorical(train_label, 10)
    test_label = tf.keras.utils.to_categorical(test_label, 10)

    # 학습
    #train(cnn, train_set, train_label)

    # 모델 로드
    filename = "./model/cnn_e({0}).h5".format(MY_EPOCH)
    cnn_model = load_model(filename)

    # 테스트 셋 전테 테스트
    test_all(cnn_model, test_set, test_label)

    # 하나의 샘플 테스트
    print(predict_one_sample(cnn_model, test_set[0:1]))

