import numpy
import numpy as np
import tensorflow as tf         # current ver. 2.3.0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
    model.add(Dense(128, input_dim=784, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.summary()
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    return model


# 하이퍼 파라메터
MY_EPOCH = 29
MY_BATCHSIZE = 100

def train(model, x, y):
    history = model.fit(x, y, epochs=MY_EPOCH, batch_size=MY_BATCHSIZE)
    filename = "./model/mlp_hd512_e({0}).h5".format(MY_EPOCH)
    model.save(filename)

    return history

def test(x, y):
    filename = "./model/mlp_hd512_e({0}).h5".format(MY_EPOCH)
    test_model = load_model(filename)
    acc = test_model.evaluate(x, y, batch_size=MY_BATCHSIZE)

    print(acc)

if __name__ == "__main__":
    (train_set, train_label), (test_set, test_label) = load_dataset()

    # 데이터 확인
    show_image(train_set, 0)
    show_data_values(train_label)
    show_data_values(test_label)

    # 데이터 변환
    train_set = train_set.reshape(60000, 784)
    train_set = train_set.astype("float32")
    train_set /= 255

    # 짧게 쓰면 이렇게...
    test_set = test_set.reshape(10000, 784).astype("float32") / 255

    # 정답은 one-hot encoding
    train_label = tf.keras.utils.to_categorical(train_label, 10)
    test_label = tf.keras.utils.to_categorical(test_label, 10)

    # 모델 생성
    mlp = make_model()

    # 학습
    train(mlp, train_set, train_label)

    # 테스트
    test(test_set, test_label)
