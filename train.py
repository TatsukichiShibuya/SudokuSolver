import numpy as np
import joblib
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

def trainModel(epc=5):
    x = np.array(joblib.load("traindata/x.joblib"))
    t = np.array(joblib.load("traindata/t.joblib"))
    l = len(x)

    p = np.random.permutation(l)
    x = x[p]
    t = t[p]

    # TrainData 90*10*(クラスごとの枚数)*0.9, TestData　90*10*(クラスごとの枚数)*0.1
    D = int(l*0.9)
    x_train = x[:D]
    y_train = t[:D]
    x_test = x[D:]
    y_test = t[D:]

    x_train1, x_valid, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.175)
    x_train = x_train1
    y_train = y_train1

    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_valid = x_valid.reshape(x_valid.shape[0], 32, 32, 1)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)

    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_valid /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_valid = keras.utils.to_categorical(y_valid, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=epc,
                        verbose=1,
                        validation_data=(x_valid, y_valid))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    joblib.dump(model, "model/model.joblib")
