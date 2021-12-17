import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from Net.MyNet import MyNet
from utils.util import get_dataset, imbalanced_learning


def training_process(mode, f1, f2):
    features_num = 11
    batch_size = 1

    x_train, y_train, x_test, y_test = get_dataset(f1, f2)
    x_train, y_train = imbalanced_learning(x_train, y_train)

    x_train = np.array(x_train)
    y_train = np.array(y_train).reshape(-1)
    y_train = tf.one_hot(y_train, 2)
    x_test = np.array(x_test)
    y_test = np.array(y_test).reshape(-1)
    y_test = tf.one_hot(y_test, 2)
    model = MyNet(features_num, 2)

    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss',
                                            factor=0.5,
                                            patience=3,
                                            verbose=2
                                            )

    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = callbacks.EarlyStopping(monitor='accuracy',
                                             min_delta=0,
                                             patience=5,
                                             verbose=2)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=1e-3),
                  metrics=['accuracy'])

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(x_train), len(x_test), batch_size))
    epoch = 1000

    history = model.fit(x=x_train,
                        y=y_train,
                        steps_per_epoch=max(1, len(x_train)),
                        epochs=epoch,
                        initial_epoch=0,
                        validation_data=(x_test, y_test),
                        callbacks=[reduce_lr, early_stopping])
    '''
    Notice:
    Change the name of .h5 file to create your own weight file
    and then run the code
    '''
    # if mode == 'White':
    #     model.save("../Model/your_weight_w.h5")
    # else:
    #     model.save("../Model/your_weight_r.h5")
    print(history.history)
    history_loss = history.history['loss']
    val_loss = history.history['val_loss']

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    EPOCH = [i for i in range(len(history_loss))]
    plt.figure(1)
    plt.title(mode + "Wine")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(EPOCH, history_loss, 'b', label="loss")
    plt.plot(EPOCH, val_loss, 'r', label="val_loss")
    plt.legend()

    plt.figure(2)
    plt.title(mode + "Wine")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(EPOCH, acc, 'b', label="accuracy")
    plt.plot(EPOCH, val_acc, 'r', label="val_acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    w_trainFile = '../data/white_train_binary.csv'
    w_testFile = '../data/white_test_binary.csv'
    r_trainFile = '../data/red_train_binary.csv'
    r_testFile = '../data/red_test_binary.csv'
    wine = 'Red'
    if wine == 'Red':
        training_process(wine, r_trainFile, r_testFile)
    else:
        training_process(wine, w_trainFile, w_testFile)





