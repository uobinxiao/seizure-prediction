import numpy
numpy.random.seed(1337) # for reproducibility
import tensorflow
tensorflow.random.set_random_seed(1337)
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, merge, Input, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.regularizers import l2, l1
from keras.optimizers import SGD, Adadelta

class MultiViewModel:

    @classmethod
    def getModel(cls, setting, channels, fft_bins, pca_bins, steps):
        input1 = Input(shape = (channels * pca_bins, steps, 1), name = "input1")
        seq1 = Conv2D(filters = setting.nb_filter, kernel_size = (channels * pca_bins, 1),
                      kernel_initializer = "lecun_uniform",
                      kernel_regularizer=l2(l = setting.l2),
                      activation="relu")(input1)
        seq1 = Dropout(setting.dropout)(seq1)
        seq1 = Conv2D(filters = setting.nb_filter, kernel_size = (1, 3),
                      kernel_regularizer=l2(l = setting.l2),
                      kernel_initializer = "lecun_uniform",
                      activation="relu")(seq1)
        seq1 = Dropout(setting.dropout)(seq1)
        seq1 = Conv2D(filters = setting.nb_filter, kernel_size = (1, 3),
                      kernel_regularizer=l2(l = setting.l2),
                      kernel_initializer = "lecun_uniform",
                      activation="relu")(seq1)
        seq1 = Dropout(setting.dropout)(seq1)
        seq1 = Conv2D(filters = setting.nb_filter, kernel_size = (1, 3),
                      kernel_regularizer=l2(l = setting.l2),
                      kernel_initializer = "lecun_uniform",
                      activation="relu")(seq1)
        seq1 = Dropout(setting.dropout)(seq1)
        seq1 = Conv2D(filters = setting.nb_filter, kernel_size = (1, 3),
                      kernel_regularizer=l2(l = setting.l2),
                      kernel_initializer = "lecun_uniform",
                      activation="relu")(seq1)
        seq1 = Flatten()(seq1)
        output1 = Dense(setting.output1, activation="tanh")(seq1)

        input2 = Input(shape = (channels * fft_bins, steps, 1), name = "input2")
        seq2 = Conv2D(filters = setting.nb_filter, kernel_size = (channels * fft_bins, 1),
                      kernel_regularizer=l2(l = setting.l2),
                      kernel_initializer = "lecun_uniform",
                      activation="relu")(input2)
        seq2 = Dropout(setting.dropout)(seq2)
        seq2 = Conv2D(filters = setting.nb_filter, kernel_size = (1, 3),
                      kernel_regularizer=l2(l = setting.l2),
                      kernel_initializer = "lecun_uniform",
                      activation="relu")(seq2)
        seq2 = Dropout(setting.dropout)(seq2)
        seq2 = Conv2D(filters = setting.nb_filter, kernel_size = (1, 3),
                      kernel_regularizer=l2(l = setting.l2),
                      kernel_initializer = "lecun_uniform",
                      activation="relu")(seq2)
        seq2 = Dropout(setting.dropout)(seq2)
        seq2 = Conv2D(filters = setting.nb_filter, kernel_size = (1, 3),
                      kernel_regularizer=l2(l = setting.l2),
                      kernel_initializer = "lecun_uniform",
                      activation="relu")(seq2)
        seq2 = Dropout(setting.dropout)(seq2)
        seq2 = Conv2D(filters = setting.nb_filter, kernel_size = (1, 3),
                      kernel_regularizer=l2(l = setting.l2),
                      kernel_initializer = "lecun_uniform",
                      activation="relu")(seq2)
        seq2 = Flatten()(seq2)
        output2 = Dense(setting.output2, activation="tanh")(seq2)

        merged = concatenate([output1, output2])
        merged = Dense(512, activation="tanh")(merged)
        merged = Dense(256, activation="tanh")(merged)
        merged = Dense(128, activation="tanh")(merged)
        output = Dense(2, activation="softmax", name="output")(merged)
        model = Model(inputs=[input1, input2], outputs=[output])
        sgd = SGD(lr = setting.lr)
        if setting.name == "Patient_1":
            model.compile(loss="binary_crossentropy", optimizer ="adam")
        else:
            model.compile(loss="binary_crossentropy", optimizer = sgd)

        return model
