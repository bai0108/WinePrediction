from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, LayerNormalization, Dropout
from tensorflow.keras import initializers


def MyNet(features_num, output):
    input_data = Input(shape=features_num)
    kernel = initializers.RandomNormal(mean=0, stddev=.5)
    bias = initializers.Zeros()

    x = Dense(110, activation='relu', name='L1'
              , kernel_initializer=kernel, bias_initializer=bias)(input_data)

    x = LayerNormalization(name='Nor_L1')(x)
    x = Dense(110, activation='relu', name='L2'
              , kernel_initializer=kernel, bias_initializer=bias)(x)

    ''' classification '''
    output = Dense(output, activation='softmax', name='Prediction')(x)
    model = Model(input_data, output, name='MyNet_classification')

    return model


if __name__ == '__main__':
    MyNet(11, 7).summary()
