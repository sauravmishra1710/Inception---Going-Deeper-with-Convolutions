import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Input, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GaussianNoise, GaussianDropout
from tensorflow.keras.layers import Activation, Dropout, ZeroPadding3D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

class InceptionFramework:
    
    def __init__(self):
        
        pass
    
    
    def Naive_Inception_Module(self, in_layer, f1x1, f3x3, f5x5):
        
        # 1x1 conv
        conv1x1 = Conv2D(f1x1, (1,1), padding='same', activation=tf.nn.relu)(in_layer)
        
        # 3x3 conv
        conv3x3 = Conv2D(f3x3, (3,3), padding='same', activation=tf.nn.relu)(in_layer)
        
        # 5x5 conv
        conv5x5 = Conv2D(f5x5, (5,5), padding='same', activation=tf.nn.relu)(in_layer)
        
        # 3x3 max pooling
        pool3x3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(in_layer)
        
        # concatenate the convolution and the pooling layers,
        out_layer = concatenate([conv1x1, conv3x3, conv5x5, pool3x3], axis = -1)
        
        return out_layer
    
    def Create_Inception_Block(self, in_layer, f1x1, f3x3_red, f3x3, f5x5_red, f5x5, fpool):
        
        # 1x1 conv
        conv1x1 = Conv2D(f1x1, (1,1), padding='same', activation=tf.nn.relu)(in_layer)

        # 3x3 reduce and conv
        conv3x3_red = Conv2D(f3x3_red, (1,1), padding='same', activation=tf.nn.relu)(in_layer)
        conv3x3 = Conv2D(f3x3, (3,3), padding='same', activation=tf.nn.relu)(conv3x3_red)

        # 5x5 reduce and conv
        conv5x5_red = Conv2D(f5x5_red, (1,1), padding='same', activation=tf.nn.relu)(in_layer)
        conv5x5 = Conv2D(f5x5, (5,5), padding='same', activation=tf.nn.relu)(conv5x5_red)

        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(in_layer)
        pool = Conv2D(fpool, (1,1), padding='same', activation=tf.nn.relu)(pool)

        # concatenate the convolutional layers , poling layer and pass on
        # to the next layers.
        out_layer = concatenate([conv1x1, conv3x3, conv5x5, pool], axis=-1)
        
        return out_layer
    
    def Build_Inception_Network(self, INPUT_SHAPE=(135, 135, 3)):
        
        # define model input
        visible = Input(shape = INPUT_SHAPE)
        
        # add inception block 1
        layer = self.Create_Inception_Block(visible, 64, 96, 128, 16, 32, 32)
        
        # add inception block 1
        layer = self.Create_Inception_Block(layer, 128, 128, 192, 32, 96, 32)

        gap = GlobalAveragePooling2D(data_format='channels_last')(layer)
        dense = Dense(1024, activation='relu')(gap)
        dropout = Dropout(0.5)(dense)
        out  = Dense(1, activation='sigmoid')(dropout)

        # create model
        model = Model(inputs=visible, outputs=out, name='inception_network')
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        # summarize model
        print(model.summary())
        return model