import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, Input, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GaussianNoise, GaussianDropout
from tensorflow.keras.layers import Activation, Dropout, ZeroPadding3D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam

class InceptionFramework:
    
    def __init__():
        
        pass
    
    
    def vanilla_inception_module(in_layer, f1, f2, f3):
        
        # 1x1 conv
        conv1x1 = Conv2D(f1, (1,1), padding='same', activation=tf.nn.relu)(in_layer)
        
        # 3x3 conv
        conv3x3 = Conv2D(f2, (3,3), padding='same', activation=tf.nn.relu)(in_layer)
        
        # 5x5 conv
        conv5x5 = Conv2D(f3, (5,5), padding='same', activation=tf.nn.relu)(in_layer)
        
        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding=tf.nn.relu)(in_layer)
        
        # concatenate the convolution layers, assumes filters/channels last
        layer_out = concatenate([conv1x1, conv3x3, conv5x5, pool], axis = -1)
        
        return layer_out
    
    def Create_Inception_Block(self, input_layer, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        
        # 1x1 conv
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(input_layer)

        # 3x3 conv
        conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(input_layer)
        conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)

        # 5x5 conv
        conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(input_layer)
        conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)

        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_layer)
        pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)

        # concatenate the output of the convolutional layers and pass on
        # to the next layers
        output_layer = concatenate([conv1, conv3, conv5, pool], axis=-1)
        return output_layer
    
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