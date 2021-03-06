from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, ReLU, PReLU, Concatenate, Layer, Rescaling, GaussianNoise, Resizing
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPool2D, Flatten, Reshape, InputLayer

from tensorflow.keras import Sequential, layers, initializers, activations
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers import SGD, Nadam, Adam

from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

from tensorflow.keras.metrics import Metric
from tensorflow.keras.metrics import Accuracy, TruePositives, TrueNegatives, FalsePositives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC

import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def zero_one_loss(y_true, y_predict):
  """
  This function perform the zero-one evaluation over the passed labels
  Parameters:
    - y_true: the true value 
    - y_predict: the prediction
  Returns:
  The errors, 1 if the real value and the prediction are different, 0 otherwise
  """
  y = tf.math.round(y_predict)
  return tf.not_equal(y_true, y)

  
def binary_MLNN_model(input_shape,
                      hidden_layers: list,
                      hid_layers_act: str = 'ReLU',
                      outp_layer_act: str = 'sigmoid',
                      optimizer : Optimizer = SGD(learning_rate = 0.2),
                      loss: Loss = BinaryCrossentropy(),
                      metrs: list = [
                                        BinaryAccuracy(),
                                        Precision(),
                                        Recall(),
                                        AUC(),
                                        TruePositives(), 
                                        TrueNegatives(),
                                        FalsePositives(),
                                        FalseNegatives()
                                      ],
                      dropout = .2
                      
                      ) -> Model:
  """ 
  Build the structure of the ffnn model
  Parameters:
    - input_shape: the number of input that the model must handle
    - hidden_layers: an iterator containing the amount of neurons in each hidden layer
    - hid_layers_act: the activation function of the neurons in the hidden layers,
    - outp_layer_act: the activation function of the neurons in the output layer,
    - optimizer : The optimizer that will be used,
    - metrs : the list of metrics used to evaluate the model's performance
  Return:
    The compiled model
  """

  # Definition of the input and output (dense) layer
  input_layer = Input(shape = input_shape)
  
  pre = Flatten()(input_layer)
  
  hidden = Rescaling(1./255)(pre)
  
  for i in hidden_layers:
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = Dropout(dropout)(hidden)
    hidden = Dense(i, activation = hid_layers_act)(hidden)
    
  output_layer = Dense(1, activation = outp_layer_act)
  mlnn = Model(inputs = input_layer, outputs = output_layer(hidden))


  mlnn.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrs
    )
  return mlnn

def binary_CNN_model( input_shape : int, 
                      hidden_layers: list,
                      hid_layers_act: str = 'ReLU',
                      outp_layer_act: str = 'sigmoid',
                      optimizer : Optimizer = SGD(learning_rate = 0.01, momentum = 0.9),
                      loss: Loss = BinaryCrossentropy(),
                      metrs: list = [ zero_one_loss,                                    
                                      BinaryAccuracy()
                                    ],
                      kernel_size = (4, 4),
                      pool_size = (4, 4),
                      dropout_size = .1
                      ) -> Model:
  """ 
  Build the structure of the cnn classification model
  Parameters:
    - input_shape: the number of input that the model must handle
    - hidden_layers: an iterator containing the amount of neurons in each hidden layer, the kernel size and the pooling size
    - hid_layers_act: the activation function of the neurons in the hidden layers
    - outp_layer_act: the activation function of the neurons in the output layer
    - optimizer: the optimizer applied
    - metrs: the list of metrics that will be retrieved by the .fit and .evaluate functions of the model
    - kernel_size: the shape of the kernel and the pooling size
    - dropout_size: the percentage of dropout

  Return:
    The compiled model
  """   

  # Definition of the input and output (dense) layer
  
  cnn = Sequential()
  
  input_layer = Input(shape = input_shape)
  
  # Define and compile the model
  cnn.add(input_layer)
  cnn.add(Rescaling(1./255))

  for size in hidden_layers:

    cnn.add(Conv2D(size, kernel_size = kernel_size, padding = 'same', activation="relu"))

    cnn.add(MaxPool2D(kernel_size))
    cnn.add(Dropout(dropout_size))

  cnn.add(Flatten())

  cnn.add(Dense(128, activation="relu"))
  
  output_layer = Dense(1, activation = outp_layer_act)
  cnn.add(output_layer)

  cnn.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrs
    )
  return cnn



