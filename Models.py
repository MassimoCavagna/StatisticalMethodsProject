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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def binary_FFNN_model(encoder: Model,
                      input_shape,
                      hidden_layers: list,
                      hid_layers_act: str = 'ReLU',
                      outp_layer_act: str = 'sigmoid',
                      optimizer : Optimizer = Adam(learning_rate=.01),
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
                      
                      ) -> Model:
  """ 
  Build the structure of the ffnn model
  Parameters:
    - encoder: a pre trained encoder which filters out the images
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
  if encoder:
    for k,v in encoder._get_trainable_state().items():
      k.trainable = False
    pre = encoder(input_layer)
  else:
    pre = Flatten()(input_layer)
  
  hidden = Rescaling(1./255)(pre)
  
  for i in hidden_layers:
    hidden = BatchNormalization()(hidden)
    hidden = ReLU()(hidden)
    hidden = Dropout(.3)(hidden)
    hidden = Dense(i, activation = hid_layers_act)(hidden)
    
  output_layer = Dense(1, activation = outp_layer_act)
  ffnn = Model(inputs = input_layer, outputs = output_layer(hidden))


  ffnn.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = metrs
    )
  return ffnn

def build_autoencoder(img_shape, code_size):
    """
    This function build an autoecoder with a encoding layer of the given code_size.
    Parameters:
      - img_shape: the image shape used in the input and output layer of the model
      - code_size: the number of neurons used to encode the image
    Returns:
      - The compiled autoencoder and the compiled encoder
    """

    encoder_input = Input(shape = img_shape)
    encoder = Flatten()(encoder_input)
    encoder_output = Dense(code_size)(encoder)

    encoder = Model(encoder_input, encoder_output, name = 'Encoder')

    decoder = Dense(np.prod(img_shape))(encoder_output)
    reconstruction = Reshape(img_shape)(decoder)


    autoencoder = Model(encoder_input,reconstruction, name = 'AutoEncoder')
    autoencoder.compile(optimizer='adamax', loss='mse')

    return autoencoder, encoder
  
def plot_history(histories : list, same_figure = False):
  """
  This function is used to easily plot the history returned by any model in the form of a dictionary.
  For each metric it plots a lineplot describing the model's trend through all the epochs
  """
  if same_figure:
    plt.figure(figsize = (15,5))

  
  df = pd.DataFrame()

  for i, history in enumerate(histories):
    if type(history) != dict:
      history = history.history
    keys, val_keys = [k for k in history.keys() if "val_" not in k], [k for k in history.keys() if "val_" in k]

    data = pd.DataFrame({k : history[k] for k in keys}, columns = keys)
    data["type"] = "T " + str(i) + "-th fold"
    data["epoch"] = list(range(len(data["type"])))

    val_data = pd.DataFrame({k.replace("val_", "") : history[k] for k in val_keys}, columns = keys)
    val_data["type"] = "V " + str(i) + "-th fold"
    val_data["epoch"] = list(range(len(val_data["type"])))

    if df.empty:
      df = pd.concat([data, val_data]).reset_index(drop=True)
    else:
      df = pd.concat([df, data, val_data]).reset_index(drop=True)
    sns.set_style("darkgrid")
    
  df.sort_values(by=['type'], inplace = True)
  df.reset_index(drop=True)

  for i, k in enumerate(df.columns[0:-2]):
    n, is_val_empty = ((df.shape[0]/2)-1, False) if len(df[df.type.str.contains('V',case=False)]) > 0 else (df.shape[0]-1, True)
    plt.subplot(1, len(df.columns[0:-2]), 1 + i)
    plt.title(k)
    sns.lineplot(data = df.loc[:n], x = "epoch", y = k, hue = "type", palette = sns.color_palette("Blues", 5))
    if not is_val_empty:
      sns.lineplot(data = df.loc[n+1:], x = "epoch", y = k, hue = "type", palette = sns.color_palette("magma", 0))
    
