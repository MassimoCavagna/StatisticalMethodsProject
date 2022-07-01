import pandas as pd
import numpy as np

import matplotlip.pyplot as plt
import seaborn as sns

import tensorflow as tf


def plot_history(histories : list, title : str, same_figure = False):
  """
  This function is used to easily plot the history returned by any model in the form of a dictionary.
  For each metric it plots a lineplot describing the model's trend through all the epochs
  """
  if same_figure:
    fig = plt.figure(figsize = (15,5))
    fig.suptitle(title)

  
  df = pd.DataFrame()

  for i, history in enumerate(histories):
    if type(history) != dict:
      history = history.history

    keys, val_keys = [k for k in history.keys() if "val_" not in k], [k for k in history.keys() if "val_" in k]

    data = pd.DataFrame({k : history[k] for k in keys}, columns = keys)
    data["type"] = "T_" + str(i) + "_fold"
    data["epoch"] = list(range(len(data["type"])))

    val_data = pd.DataFrame({k.replace("val_", "") : history[k] for k in val_keys}, columns = keys)
    val_data["type"] = "V_" + str(i) + "_fold"
    val_data["epoch"] = list(range(len(val_data["type"])))

    if df.empty:
      df = pd.concat([data, val_data]).reset_index(drop=True)
    else:
      tmp = pd.concat([data, val_data]).reset_index(drop=True)
      df = pd.concat([df, tmp]).reset_index(drop=True)
    sns.set_style("darkgrid")
    
  df.sort_values(by=['type'], inplace = True)
  df.reset_index(drop=True)
  for i, k in enumerate(df.columns[0:-2]):
    n, is_val_empty = ((df.shape[0]/2)-1, False) if len(df[df.type.str.contains('V', case=False)]) > 0 else (df.shape[0]-1, True)
    plt.subplot(1, len(df.columns[0:-2]), 1 + i)
    plt.title(k)
    sns.lineplot(data = df.iloc[:int(n)], x = "epoch", y = k, hue = "type", palette = sns.color_palette(["blue"]*len(histories), len(histories)))
    if not is_val_empty:
      sns.lineplot(data = df.iloc[int(n+1):], x = "epoch", y = k, hue = "type", palette = sns.color_palette(["red"]*len(histories), len(histories)))
    
def mapping_function(image_path, label, target_size = (75, 75), color_mode = 1, encoder_mode: bool = False):
  """
  This function is used to read an image by its path, convert it to jpeg and resize it to the given target size.
  
  Parameters:
    - image_path: the path's image
    - label: the label associated with the image
    - target_size: the final size the image will have
    - color_mode: 1 for grayscale 3 for RGB
    - encoder_mode: by setting this to True, the labels corresponds to the image itself.
  
  Returns:
    An image, label tuple
  """
  image = tf.io.read_file(image_path)
  data = tf.io.decode_jpeg(image, channels = color_mode)
  data = tf.image.resize(data, target_size)
  image = tf.reshape(data, target_size + (color_mode, ) )
 
  return (image, label) if not encoder_mode else (image, image)

def build_dataset(data, labels, color_mode = 1, batch_size = 50, target_size = (75, 75), encoder_mode: bool = False):
  """
  This function uses "mapping function" to build the images dataset

  Parameters:
    - data: a list of images paths
    - labels: the associated labels
    - batch_size: the size of the batches the Dataset is divided into
    - target_size: the final size the image will have
    - encoder_mode: by setting this to True, the labels corresponds to the image itself

  Returns:
    A tensorflow Dataset
  """
  dataset = tf.data.Dataset.from_tensor_slices((data, labels))\
                            .map(
                                  lambda data, label: mapping_function( data, label, 
                                                                        encoder_mode = encoder_mode,
                                                                        target_size = target_size,
                                                                        color_mode = color_mode
                                                                      ), 
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE
                                 )\
                            .batch(batch_size)
  return dataset
