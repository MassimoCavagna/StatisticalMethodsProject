import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm_notebook
import math

import matplotlib.pyplot as plt
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

def save_json_history(filename: str, histories: dict, key: str):
  """
  This function saves the histories returned by the model training into JSON file
  
  Parameters:
    - filename: name of the JSON file the histories are saved into
    - histories: a dict containing the histories
    - key: the key in histories the function is going to update the value of
  
  Returns:
    None
  """
  if os.path.isfile(filename) is False:
    with open(filename, 'w') as f:
      json.dump({key: [h.history for h in histories]}, f)
      print("New file saved")
      f.close()
  else:
    with open(filename, 'r') as f:
      results = json.load(f)
      f.close()
      results[key] =  [h.history for h in histories]
      
    with open(filename, 'w') as f:
      json.dump(results, f)
      print("File saved")
      f.close()
      
def five_fold_cross_validation(model_f, parameters: dict, dataset: list, labels: list, epochs: int = 10, color_mode: int = 3, desc: str = "", img_size = (75,75)):
  """
  This function perform a five fold cross validation over the model_f passed,
  retrieving average error obtained
  Params:
  - model_f: the function to create the model
  - parameters: the list of parameters used to create the model
  - dataset: the list of paths to the images
  - labels: the list of labels
  - epochs: the nubmer of epochs for the training
  - color_mode: the number of channel of the iamge (1 or 3)
  - desc: the description of for the progress bar
  - color_mode: 1 or 3
  - batch_size: the batch size
  - img_size: a tuple of the size of the images
  Return:
  The average error of model_f with the specified parameters

  """
  kf = KFold(n_splits=5, shuffle=True, random_state=69)
  error = 0

  for train_index, val_index in tqdm_notebook(kf.split(dataset, labels), desc = f"Five fold {desc}:"):

      x_train, y_train = dataset[train_index], labels[train_index]
      x_valid, y_valid = dataset[val_index], labels[val_index]

      train_dataset = build_dataset(x_train, y_train, encoder_mode = False, target_size = img_size, color_mode = color_mode, batch_size = batch_size)
      validation_dataset = build_dataset(x_valid, y_valid, encoder_mode = False, target_size = img_size, color_mode = color_mode, batch_size = batch_size)

      model_copy = model_f(**parameters)
      
      model_copy.fit( train_dataset,
                      epochs=epochs,
                      verbose=0,
                      batch_size = 50
                    )
      # The first metric must be the 0-1 loss
      error += model_copy.evaluate(validation_dataset, verbose = 0)[1]
  return error/5

def nested_cross_validation(model_f, parameters : dict, th : str, dataset : list, labels : list, epochs : int, hyperp : list, color_mode : int = 3, img_size = (75, 75)):
  """
  This function perform a nested cross validation over the model_f passed,
  applying a 5-fold split and retrieving the hyperparameter in hyperp with
  the lowest average error among the hyperparameters selected by the internal 
  5-fold cross validation
  Params:
  - model_f: the function to create the model
  - parameters: the list of parameters used to create the model
  - th: the name of the parameter that will be tuned
  - dataset: the list of paths to the images
  - labels: the list of labels
  - epochs: the nubmer of epochs for the training
  - hyperp: the list of the hyperparameter over which the nested cross validation is performed
  - color_mode: 1 or 3
  - img_size: a tuple of the size of the images

  Return:
  The hyperparameter with the lowest average error

  """
  kf = KFold(n_splits=3, shuffle=True, random_state=69)
  final_theta = {i : [] for i in range(len(hyperp))}

  for train_index, val_index in tqdm_notebook(kf.split(dataset, labels), desc = "Nested:"):

      x_train, y_train = dataset[train_index], labels[train_index]
      x_valid, y_valid = dataset[val_index], labels[val_index]
      
      error = math.inf
      best_theta = 0

      # Search for best theta with this fold
      for i, theta in enumerate(hyperp):
        parameters[th] = theta

        error_ffcv = five_fold_cross_validation(model_f, parameters, x_train, y_train, epochs, color_mode, str(theta))

        if error_ffcv < error:
          error = error_ffcv
          best_theta = i
      
      # Re-train over the whole training set and evaluate with test set
      train_dataset = build_dataset(x_train, y_train, encoder_mode = False, target_size = img_size, color_mode = color_mode, batch_size = batch_size)
      test_dataset = build_dataset(x_valid, y_valid, encoder_mode = False, target_size = img_size, color_mode = color_mode, batch_size = batch_size)

      parameters[th] = hyperp[best_theta]
      model = model_f(**parameters)
      model.fit(train_dataset,
                epochs=epochs,
                verbose=0,
                batch_size = batch_size
               )
      
      final_theta[best_theta].append(model.evaluate(test_dataset, verbose = 0)[1])
  final_theta = {k : np.mean(final_theta[k]) if len(final_theta[k]) > 0 else math.inf for k in final_theta.keys()}
  return hyperp[min(final_theta, key = final_theta.get)], final_theta
