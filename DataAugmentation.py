from PIL import Image, ImageChops, UnidentifiedImageError, ImageOps, ImageFilter, ImageEnhance
import os
import numpy as np
from tqdm.notebook import tqdm_notebook
import random

def extract_random_images(source_path : str, per : float = .1) -> list:
  """
  This function extracts randomly a portion of images from the specified source path.
  Parameters:
  - source_path: path of the directory containing the images.
  - per: the percentage of images that must be extracted.

  Returns:
  list of images named extracted
  """
  if per >= 0.0 and per <= 1.0:
    images = os.listdir(source_path)
    selection = random.sample(images, round(len(images)*per))
    return [source_path + image_name for image_name in selection]
  else:
    raise Exception('Invalid per value: it must be a float between 0 and 1')

def transform_images(source_paths : list, destination_path : str, transformation : str):
  """
  This function applies the specified transformation over the images whose file path is contained in source_paths
  and saves the resulting images in the destination_path' directory

  Parameter:
  - source_paths: list of images file paths in the form <path><file_name>.
  - destination_path: the directory's path into which the new images are saved.
  - transformation: a string stating the desired transformation to be applied.
  NOTE: this value must be one of the following:
        'mirror'
        'rotate': the rotation angle is randomly chosen
        'flip'
  Returns:
  None
  """

  t = ['mirror' , 'rotate', 'flip']
  if transformation not in t:
    raise Exception(f"Unexpected transformation value: it must be one of the following {t}")

  if not os.path.isdir(destination_path):
    os.mkdir(destination_path)

  for name in tqdm_notebook(source_paths, desc = "applying '" + transformation + "' over " + source_paths[0].split('/')[1]):
    with Image.open(name) as im:
      if transformation == 'mirror':  
        ImageOps.mirror(im).save(destination_path + "m_" + name.split("/")[-1])
      elif transformation == 'rotate':
        theta = np.random.randint(1, 18)*10
        theta = -theta if np.random.random() <= 0.5 else theta
        Image.Image.rotate(im, theta).save(destination_path + "r_" + name.split("/")[-1])
      elif transformation == 'flip':
        ImageOps.flip(im).save(destination_path + "f_" + name.split("/")[-1])
    im.close()

def apply_filter(source_paths : list, destination_path : str, filter : str):
  """
  This function applies the specified filter over the images whose file path is contained in source_paths
  and saves the resulting images in the destination_path' directory

  Parameter:
  - source_paths: list of images file paths in the form <path><file_name>.
  - destination_path: the directory's path into which the new images are saved.
  - filter: a string stating the desired filter to be applied.
  NOTE: this value must be one of the following:
        "blur", "unsharp", "spread", "contrast", "color"   
  Returns:
  None
  """

  f = ["blur", "unsharp", "spread", "contrast", "color"]
  if filter not in f:
    raise Exception(f"Unexpected filter value: it must be one of the following {f}")

  if not os.path.isdir(destination_path):
    os.mkdir(destination_path)

  for name in tqdm_notebook(source_paths, desc = "applying '" + filter + "' over " + source_paths[0].split('/')[1]):
    with Image.open(name) as im:
      if filter == 'blur':
        im.filter(ImageFilter.BoxBlur(5)).save(destination_path + "b_" + name.split("/")[-1])
      elif filter == 'unsharp':
        im.filter(ImageFilter.UnsharpMask(5, 500, 0)).save(destination_path + "u_" + name.split("/")[-1])
      elif filter == 'spread':
        im.effect_spread(15).save(destination_path + "s_" + name.split("/")[-1])
      elif filter == 'contrast':
        ImageEnhance.Contrast(im).enhance(100.).save(destination_path + "c_" + name.split("/")[-1])
      elif filter == 'color':
        ImageEnhance.Color(im).enhance(1.5).save(destination_path + "cl_" + name.split("/")[-1])
    im.close()
