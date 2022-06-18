from PIL import Image, ImageChops, UnidentifiedImageError, ImageOps
from PIL.Image import NEAREST
from PIL.JpegImagePlugin import JpegImageFile
from tqdm.notebook import tqdm_notebook
import os

def image_info(image_path : str) -> None:
  """
  This function reads the specified image, prints its informations and displays them
  
  Parameters:
  - image_path: the path to the image

  Returns:
  None
  """
  with Image.open(image_path) as im:
    info = {key: im.__dict__[key] for key in ['filename', '_size','mode']}
    info['format'] = im.format
    print(info)
    display(im)
    
def get_images_format(source_path: str) -> list:
  """
  This function returns all the formats of the images inside the specified source path.
  NOTE: if the image cannot be opened a 'Unidentified' format will be added into the output list.

  Parameters:
  - source_path: path of the directory containing the images.

  Returns:
  a list of formats
  """
  s = set()
  for name in tqdm_notebook(os.listdir(source_path), desc = f"Retrieving formats from {source_path}"):
      try:
        with Image.open(source_path + name) as im:
          s.add(im.format)
      except UnidentifiedImageError:
        s.add('Unidentified')
  return list(s)

def to_RGB_JPEG(source_path : str):
  """
  This function converts all the images in the specified source path into an JPEG image in RGB

  Parameters:
  - source_path: path of the directory containing the images.

  Returns:
  The list containing the path to the images that raised an error
  """
  errs = []
  images_names = os.listdir(source_path)
  for name in tqdm_notebook(images_names, desc = "Converting"):
    try:
      with Image.open(f'{source_path}/{name}') as im:
        if not isinstance(im, JpegImageFile):
          bg = Image.new("RGB", im.size, (255,255,255))
          bg.paste(im)
          bg.save(f'{source_path}/{name}')
        elif im.mode != "RGB":
          im = im.convert(mode = "RGB")
          im.save(f'{source_path}/{name}')
    except UnidentifiedImageError:
      errs.append(source_path + name)
      pass
  return errs

def sum_image_sizes(source_path : str) -> tuple:
  """
  This function sums up all separately the images dimensions

  Parameters:
  - source_path: path of the directory containing the images.

  Returns:
  the x and y axis images dimensions summed up and the number of images in the directory.
  """
  x = 0
  y = 0
  images_names = os.listdir(source_path)

  for name in tqdm_notebook(images_names, desc = "Computing"):
    with Image.open(f'{source_path}/{name}') as im:
      x += im.size[0]
      y += im.size[1]

  return x, y, len(images_names)

def crop_resize_images(source_path : str, destination_path : str = None, target_size: tuple = (400, 400)) -> None:
  """
  This function reads, crops and resizes all the images with respect to its arguments.
  
  Parameters:
  - source_path: path of the directory containing the images.
  - destination_path: the directory's path into which the new images are saved.
    NOTE: if the destination_dir is not provided the function saves all the new images in the source directory, so overwriting the old content.
  - target_size: a tuple containing the sizes the function will convert the images to.

  Returns:
  None
  """
  if not destination_path:
    destination_path = source_path
  elif not os.path.isdir(destination_path):
    os.mkdir(destination_path)

  images_names = os.listdir(source_path)
  
  sort_keys = lambda x: int(x.split('.')[0])

  border_color = (255, 255, 255)

  for name in tqdm_notebook(sorted(images_names, key = sort_keys ), desc = f"Converting {source_path}"):
    with Image.open(source_path + '/' + name) as im:
      bg = Image.new(im.mode, im.size, border_color)
      diff = ImageChops.difference(im, bg)
      diff = ImageChops.add(diff, diff, 2.0, offset = -100)
      bbox = diff.getbbox()
      if bbox:
        im = im.crop(bbox)

      im.resize(target_size).save(destination_path + name)
