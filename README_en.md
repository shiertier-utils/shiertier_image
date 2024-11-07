# shiertier_image

[中文](https://github.com/shiertier-utils/shiertier_image/blob/main/README.md) | English

## 1. Introduction

`shiertier_image` is a Python utility library for image processing. It provides functionalities such as finding images in a directory, loading and processing images, resizing images, and more.

## 2. Installation

### Install via pip

```bash
pip install shiertier_image
```

### Install via git (dev)

```bash
pip install git+https://github.com/shiertier-utils/shiertier_image.git
```

## 3. Usage Example

```python
from shiertier_image import ez_imgutils

# Find images in a directory
images = ez_imgutils.find_images_in_directory('path/to/directory')

# Load an image
image = ez_imgutils.load_image('path/to/image.jpg')

# Resize an image
resized_image = ez_imgutils.resize_image(image, max_size=1024)

# Save the resized image
ez_imgutils.resize_and_save_image('path/to/image.jpg', 'path/to/save_dir')
```

## 4. License

This project is licensed under the [MIT License](LICENSE). 