# shiertier_image

中文 | [English](https://github.com/shiertier-utils/shiertier_image/blob/main/README_en.md)

## 1. 简介

`shiertier_image` 是一个用于图像处理的 Python 工具库。它提供了查找目录中的图像、加载和处理图像、调整图像大小等功能。

## 2. 安装

### 通过 pip 安装

```bash
pip install shiertier_image
```

### 通过 git 安装（开发版）

```bash
pip install git+https://github.com/shiertier-utils/shiertier_image.git
```

## 3. 使用示例

```python
from shiertier_image import ez_imgutils

# 查找目录中的图像
images = ez_imgutils.find_images_in_directory('path/to/directory')

# 加载图像
image = ez_imgutils.load_image('path/to/image.jpg')

# 调整图像大小
resized_image = ez_imgutils.resize_image(image, max_size=1024)

# 保存调整大小后的图像
ez_imgutils.resize_and_save_image('path/to/image.jpg', 'path/to/save_dir')
```

## 4. 许可证

本项目采用 [MIT 许可证](LICENSE)。 