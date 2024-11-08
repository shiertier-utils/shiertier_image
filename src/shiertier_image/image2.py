from pathlib import Path
from typing import Union, List, BinaryIO, Optional, Tuple
from PIL import Image, ImageColor
import numpy as np
import os
from shiertier_logger import ez_logger as logger

ImageTyping = Union[str, Path, bytes, bytearray, BinaryIO, Image.Image]
MultiImagesTyping = Union[ImageTyping, List[ImageTyping], Tuple[ImageTyping, ...]]
_AlphaTyping = Union[float, np.ndarray]

class Config:
    """配置类，存储所有静态配置"""
    DEFAULT_IMAGE_EXTENSIONS_LIST_STR = "jpg;jpeg;png;bmp;webp"
    DEFAULT_RESIZE_BUCKETS = {
        2048: [(960, 4352), (1024, 4096), (1088, 3840), (1152, 3648), (1216, 3456), (1280, 3264), 
               (1344, 3136), (1600, 2624), (1728, 2432), (1920, 2176), (1984, 2112), (2048, 2048), 
               (2112, 1984), (2176, 1920), (2432, 1728), (2624, 1600), (3136, 1344), (3264, 1280), 
               (3456, 1216), (3648, 1152), (3840, 1088), (4096, 1024), (4352, 960)], 
        1536: [(704, 3328), (768, 3072), (832, 2816), (896, 2624), (1024, 2304), (1088, 2176), 
               (1152, 2048), (1280, 1856), (1408, 1664), (1472, 1600), (1536, 1536), (1600, 1472), 
               (1664, 1408), (1856, 1280), (2048, 1152), (2176, 1088), (2304, 1024), (2624, 896), 
               (2816, 832), (3072, 768), (3328, 704)], 
        1024: [(512, 2048), (576, 1792), (640, 1664), (704, 1472), (768, 1344), (832, 1280), 
               (896, 1152), (960, 1088), (1024, 1024), (1088, 960), (1152, 896), (1280, 832), 
               (1344, 768), (1472, 704), (1664, 640), (1792, 576), (2048, 512)], 
        768: [(384, 1536), (448, 1344), (512, 1152), (576, 1024), (704, 832), (768, 768), 
               (832, 704), (1024, 576), (1152, 512), (1344, 448), (1536, 384)], 
        512: [(256, 1024), (320, 832), (384, 704), (448, 576), (512, 512), (576, 448), 
               (704, 384), (832, 320), (1024, 256)]
    }
    author = "shiertier"
    email = "junjie.text@gmail.com"
    requirements = [
        "Pillow", 
        "numpy", 
        "opencv-python", 
        "shiertier_logger"
    ]

class LoadUtils:
    """图像加载相关的功能"""
    def __init__(self, image_extensions_list: List[str] = None):
        self.image_extensions_list = image_extensions_list or os.environ.get(
            'IMAGE_EXTENSIONS_LIST_STR', 
            Config.DEFAULT_IMAGE_EXTENSIONS_LIST_STR
        ).split(';')

    def _is_readable(self, obj) -> bool:
        """检查对象是否可读"""
        return hasattr(obj, 'read') and hasattr(obj, 'seek')

    def _has_alpha_channel(self, image: Image.Image) -> bool:
        """检查图像是否有alpha通道"""
        return any(band in {'A', 'a', 'P'} for band in image.getbands())

    def _load_image_or_color(self, image) -> Union[str, Image.Image]:
        """加载图像或返回颜色字符串"""
        if isinstance(image, str):
            try:
                _ = ImageColor.getrgb(image)
            except ValueError:
                pass
            else:
                return image
        return self.load_image(image, mode='RGBA', force_background=None)

    def _process(self, item) -> Tuple[Union[str, Image.Image], float]:
        """处理图像项目及其alpha值"""
        if isinstance(item, tuple):
            image, alpha = item
        else:
            image, alpha = item, 1
        return self._load_image_or_color(image), alpha

    def _add_alpha(self, image: Image.Image, alpha: _AlphaTyping) -> Image.Image:
        """添加或修改图像的alpha通道"""
        data = np.array(image.convert('RGBA')).astype(np.float32)
        data[:, :, 3] = (data[:, :, 3] * alpha).clip(0, 255)
        return Image.fromarray(data.astype(np.uint8), mode='RGBA')

    def find_images_in_directory(self, directory: Path, recursive: bool = True) -> List[Path]:
        """查找目录中的所有图像"""
        images = []
        for item in directory.iterdir():
            if item.is_file() and item.suffix.lower().lstrip('.') in self.image_extensions_list:
                images.append(item)
            elif item.is_dir() and recursive:
                images.extend(self.find_images_in_directory(item, recursive))
        return images
    
    def filter_images_with_txt(self, image_paths: List[Path]) -> List[Path]:
        """过滤带有对应txt文件的图像"""
        image_paths_with_caption = [path for path in image_paths if Path(path).with_suffix(".txt").exists()]
        logger.info(f"Info: {len(image_paths_with_caption)} captions exist")
        image_paths_set = set(image_paths)
        image_paths_with_caption_set = set(image_paths_with_caption)
        difference = image_paths_set - image_paths_with_caption_set
        return list(difference)

    def load_image(self, image: ImageTyping, mode=None, force_background: Optional[str] = 'white') -> Image.Image:
        """加载单个图像"""
        if isinstance(image, (str, Path, bytes, bytearray, BinaryIO)) or self._is_readable(image):
            image = Image.open(image)
        elif isinstance(image, Image.Image):
            pass  # just do nothing
        else:
            raise TypeError(f'Unknown image type - {image!r}.')

        if self._has_alpha_channel(image) and force_background is not None:
            image = self.add_background_for_rgba(image, force_background)

        if mode is not None and image.mode != mode:
            image = image.convert(mode)

        return image

    def load_images(self, images: MultiImagesTyping, mode=None, 
                   force_background: Optional[str] = 'white') -> List[Image.Image]:
        """加载多个图像"""
        if not isinstance(images, (list, tuple)):
            images = [images]
        return [self.load_image(item, mode, force_background) for item in images]

    def istack(self, *items: Union[ImageTyping, str, Tuple[ImageTyping, _AlphaTyping], 
               Tuple[str, _AlphaTyping]], size: Optional[Tuple[int, int]] = None) -> Image.Image:
        """堆叠多个带有alpha通道的图像或颜色"""
        if size is None:
            height, width = None, None
            items = list(map(self._process, items))
            for item, alpha in items:
                if isinstance(item, Image.Image):
                    height, width = item.height, item.width
                    break
        else:
            width, height = size

        if height is None:
            raise ValueError('Unable to determine image size, please make sure '
                           'you have provided at least one image object.')

        retval = Image.fromarray(np.zeros((height, width, 4), dtype=np.uint8), mode='RGBA')
        for item, alpha in items:
            if isinstance(item, str):
                current = Image.new("RGBA", (width, height), item)
            elif isinstance(item, Image.Image):
                current = item
            else:
                raise ValueError(f'Invalid type - {item!r}.')

            current = self._add_alpha(current, alpha)
            retval.paste(current, mask=current)

        return retval

    def add_background_for_rgba(self, image: ImageTyping, background: str = 'white') -> Image.Image:
        """为RGBA图像添加背景"""
        return self.istack(background, image)

class ResizeUtils:
    """图像缩放相关的功能"""
    def __init__(self, buckets: dict = None):
        self.buckets = buckets or Config.DEFAULT_RESIZE_BUCKETS

    def resize_image(self, image: ImageTyping, method: int = Image.BICUBIC, 
                    max_size: int = 2048, min_size: int = 1024) -> Optional[Image.Image]:
        """调整图像大小"""
        image = self.load_image(image)

        width, height = image.size
        total_resolution = width * height

        def get_size_key():
            for key in self.buckets.keys():
                if int(key) > min_size:
                    if total_resolution >= int(key) * 0.95:
                        if int(key) > max_size:
                            return max_size
                        else:
                            return key
            return None

        size_key = get_size_key()

        if size_key is None:
            return None

        def get_bucket_size():
            closest_pair = min(self.buckets[size_key], key=lambda pair: abs(width / height - pair[0] / pair[1]))
            ratio = max(closest_pair[0] / width, closest_pair[1] / height)
            new_width = int(ratio * width)
            new_height = int(ratio * height)
            crop_x = (new_width - closest_pair[0]) // 2
            crop_y = (new_height - closest_pair[1]) // 2
            return new_width, new_height, crop_x, crop_y

        new_width, new_height, crop_x, crop_y = get_bucket_size()
        new_width = (new_width // 2) * 2  # to avoid odd number
        new_height = (new_height // 2) * 2  # to avoid odd number

        with image.resize((new_width, new_height), resample=method) as img_new:
            cropped_image = img_new.crop((crop_x, crop_y, new_width - crop_x, new_height - crop_y))

        return cropped_image
    
    def resize_and_save_image(self, image_path: str, save_dir: str, **kwargs):
        """调整图像大小并保存"""
        if image_dir:
            relative_path = os.path.relpath(image_path, image_dir)
            base_name = os.path.splitext(relative_path)[0]
        else:
            base_name = os.path.splitext(os.path.basename(image_path))[0]

        lossless = os.path.getsize(image_path) > 1024 ** 2
        save_path = os.path.join(save_dir, f"{base_name}.{'webp' if lossless else 'jpg'}")

        if not overwrite and os.path.exists(save_path):
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        crop_image = self.resize_image(image_path, max_size=max_size, min_size=min_size, buckets=buckets)

        if crop_image is None:
            logger.error(f"Failed to resize image: {image_path}")
            return

        if lossless:
            crop_image.save(save_path, format='WEBP', quality=100, lossless=True)
        else:
            crop_image.save(save_path, format='JPEG', quality=92)  # save as JPEG 92

        crop_image.close() 

class ImageUtils:
    """主类，整合所有功能"""
    def __init__(self, image_extensions_list: List[str] = None, buckets: dict = None):
        self.config = Config()
        self.load = LoadUtils(image_extensions_list)
        self.resize = ResizeUtils(buckets)
        
    @property
    def help(self):
        help_text = """
        ImageUtils class provides various utilities for image processing.

        Methods:
            - find_images_in_directory(directory: Path, recursive: bool = True) -> List[Path]
                Find all images in a directory.

            - logging_if_none_image(image_paths: List[Path], raise_error: bool = True)
                Log a warning or raise an error if no images are found.

            - filter_images_with_txt(image_paths: List[Path]) -> List[Path]
                Filter images that have corresponding .txt files.

            - load_image(image: ImageTyping, mode=None, force_background: Optional[str] = 'white') -> Image.Image
                Load an image from various input types.

            - load_images(images: MultiImagesTyping, mode=None, force_background: Optional[str] = 'white') -> List[Image.Image]
                Load multiple images.

            - add_background_for_rgba(image: ImageTyping, background: str = 'white') -> Image.Image
                Add a background to an RGBA image.

            - resize_image(image: ImageTyping, method: int = Image.BICUBIC, max_size: int = 2048, min_size: int = 1024, buckets: dict = None) -> Optional[Image.Image]
                Resize an image to fit within specified size buckets.

            - resize_and_save_image(image_path: str, save_dir: str, overwrite: bool = False, image_dir: str = None, max_size: int = 2048, min_size: int = 1024, buckets: dict = None)
                Resize an image and save it to a specified directory.
        """
        print(help_text)

    @property
    def help_zh(self):
        help_text_zh = """
        ImageUtils 类提供了各种图像处理的实用工具。

        方法:
            - find_images_in_directory(directory: Path, recursive: bool = True) -> List[Path]
                查找目录中的所有图像。

            - logging_if_none_image(image_paths: List[Path], raise_error: bool = True)
                如果没有找到图像，则记录警告或引发错误。

            - filter_images_with_txt(image_paths: List[Path]) -> List[Path]
                过滤出具有对应 .txt 文件的图像。

            - load_image(image: ImageTyping, mode=None, force_background: Optional[str] = 'white') -> Image.Image
                从各种输入类型加载图像。

            - load_images(images: MultiImagesTyping, mode=None, force_background: Optional[str] = 'white') -> List[Image.Image]
                加载多个图像。

            - add_background_for_rgba(image: ImageTyping, background: str = 'white') -> Image.Image
                为 RGBA 图像添加背景。

            - resize_image(image: ImageTyping, method: int = Image.BICUBIC, max_size: int = 2048, min_size: int = 1024, buckets: dict = None) -> Optional[Image.Image]
                调整图像大小以适应指定的大小桶。

            - resize_and_save_image(image_path: str, save_dir: str, overwrite: bool = False, image_dir: str = None, max_size: int = 2048, min_size: int = 1024, buckets: dict = None)
                调整图像大小并将其保存到指定目录。
        """
        print(help_text_zh)

    # 完整的代理方法
    def find_images_in_directory(self, *args, **kwargs):
        return self.load.find_images_in_directory(*args, **kwargs)

    def filter_images_with_txt(self, *args, **kwargs):
        return self.load.filter_images_with_txt(*args, **kwargs)

    def load_image(self, *args, **kwargs):
        return self.load.load_image(*args, **kwargs)

    def load_images(self, *args, **kwargs):
        return self.load.load_images(*args, **kwargs)

    def add_background_for_rgba(self, *args, **kwargs):
        return self.load.add_background_for_rgba(*args, **kwargs)

    def istack(self, *args, **kwargs):
        return self.load.istack(*args, **kwargs)

    def resize_image(self, *args, **kwargs):
        return self.resize.resize_image(*args, **kwargs)

    def resize_and_save_image(self, *args, **kwargs):
        return self.resize.resize_and_save_image(*args, **kwargs)

    def logging_if_none_image(self, image_paths: List[Path], raise_error: bool = True):
        """记录警告或引发错误（如果未找到图像）"""
        if len(image_paths) == 0:
            if raise_error:
                raise ValueError("No images found")
            else:
                logger.warning("Warning: No images found")

ez_imgutils = ImageUtils()