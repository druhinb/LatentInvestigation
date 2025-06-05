import torch
from typing import Tuple, Union, Dict


class BasePreprocessor:
    """
    Base class for image preprocessors. Provides interface to transform raw image tensors
    into model inputs.
    """

    def __init__(self, device: str):
        self.device = device

    def preprocess(
        self, images: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, Dict], bool]:
        """
        Preprocess images and return (input, is_dict).
        If is_dict is True, input should be passed as kwargs to the model.
        Else, input is a direct tensor for the model.
        """
        raise NotImplementedError


class HFPreprocessor(BasePreprocessor):
    """
    Preprocessor for HuggingFace image models using an AutoImageProcessor.
    """

    def __init__(self, processor, device: str):
        super().__init__(device)
        self.processor = processor

    def preprocess(self, images: torch.Tensor):
        # Normalize and batch images via the HF processor
        imgs = images.to(self.device)
        inputs = self.processor(imgs, return_tensors="pt").to(self.device)
        return inputs, True


class TimmPreprocessor(BasePreprocessor):
    """
    Preprocessor for timm models using a torchvision-like transform.
    """

    def __init__(self, transform, device: str, size: Tuple[int, ...] = (224, 224)):
        super().__init__(device)
        self.transform = transform
        # expected input size: (channels, height, width) or (height, width)
        if hasattr(size, "__len__"):
            if len(size) == 3:
                _, h, w = size
            elif len(size) == 2:
                h, w = size
            else:
                h, w = 224, 224
        else:
            h, w = 224, 224
        self.size = {"height": h, "width": w}

    def preprocess(self, images: torch.Tensor):
        # Apply transform to each image and stack
        processed = torch.stack([self.transform(img) for img in images]).to(self.device)
        return processed, False


class IdentityPreprocessor(BasePreprocessor):
    """
    Fallback preprocessor that moves raw image tensors to device without other transforms.
    """

    def __init__(self, device: str):
        super().__init__(device)

    def preprocess(self, images: torch.Tensor):
        return images.to(self.device), False
