"""
Standalone face restoration module for Wav2Lip UHQ
Replaces modules.face_restoration from Stable Diffusion WebUI
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class FaceRestoration:
    """Standalone face restoration using CodeFormer or GFPGAN"""
    
    def __init__(self, model_name: str = "GFPGAN", code_former_weight: float = 0.75, 
                 device: str = "cuda", low_vram: bool = False):
        self.model_name = model_name
        self.code_former_weight = code_former_weight
        self.device = device
        self.low_vram = low_vram
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the face restoration model"""
        try:
            if self.model_name == "CodeFormer":
                from gfpgan import GFPGANer
                # CodeFormer is part of GFPGAN package
                self.model = GFPGANer(
                    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
                logger.info("CodeFormer model loaded successfully")
                
            elif self.model_name == "GFPGAN":
                from gfpgan import GFPGANer
                self.model = GFPGANer(
                    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
                logger.info("GFPGAN model loaded successfully")
            else:
                raise ValueError(f"Unknown face restoration model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load face restoration model: {e}")
            raise
    
    def restore_faces(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Restore faces in the given image
        
        Args:
            image: Input image as numpy array or PIL Image
            
        Returns:
            Restored image as numpy array
        """
        if self.model is None:
            logger.warning("Face restoration model not loaded, returning original image")
            if isinstance(image, Image.Image):
                return np.array(image)
            return image
        
        try:
            # Convert PIL to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure image is in RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Already RGB
                pass
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA to RGB
                image = image[:, :, :3]
            else:
                logger.warning(f"Unexpected image shape: {image.shape}")
                return image
            
            # Apply face restoration
            if self.model_name == "CodeFormer":
                # CodeFormer specific processing
                result = self.model.enhance(
                    image, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True,
                    weight=self.code_former_weight
                )
                logger.debug(f"CodeFormer result type: {type(result)}, length: {len(result) if isinstance(result, (tuple, list)) else 'N/A'}")
                # Handle different return formats - CodeFormer can return multiple values
                if isinstance(result, tuple):
                    if len(result) >= 1:
                        restored_image = result[0]
                        logger.debug(f"CodeFormer tuple[0] type: {type(restored_image)}")
                    else:
                        logger.error("CodeFormer returned empty tuple")
                        return image
                elif isinstance(result, list):
                    if len(result) >= 1:
                        restored_image = result[0]
                        logger.debug(f"CodeFormer list[0] type: {type(restored_image)}")
                    else:
                        logger.error("CodeFormer returned empty list")
                        return image
                else:
                    restored_image = result
                    logger.debug(f"CodeFormer direct result type: {type(restored_image)}")
            else:
                # GFPGAN processing
                result = self.model.enhance(
                    image, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
                logger.debug(f"GFPGAN result type: {type(result)}, length: {len(result) if isinstance(result, (tuple, list)) else 'N/A'}")
                # Handle different return formats - GFPGAN can return multiple values
                if isinstance(result, tuple):
                    if len(result) >= 1:
                        restored_image = result[0]
                        logger.debug(f"GFPGAN tuple[0] type: {type(restored_image)}")
                    else:
                        logger.error("GFPGAN returned empty tuple")
                        return image
                elif isinstance(result, list):
                    if len(result) >= 1:
                        restored_image = result[0]
                        logger.debug(f"GFPGAN list[0] type: {type(restored_image)}")
                    else:
                        logger.error("GFPGAN returned empty list")
                        return image
                else:
                    restored_image = result
                    logger.debug(f"GFPGAN direct result type: {type(restored_image)}")
            
            # Handle different return types from the model
            if isinstance(restored_image, list):
                # If it's a list, take the first element
                if len(restored_image) > 0:
                    restored_image = restored_image[0]
                else:
                    logger.error("Model returned empty list")
                    return image
            
            # Convert to numpy array if it's not already
            if not isinstance(restored_image, np.ndarray):
                try:
                    restored_image = np.array(restored_image)
                except Exception as e:
                    logger.error(f"Failed to convert result to numpy array: {e}")
                    return image
            
            # Ensure output is uint8
            if hasattr(restored_image, 'dtype') and restored_image.dtype != np.uint8:
                restored_image = np.clip(restored_image, 0, 255).astype(np.uint8)
            
            return restored_image
            
        except Exception as e:
            logger.error(f"Face restoration failed: {e}")
            # Return original image on failure
            if isinstance(image, Image.Image):
                return np.array(image)
            return image
    
    def cleanup(self):
        """Clean up model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Face restoration model cleaned up")


def restore_faces(image: Union[np.ndarray, Image.Image], 
                 model_name: str = "GFPGAN", 
                 code_former_weight: float = 0.75,
                 device: str = "cuda",
                 low_vram: bool = False) -> np.ndarray:
    """
    Convenience function to restore faces in an image
    
    Args:
        image: Input image as numpy array or PIL Image
        model_name: Face restoration model ("CodeFormer" or "GFPGAN")
        code_former_weight: CodeFormer fidelity weight (0.0-1.0)
        device: Device to use ("cuda" or "cpu")
        low_vram: Whether to use low VRAM mode
        
    Returns:
        Restored image as numpy array
    """
    restorer = FaceRestoration(
        model_name=model_name,
        code_former_weight=code_former_weight,
        device=device,
        low_vram=low_vram
    )
    
    try:
        result = restorer.restore_faces(image)
        return result
    finally:
        restorer.cleanup()
