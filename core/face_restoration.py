import os
import logging
import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FaceRestoration:
    def __init__(self, device='cuda', low_vram=False):
        self.device = device
        self.low_vram = low_vram
        self.model = None
        self.model_name = None

    def _load_model(self, model_name):
        """Load the face restoration model"""
        if self.model_name == model_name and self.model is not None:
            return  # Model already loaded
            
        try:
            # Normalize model name to handle case variations
            model_name_lower = model_name.lower()
            if model_name_lower == 'codeformer':
                from basicsr.archs.codeformer_arch import CodeFormer
                from basicsr.utils.realesrgan_utils import RealESRGANer
                from facexlib.utils.face_restoration_helper import FaceRestoreHelper
                
                # Load CodeFormer model
                model_path = os.path.join('weights', 'codeformer', 'codeformer.pth')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"CodeFormer model not found at {model_path}")
                
                self.model = CodeFormer(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=['32', '64', '128', '256'],
                ).to(self.device)
                
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['params_ema'])
                self.model.eval()
                
            elif model_name_lower == 'gfpgan':
                from gfpgan import GFPGANer
                
                model_path = os.path.join('weights', 'gfpgan', 'GFPGANv1.4.pth')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"GFPGAN model not found at {model_path}")
                
                self.model = GFPGANer(
                    model_path=model_path,
                    upscale=1,
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=self.device
                )
                
            elif model_name_lower == 'none':
                # No face restoration model
                self.model = None
                
            else:
                raise ValueError(f"Unknown face restoration model: {model_name}")
                
            self.model_name = model_name
            logger.info(f"{model_name} model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load {model_name} model: {str(e)}")
            raise

    def restore_faces(self, image, model_name, code_former_weight=0.5, device='cuda', low_vram=False):
        """Restore faces in the given image"""
        try:
            self._load_model(model_name)
            
            # If no model is loaded, return original image
            if self.model is None:
                return image
            
            if model_name.lower() == 'codeformer':
                # Convert to PIL Image for CodeFormer
                if isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(image)
                else:
                    pil_image = image
                
                # CodeFormer expects PIL Image
                with torch.no_grad():
                    result = self.model.restore(pil_image, w=code_former_weight)
                    
                # Handle different return types
                if isinstance(result, tuple):
                    restored_image = result[0]
                    if len(result) > 1:
                        logger.debug(f"CodeFormer result type: {type(result)}, length: {len(result)}")
                else:
                    restored_image = result
                    
                # Ensure we have a numpy array
                if hasattr(restored_image, 'dtype'):
                    # Already a numpy array
                    pass
                elif isinstance(restored_image, list):
                    # Convert list to numpy array
                    restored_image = np.array(restored_image)
                else:
                    # Convert PIL Image to numpy array
                    restored_image = np.array(restored_image)
                    
            elif model_name.lower() == 'gfpgan':
                # GFPGAN expects numpy array
                if isinstance(image, Image.Image):
                    image = np.array(image)
                
                # GFPGAN processing
                _, _, restored_image = self.model.enhance(
                    image, 
                    has_aligned=False, 
                    only_center_face=False, 
                    paste_back=True
                )
                
                # Handle different return types from GFPGAN
                if isinstance(restored_image, tuple):
                    logger.debug(f"GFPGAN result type: {type(restored_image)}, length: {len(restored_image)}")
                    if len(restored_image) > 0:
                        restored_image = restored_image[0]
                    else:
                        logger.warning("GFPGAN returned empty tuple")
                        return image
                elif isinstance(restored_image, list):
                    logger.debug(f"GFPGAN result type: {type(restored_image)}, length: {len(restored_image)}")
                    if len(restored_image) > 0:
                        restored_image = restored_image[0]
                    else:
                        logger.warning("GFPGAN returned empty list")
                        return image
                
                # Ensure we have a numpy array
                if not hasattr(restored_image, 'dtype'):
                    if isinstance(restored_image, list):
                        restored_image = np.array(restored_image)
                    else:
                        restored_image = np.array(restored_image)
            
            # Clean up model if low_vram is enabled
            if low_vram:
                self.model = None
                torch.cuda.empty_cache()
                logger.info("Face restoration model cleaned up")
            
            return restored_image
            
        except Exception as e:
            logger.error(f"Face restoration failed: {str(e)}")
            # Return original image if restoration fails
            return image


def restore_faces(image, model_name, code_former_weight=0.5, device='cuda', low_vram=False):
    """Standalone function for face restoration"""
    face_restoration = FaceRestoration(device=device, low_vram=low_vram)
    return face_restoration.restore_faces(image, model_name, code_former_weight, device, low_vram)
