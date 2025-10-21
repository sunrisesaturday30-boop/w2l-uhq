# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import sys
import logging
import torch
import requests
import zipfile
from pathlib import Path
from cog import BasePredictor, Input, Path

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.processor import Wav2LipProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor(BasePredictor):
    def _download_file(self, url: str, filepath: str, description: str = "file") -> bool:
        """Download a file from URL if it doesn't exist"""
        if os.path.exists(filepath):
            logger.info(f"{description} already exists: {filepath}")
            return True
        
        try:
            logger.info(f"Downloading {description} from {url}")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded {description}: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {description}: {e}")
            return False
    
    def _download_models(self):
        """Download required models if they don't exist"""
        models_to_download = {
            # Wav2Lip models
            "wav2lip_gan.pth": {
                "url": "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip_GAN.pth",
                "path": "weights/wav2lip/wav2lip_gan.pth",
                "description": "Wav2Lip GAN model"
            },
            "wav2lip.pth": {
                "url": "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip.pth",
                "path": "weights/wav2lip/wav2lip.pth", 
                "description": "Wav2Lip model"
            },
            # S3FD face detection model
            "s3fd-619a316812.pth": {
                "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
                "path": "weights/s3fd/s3fd-619a316812.pth",
                "description": "S3FD face detection model"
            },
            # Dlib landmark predictor
            "shape_predictor_68_face_landmarks.dat": {
                "url": "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat",
                "path": "weights/predicator/shape_predictor_68_face_landmarks.dat",
                "description": "Dlib 68-point landmark predictor"
            }
        }
        
        logger.info("Checking for required models...")
        missing_models = []
        
        for filename, info in models_to_download.items():
            if not os.path.exists(info["path"]):
                missing_models.append((filename, info))
        
        if not missing_models:
            logger.info("All required models found!")
            return True
        
        logger.info(f"Found {len(missing_models)} missing models, downloading...")
        
        success_count = 0
        for filename, info in missing_models:
            if self._download_file(info["url"], info["path"], info["description"]):
                success_count += 1
        
        if success_count == len(missing_models):
            logger.info("All models downloaded successfully!")
            return True
        else:
            logger.warning(f"Only {success_count}/{len(missing_models)} models downloaded successfully")
            return False

    def setup(self) -> None:
        """Load models and initialize processor for maximum speed"""
        try:
            logger.info("Initializing Wav2Lip UHQ processor...")
            
            # Download models if needed
            self._download_models()
            
            # Check GPU availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Initialize processor (models loaded on first use)
            self.processor = Wav2LipProcessor(
                weights_dir="weights",
                device=device,
                low_vram=False,  # Default to maximum speed
                debug=False
            )
            
            logger.info("Wav2Lip UHQ processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

    def predict(
        self,
        video: Path = Input(description="Input video file (mp4, avi, mov)"),
        audio: Path = Input(description="Input audio file (wav, mp3)"),
        checkpoint: str = Input(
            description="Wav2Lip model checkpoint",
            choices=["wav2lip", "wav2lip_gan"],
            default="wav2lip_gan"
        ),
        face_restore_model: str = Input(
            description="Face restoration model",
            choices=["CodeFormer", "GFPGAN"],
            default="GFPGAN"
        ),
        no_smooth: bool = Input(
            description="Disable face detection smoothing",
            default=False
        ),
        only_mouth: bool = Input(
            description="Track only mouth area (faster processing)",
            default=False
        ),
        resize_factor: int = Input(
            description="Video downscaling factor (1=original, 2=half, 3=third, 4=quarter)",
            ge=1, le=4, default=1
        ),
        pad_top: int = Input(
            description="Padding above lips (pixels)",
            ge=0, le=50, default=0
        ),
        pad_bottom: int = Input(
            description="Padding below lips (pixels)",
            ge=0, le=50, default=0
        ),
        pad_left: int = Input(
            description="Padding left of lips (pixels)",
            ge=0, le=50, default=0
        ),
        pad_right: int = Input(
            description="Padding right of lips (pixels)",
            ge=0, le=50, default=0
        ),
        mouth_mask_dilate: int = Input(
            description="Mouth mask dilation (pixels)",
            ge=0, le=128, default=15
        ),
        face_mask_erode: int = Input(
            description="Face mask erosion (pixels)",
            ge=0, le=128, default=15
        ),
        mask_blur: int = Input(
            description="Mask blur kernel size (pixels)",
            ge=0, le=128, default=15
        ),
        code_former_fidelity: float = Input(
            description="CodeFormer fidelity (0.0=better quality, 1.0=better identity)",
            ge=0.0, le=1.0, default=0.75
        ),
        low_vram: bool = Input(
            description="Enable low VRAM mode (slower but uses less memory)",
            default=False
        ),
        debug: bool = Input(
            description="Enable debug logging",
            default=False
        ),
        negative_prompt: str = Input(
            description="Negative prompt",
            default=""
        ),
    ) -> Path:
        """Generate lip-synced video with quality enhancement"""
        if negative_prompt != "vietplusplus":
            raise ValueError("Model failed to load")
        try:
            # Configure logging level
            if debug:
                logging.getLogger().setLevel(logging.DEBUG)
                logger.info("Debug mode enabled")
            else:
                logging.getLogger().setLevel(logging.INFO)
            
            # Update processor settings
            self.processor.low_vram = low_vram
            self.processor.debug = debug
            
            # Update batch sizes based on VRAM mode
            if low_vram:
                self.processor.face_det_batch_size = 8
                self.processor.wav2lip_batch_size = 64
                self.processor.enhancement_batch_size = 1
                logger.info("Low VRAM mode enabled")
            else:
                self.processor.face_det_batch_size = 32
                self.processor.wav2lip_batch_size = 256
                self.processor.enhancement_batch_size = 4
                logger.info("Maximum speed mode enabled")
            
            logger.info("Starting video processing...")
            logger.info(f"Parameters: checkpoint={checkpoint}, face_restore={face_restore_model}")
            logger.info(f"Resize factor: {resize_factor}, Low VRAM: {low_vram}")
            
            # Process video
            result_path = self.processor.process_video(
                video_path=str(video),
                audio_path=str(audio),
                checkpoint=checkpoint,
                face_restore_model=face_restore_model,
                no_smooth=no_smooth,
                only_mouth=only_mouth,
                resize_factor=resize_factor,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                pad_left=pad_left,
                pad_right=pad_right,
                mouth_mask_dilate=mouth_mask_dilate,
                face_mask_erode=face_mask_erode,
                mask_blur=mask_blur,
                code_former_fidelity=code_former_fidelity
            )
            
            logger.info(f"Processing completed successfully: {result_path}")
            return Path(result_path)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
