#!/usr/bin/env python3
"""
Local test script for Wav2Lip UHQ processor
Run this script to test the lip-sync functionality locally
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add core module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from core.processor import Wav2LipProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_models():
    """Download models if they don't exist (same as in predict.py)"""
    import requests
    
    def download_file(url: str, filepath: str, description: str = "file") -> bool:
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
        if download_file(info["url"], info["path"], info["description"]):
            success_count += 1
    
    if success_count == len(missing_models):
        logger.info("All models downloaded successfully!")
        return True
    else:
        logger.warning(f"Only {success_count}/{len(missing_models)} models downloaded successfully")
        return False

def main():
    parser = argparse.ArgumentParser(description="Wav2Lip UHQ Local Test Script")
    
    # Required arguments
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("audio", type=str, help="Path to input audio file")
    parser.add_argument("output", type=str, help="Path to output video file")
    
    # Optional arguments with defaults
    parser.add_argument("--checkpoint", type=str, default="wav2lip_gan", 
                       choices=["wav2lip", "wav2lip_gan"],
                       help="Wav2Lip model checkpoint")
    parser.add_argument("--face-restore-model", type=str, default="GFPGAN",
                       choices=["CodeFormer", "GFPGAN"],
                       help="Face restoration model")
    parser.add_argument("--no-smooth", action="store_true",
                       help="Disable face detection smoothing")
    parser.add_argument("--only-mouth", action="store_true",
                       help="Track only mouth area (faster processing)")
    parser.add_argument("--resize-factor", type=int, default=1,
                       choices=[1, 2, 3, 4],
                       help="Video downscaling factor")
    parser.add_argument("--pad-top", type=int, default=0,
                       help="Padding above lips (pixels)")
    parser.add_argument("--pad-bottom", type=int, default=0,
                       help="Padding below lips (pixels)")
    parser.add_argument("--pad-left", type=int, default=0,
                       help="Padding left of lips (pixels)")
    parser.add_argument("--pad-right", type=int, default=0,
                       help="Padding right of lips (pixels)")
    parser.add_argument("--mouth-mask-dilate", type=int, default=15,
                       help="Mouth mask dilation (pixels)")
    parser.add_argument("--face-mask-erode", type=int, default=15,
                       help="Face mask erosion (pixels)")
    parser.add_argument("--mask-blur", type=int, default=15,
                       help="Mask blur kernel size (pixels)")
    parser.add_argument("--code-former-fidelity", type=float, default=0.75,
                       help="CodeFormer fidelity (0.0-1.0)")
    parser.add_argument("--low-vram", action="store_true",
                       help="Enable low VRAM mode (slower but uses less memory)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--download-models", action="store_true",
                       help="Download models if missing")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Download models if requested
        if args.download_models:
            logger.info("Downloading models...")
            download_models()
        
        # Validate input files
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        
        if not os.path.exists(args.audio):
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        
        logger.info("Input validation passed")
        logger.info(f"Video: {args.video}")
        logger.info(f"Audio: {args.audio}")
        logger.info(f"Output: {args.output}")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Face restore model: {args.face_restore_model}")
        logger.info(f"Resize factor: {args.resize_factor}")
        logger.info(f"Low VRAM: {args.low_vram}")
        
        # Initialize processor
        logger.info("Initializing Wav2Lip UHQ processor...")
        processor = Wav2LipProcessor(
            weights_dir="weights",
            device="cuda" if os.system("nvidia-smi") == 0 else "cpu",
            low_vram=args.low_vram,
            debug=args.debug
        )
        
        # Process video
        logger.info("Starting video processing...")
        result_path = processor.process_video(
            video_path=args.video,
            audio_path=args.audio,
            checkpoint=args.checkpoint,
            face_restore_model=args.face_restore_model,
            no_smooth=args.no_smooth,
            only_mouth=args.only_mouth,
            resize_factor=args.resize_factor,
            pad_top=args.pad_top,
            pad_bottom=args.pad_bottom,
            pad_left=args.pad_left,
            pad_right=args.pad_right,
            mouth_mask_dilate=args.mouth_mask_dilate,
            face_mask_erode=args.face_mask_erode,
            mask_blur=args.mask_blur,
            code_former_fidelity=args.code_former_fidelity
        )
        
        # Copy result to output path
        import shutil
        shutil.copy2(result_path, args.output)
        
        logger.info(f"Processing completed successfully!")
        logger.info(f"Output saved to: {args.output}")
        
        # Cleanup
        processor.cleanup()
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
