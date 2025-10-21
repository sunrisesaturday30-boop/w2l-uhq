#!/usr/bin/env python3
"""
Model download utility for Wav2Lip UHQ
Downloads required models if they don't exist in the weights folder
"""

import os
import sys
import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def download_models(weights_dir: str = "weights") -> bool:
    """Download required models if they don't exist"""
    models_to_download = {
        # Wav2Lip models
        "wav2lip_gan.pth": {
            "url": "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip_GAN.pth",
            "path": os.path.join(weights_dir, "wav2lip", "wav2lip_gan.pth"),
            "description": "Wav2Lip GAN model"
        },
        "wav2lip.pth": {
            "url": "https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip.pth",
            "path": os.path.join(weights_dir, "wav2lip", "wav2lip.pth"), 
            "description": "Wav2Lip model"
        },
        # S3FD face detection model
        "s3fd-619a316812.pth": {
            "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
            "path": os.path.join(weights_dir, "s3fd", "s3fd-619a316812.pth"),
            "description": "S3FD face detection model"
        },
        # Dlib landmark predictor
        "shape_predictor_68_face_landmarks.dat": {
            "url": "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat",
            "path": os.path.join(weights_dir, "predicator", "shape_predictor_68_face_landmarks.dat"),
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

def check_models(weights_dir: str = "weights") -> bool:
    """Check if all required models exist"""
    required_files = [
        os.path.join(weights_dir, "wav2lip", "wav2lip_gan.pth"),
        os.path.join(weights_dir, "wav2lip", "wav2lip.pth"),
        os.path.join(weights_dir, "s3fd", "s3fd-619a316812.pth"),
        os.path.join(weights_dir, "predicator", "shape_predictor_68_face_landmarks.dat")
    ]
    
    missing_files = []
    for filepath in required_files:
        if not os.path.exists(filepath):
            missing_files.append(filepath)
    
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} model files:")
        for filepath in missing_files:
            logger.warning(f"  - {filepath}")
        return False
    else:
        logger.info("All required models found!")
        return True

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Wav2Lip UHQ models")
    parser.add_argument("--weights-dir", type=str, default="weights",
                       help="Directory to store model weights")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check if models exist, don't download")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.check_only:
        success = check_models(args.weights_dir)
    else:
        success = download_models(args.weights_dir)
    
    if success:
        logger.info("Model check/download completed successfully!")
        sys.exit(0)
    else:
        logger.error("Model check/download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
