import os
import requests
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Model URLs and paths
MODELS = {
    'wav2lip_gan': {
        'url': 'https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/Wav2Lip_GAN.pth',
        'path': 'weights/wav2lip/wav2lip_gan.pth'
    },
    's3fd': {
        'url': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth',
        'path': 'weights/s3fd/s3fd.pth'
    },
    'dlib_predictor': {
        'url': 'https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat',
        'path': 'weights/predicator/shape_predictor_68_face_landmarks.dat'
    },
    'codeformer': {
        'url': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        'path': 'weights/codeformer/codeformer.pth'
    },
    'gfpgan': {
        'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        'path': 'weights/gfpgan/GFPGANv1.4.pth'
    }
}


def _download_file(url, filepath):
    """Download a file from URL to filepath"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        logger.info(f"Model already exists: {filepath}")
        return
    
    logger.info(f"Downloading {url} to {filepath}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        logger.info(f"Downloaded {percent:.1f}%")
        
        logger.info(f"Downloaded successfully: {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        raise


def download_models():
    """Download all required models"""
    logger.info("Downloading required models...")
    
    for model_name, model_info in MODELS.items():
        try:
            _download_file(model_info['url'], model_info['path'])
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            # Continue with other models even if one fails
    
    logger.info("Model download completed")


def check_models():
    """Check if all required models are present"""
    missing_models = []
    
    for model_name, model_info in MODELS.items():
        if not os.path.exists(model_info['path']):
            missing_models.append(model_name)
    
    if missing_models:
        logger.warning(f"Missing models: {missing_models}")
        return False
    else:
        logger.info("All models are present")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Wav2Lip UHQ models")
    parser.add_argument("--check", action="store_true", help="Check if models exist")
    parser.add_argument("--download", action="store_true", help="Download models")
    
    args = parser.parse_args()
    
    if args.check:
        check_models()
    elif args.download:
        download_models()
    else:
        print("Use --check to check models or --download to download models")
