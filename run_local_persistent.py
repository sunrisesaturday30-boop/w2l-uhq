#!/usr/bin/env python3
"""
Persistent local runner for Wav2Lip UHQ processor
Loads models once and allows multiple predictions
"""

import os
import sys
import argparse
import logging
import traceback
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.processor import Wav2LipProcessor
from download_models import download_models, check_models


def setup_logging(debug=False):
    """Setup logging configuration"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description="Wav2Lip UHQ Persistent Local Runner")
    
    # Model management
    parser.add_argument("--download-models", action="store_true", help="Download required models")
    parser.add_argument("--check-models", action="store_true", help="Check if models exist")
    
    # System parameters
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--low-vram", action="store_true", help="Enable low VRAM mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Handle model management
    if args.download_models:
        logger.info("Downloading models...")
        download_models()
        return
    
    if args.check_models:
        logger.info("Checking models...")
        if check_models():
            logger.info("All models are present")
            sys.exit(0)
        else:
            logger.error("Some models are missing")
            sys.exit(1)
    
    # Initialize processor once
    logger.info("Initializing Wav2Lip processor...")
    processor = Wav2LipProcessor(device=args.device, low_vram=args.low_vram)
    
    logger.info("Processor initialized. Ready for predictions!")
    logger.info("Enter video and audio paths, or 'quit' to exit.")
    
    # Interactive loop
    while True:
        try:
            # Get input from user
            video_path = input("\nEnter video path (or 'quit'): ").strip()
            if video_path.lower() == 'quit':
                break
            
            audio_path = input("Enter audio path: ").strip()
            output_path = input("Enter output path (optional): ").strip()
            
            # Validate inputs
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                continue
            
            if not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                continue
            
            # Set default output path
            if not output_path:
                video_path_obj = Path(video_path)
                output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_lipsync{video_path_obj.suffix}")
            
            # Process video
            logger.info("Starting video processing...")
            result_path = processor.process_video(
                video_path=video_path,
                audio_path=audio_path,
                checkpoint="wav2lip_gan",
                nosmooth=False,
                resize_factor=1,
                pad_top=0,
                pad_bottom=10,
                pad_left=0,
                pad_right=0,
                face_swap_img=None,
                face_restore_model="GFPGAN",
                mouth_mask_dilatation=0,
                erode_face_mask=0,
                mask_blur=0,
                only_mouth=False,
                code_former_weight=0.5,
                debug=args.debug
            )
            
            # Copy result to output path
            import shutil
            shutil.copy2(result_path, output_path)
            
            logger.info(f"Processing completed! Output saved to: {output_path}")
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            break
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            continue
    
    logger.info("Exiting...")


if __name__ == "__main__":
    # Set up exception hook for full tracebacks
    sys.excepthook = lambda exc_type, exc_value, exc_traceback: (
        logging.error(f"Uncaught exception: {exc_type.__name__}: {exc_value}"),
        logging.error(f"Full traceback: {traceback.format_exc()}"),
        sys.exit(1)
    )
    
    main()
