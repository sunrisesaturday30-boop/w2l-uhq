#!/usr/bin/env python3
"""
Local runner for Wav2Lip UHQ processor
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
    parser = argparse.ArgumentParser(description="Wav2Lip UHQ Local Runner")
    
    # Model management
    parser.add_argument("--download-models", action="store_true", help="Download required models")
    parser.add_argument("--check-models", action="store_true", help="Check if models exist")
    
    # Input files
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    parser.add_argument("--output", type=str, help="Path to output video file")
    
    # Wav2Lip parameters
    parser.add_argument("--checkpoint", type=str, default="wav2lip_gan", help="Wav2Lip checkpoint")
    parser.add_argument("--nosmooth", action="store_true", help="Disable face smoothing")
    parser.add_argument("--resize-factor", type=int, default=1, help="Resize factor for video")
    parser.add_argument("--pad-top", type=int, default=0, help="Padding top")
    parser.add_argument("--pad-bottom", type=int, default=10, help="Padding bottom")
    parser.add_argument("--pad-left", type=int, default=0, help="Padding left")
    parser.add_argument("--pad-right", type=int, default=0, help="Padding right")
    
    # Enhancement parameters
    parser.add_argument("--face-restore-model", type=str, default="GFPGAN", 
                       choices=["GFPGAN", "CodeFormer"], help="Face restoration model")
    parser.add_argument("--mouth-mask-dilatation", type=int, default=0, help="Mouth mask dilation")
    parser.add_argument("--erode-face-mask", type=int, default=0, help="Erode face mask")
    parser.add_argument("--mask-blur", type=int, default=0, help="Mask blur")
    parser.add_argument("--only-mouth", action="store_true", help="Only process mouth region")
    parser.add_argument("--code-former-weight", type=float, default=0.5, help="CodeFormer weight")
    
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
    
    # Validate inputs
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)
    
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Set output path
    if not args.output:
        video_path = Path(args.video)
        args.output = str(video_path.parent / f"{video_path.stem}_lipsync{video_path.suffix}")
    
    try:
        # Initialize processor
        logger.info("Initializing Wav2Lip processor...")
        processor = Wav2LipProcessor(device=args.device, low_vram=args.low_vram)
        
        # Process video
        logger.info("Starting video processing...")
        result_path = processor.process_video(
            video_path=args.video,
            audio_path=args.audio,
            checkpoint=args.checkpoint,
            nosmooth=args.nosmooth,
            resize_factor=args.resize_factor,
            pad_top=args.pad_top,
            pad_bottom=args.pad_bottom,
            pad_left=args.pad_left,
            pad_right=args.pad_right,
            face_swap_img=None,
            face_restore_model=args.face_restore_model,
            mouth_mask_dilatation=args.mouth_mask_dilatation,
            erode_face_mask=args.erode_face_mask,
            mask_blur=args.mask_blur,
            only_mouth=args.only_mouth,
            code_former_weight=args.code_former_weight,
            debug=args.debug
        )
        
        # Copy result to output path
        import shutil
        shutil.copy2(result_path, args.output)
        
        logger.info(f"Processing completed! Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up exception hook for full tracebacks
    sys.excepthook = lambda exc_type, exc_value, exc_traceback: (
        logging.error(f"Uncaught exception: {exc_type.__name__}: {exc_value}"),
        logging.error(f"Full traceback: {traceback.format_exc()}"),
        sys.exit(1)
    )
    
    main()
