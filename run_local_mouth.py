#!/usr/bin/env python3
"""
Local runner for Wav2Lip UHQ with mouth-only detection.
This version skips full face detection and directly processes mouth regions.
"""

import os
import sys
import argparse
import logging
import tempfile
import traceback
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.processor import Wav2LipProcessor
from download_models import download_models, check_models

def setup_logging(debug=False):
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    parser = argparse.ArgumentParser(description='Wav2Lip UHQ - Mouth-Only Detection')
    
    # Input files
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--audio', required=True, help='Path to input audio file')
    
    # Model management
    parser.add_argument('--download-models', action='store_true', 
                       help='Download all required models')
    parser.add_argument('--check-models', action='store_true', 
                       help='Check if all models are available')
    
    # Processing options
    parser.add_argument('--face-restore', choices=['codeformer', 'gfpgan', 'none'], 
                       default='codeformer', help='Face restoration method')
    parser.add_argument('--only-mouth', action='store_true', default=True,
                       help='Process only mouth region (always enabled for mouth-only mode)')
    
    # Masking and padding
    parser.add_argument('--mask-dilate', type=int, default=5, 
                       help='Mask dilation radius (default: 5)')
    parser.add_argument('--mask-blur', type=int, default=5, 
                       help='Mask blur radius (default: 5)')
    parser.add_argument('--pad-top', type=int, default=0, 
                       help='Top padding (default: 0)')
    parser.add_argument('--pad-bottom', type=int, default=10, 
                       help='Bottom padding (default: 10)')
    parser.add_argument('--pad-left', type=int, default=0, 
                       help='Left padding (default: 0)')
    parser.add_argument('--pad-right', type=int, default=0, 
                       help='Right padding (default: 0)')
    
    # Performance options
    parser.add_argument('--low-vram', action='store_true', 
                       help='Enable low VRAM mode for smaller GPUs')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug logging')
    
    # Output options
    parser.add_argument('--output', help='Output video path (optional)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Handle model management
    if args.download_models:
        logger.info("Downloading models...")
        download_models()
        logger.info("Model download completed!")
        return
    
    if args.check_models:
        logger.info("Checking models...")
        missing_models = check_models()
        if missing_models:
            logger.warning(f"Missing models: {missing_models}")
            logger.info("Run with --download-models to download missing models")
        else:
            logger.info("All models are available!")
        return
    
    # Validate input files
    if not os.path.exists(args.video):
        logger.error(f"Video file not found: {args.video}")
        return 1
    
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        return 1
    
    # Set up exception handling for full tracebacks
    def excepthook(exc_type, exc_value, exc_traceback):
        logger.error(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
        logger.error("Full traceback:")
        for line in traceback.format_tb(exc_traceback):
            logger.error(line.rstrip())
        sys.exit(1)
    
    sys.excepthook = excepthook
    
    try:
        # Initialize processor with mouth-only mode
        logger.info("Initializing Wav2Lip processor with mouth-only detection...")
        processor = Wav2LipProcessor(
            low_vram=args.low_vram,
            mouth_only_mode=True  # Enable mouth-only detection
        )
        
        # Process video with mouth-only detection
        logger.info("Starting mouth-only lip-sync processing...")
        result_path = processor.process_video(
            video_path=args.video,
            audio_path=args.audio,
            face_restore_model=args.face_restore,
            only_mouth=True,  # Always True for mouth-only mode
            mouth_mask_dilatation=args.mask_dilate,
            erode_face_mask=args.mask_blur,  # Map mask_blur to erode_face_mask
            pad_top=args.pad_top,
            pad_bottom=args.pad_bottom,
            pad_left=args.pad_left,
            pad_right=args.pad_right
        )
        
        # Handle output
        if args.output:
            import shutil
            shutil.copy2(result_path, args.output)
            logger.info(f"Output saved to: {args.output}")
        else:
            logger.info(f"Processing completed: {result_path}")
            
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.error("Full traceback:")
        for line in traceback.format_tb(e.__traceback__):
            logger.error(line.rstrip())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
