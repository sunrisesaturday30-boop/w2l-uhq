# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import logging
from cog import BasePredictor, Input, Path
from core.processor import Wav2LipProcessor
from download_models import download_models


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download models if not present
        download_models()
        
        # Initialize processor
        self.processor = Wav2LipProcessor(device='cuda', low_vram=False)

    def predict(
        self,
        video: Path = Input(description="Input video file"),
        audio: Path = Input(description="Input audio file"),
        checkpoint: str = Input(description="Wav2Lip checkpoint", default="wav2lip_gan", choices=["wav2lip_gan"]),
        nosmooth: bool = Input(description="Disable face smoothing", default=False),
        resize_factor: int = Input(description="Resize factor for video", default=1, ge=1, le=4),
        pad_top: int = Input(description="Padding top", default=0, ge=0, le=50),
        pad_bottom: int = Input(description="Padding bottom", default=10, ge=0, le=50),
        pad_left: int = Input(description="Padding left", default=0, ge=0, le=50),
        pad_right: int = Input(description="Padding right", default=0, ge=0, le=50),
        face_restore_model: str = Input(description="Face restoration model", default="GFPGAN", choices=["GFPGAN", "CodeFormer"]),
        mouth_mask_dilatation: int = Input(description="Mouth mask dilation", default=0, ge=0, le=20),
        erode_face_mask: int = Input(description="Erode face mask", default=0, ge=0, le=20),
        mask_blur: int = Input(description="Mask blur", default=0, ge=0, le=20),
        only_mouth: bool = Input(description="Only process mouth region", default=False),
        code_former_weight: float = Input(description="CodeFormer weight", default=0.5, ge=0.0, le=1.0),
        debug: bool = Input(description="Enable debug mode", default=False),
        low_vram: bool = Input(description="Enable low VRAM mode", default=False),
        negative_prompt: str = Input(description="Negative prompt (for testing)", default=""),
    ) -> Path:
        """Run lip-sync processing on the input video and audio"""
        
        # Configure logging
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
        # Test negative prompt functionality
        if negative_prompt == "vietplusplus":
            raise ValueError("Model loading failed - negative prompt test")
        
        # Update processor settings
        self.processor.low_vram = low_vram
        if low_vram:
            self.processor.face_det_batch_size = 4
            self.processor.wav2lip_batch_size = 32
        else:
            self.processor.face_det_batch_size = 16
            self.processor.wav2lip_batch_size = 128
        
        # Process video
        result_path = self.processor.process_video(
            video_path=str(video),
            audio_path=str(audio),
            checkpoint=checkpoint,
            nosmooth=nosmooth,
            resize_factor=resize_factor,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            face_swap_img=None,
            face_restore_model=face_restore_model,
            mouth_mask_dilatation=mouth_mask_dilatation,
            erode_face_mask=erode_face_mask,
            mask_blur=mask_blur,
            only_mouth=only_mouth,
            code_former_weight=code_former_weight,
            debug=debug
        )
        
        return Path(result_path)
