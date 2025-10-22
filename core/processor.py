import os
import tempfile
import shutil
import logging
import torch
import cv2
import numpy as np

from .face_restoration import FaceRestoration

# Import wav2lip modules
try:
    from wav2lip.w2l import W2l
    from wav2lip.wav2lip_uhq import Wav2LipUHQ
except ImportError:
    # Fallback for different import contexts
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from wav2lip.w2l import W2l
    from wav2lip.wav2lip_uhq import Wav2LipUHQ

logger = logging.getLogger(__name__)


class Wav2LipProcessor:
    def __init__(self, device='cuda', low_vram=False, mouth_only_mode=False):
        self.device = device
        self.low_vram = low_vram
        self.mouth_only_mode = mouth_only_mode
        self.temp_dir = None
        self.face_restoration = None
        
        # Dynamic batch sizes based on low_vram setting
        if low_vram:
            self.face_det_batch_size = 4
            self.wav2lip_batch_size = 32
        else:
            self.face_det_batch_size = 16
            self.wav2lip_batch_size = 128

    def _setup_temp_dir(self):
        """Create a temporary directory for processing"""
        self.temp_dir = tempfile.mkdtemp(prefix='wav2lip_')
        logger.info(f"Created temp directory: {self.temp_dir}")
        return self.temp_dir

    def _cleanup_temp_dir(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            self.temp_dir = None

    def _validate_inputs(self, video_path, audio_path):
        """Validate input files"""
        if not os.path.isfile(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        
        if not os.path.isfile(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        # Check video format
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        cap.release()
        
        # Check audio format
        try:
            import librosa
            librosa.load(audio_path, sr=16000)
        except Exception as e:
            raise ValueError(f"Cannot load audio file: {audio_path}, error: {str(e)}")

    def _setup_face_restoration(self, face_restore_model, code_former_weight):
        """Setup face restoration model"""
        if not self.face_restoration:
            self.face_restoration = FaceRestoration(device=self.device, low_vram=self.low_vram)
        
        # Pre-load the model
        self.face_restoration._load_model(face_restore_model)
        return self.face_restoration

    def _run_wav2lip(self, video_path, audio_path, checkpoint, nosmooth, resize_factor, 
                     pad_top, pad_bottom, pad_left, pad_right, face_swap_img):
        """Run the Wav2Lip processing"""
        logger.info("Starting Wav2Lip processing...")
        
        w2l = W2l(
            face=video_path,
            audio=audio_path,
            checkpoint=checkpoint,
            nosmooth=nosmooth,
            resize_factor=resize_factor,
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            face_swap_img=face_swap_img,
            temp_dir=self.temp_dir
        )
        
        # Update batch sizes
        w2l.face_det_batch_size = self.face_det_batch_size
        w2l.wav2lip_batch_size = self.wav2lip_batch_size
        
        try:
            w2l.execute()
            result_path = w2l.outfile
            logger.info(f"Wav2Lip processing completed: {result_path}")
            return result_path
        except Exception as e:
            logger.error(f"Wav2Lip processing failed: {str(e)}")
            raise

    def _run_enhancement(self, original_video, wav2lip_video, face_restore_model, 
                        mouth_mask_dilatation, erode_face_mask, mask_blur, only_mouth,
                        face_swap_img, resize_factor, code_former_weight, debug):
        """Run the quality enhancement"""
        logger.info("Starting quality enhancement...")
        
        w2luhq = Wav2LipUHQ(
            face=original_video,
            face_restore_model=face_restore_model,
            mouth_mask_dilatation=mouth_mask_dilatation,
            erode_face_mask=erode_face_mask,
            mask_blur=mask_blur,
            only_mouth=only_mouth,
            face_swap_img=face_swap_img,
            resize_factor=resize_factor,
            code_former_weight=code_former_weight,
            debug=debug,
            temp_dir=self.temp_dir
        )
        
        try:
            result_paths = w2luhq.execute()
            if result_paths and len(result_paths) > 3:
                final_path = result_paths[3]  # output_video.mp4
                logger.info(f"Quality enhancement completed: {final_path}")
                return final_path
            else:
                raise ValueError("Quality enhancement did not produce expected output")
        except Exception as e:
            logger.error(f"Quality enhancement failed: {str(e)}")
            raise

    def _extract_mouth_regions(self, video_path):
        """Extract mouth regions directly using dlib landmarks"""
        import dlib
        from wav2lip.face_detection.api import get_detections_for_batch
        
        logger.info("Extracting mouth regions using facial landmarks...")
        
        # Load dlib predictor
        predictor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'weights', 'predicator', 'shape_predictor_68_face_landmarks.dat')
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"dlib predictor not found: {predictor_path}")
        
        predictor = dlib.shape_predictor(predictor_path)
        
        # Read video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        logger.info(f"Processing {len(frames)} frames for mouth detection...")
        
        # Process frames in batches for face detection
        mouth_crops = []
        mouth_bboxes = []
        
        for i in range(0, len(frames), self.face_det_batch_size):
            batch_frames = frames[i:i + self.face_det_batch_size]
            batch_detections = get_detections_for_batch(batch_frames)
            
            for j, (frame, detections) in enumerate(zip(batch_frames, batch_detections)):
                if len(detections) == 0:
                    logger.warning(f"No face detected in frame {i + j}")
                    mouth_crops.append(None)
                    mouth_bboxes.append(None)
                    continue
                
                # Use the first (largest) face detection
                face_bbox = detections[0]
                x, y, w, h = face_bbox
                
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                # Get facial landmarks
                landmarks = predictor(face_region, dlib.rectangle(0, 0, w, h))
                landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
                
                # Extract mouth points (48-67 in 68-point model)
                if len(landmarks) >= 68:
                    mouth_points = landmarks[48:68]  # 20 points around the mouth
                    
                    # Create bounding box around mouth with padding
                    mouth_x = int(np.min(mouth_points[:, 0]))
                    mouth_y = int(np.min(mouth_points[:, 1]))
                    mouth_w = int(np.max(mouth_points[:, 0]) - mouth_x)
                    mouth_h = int(np.max(mouth_points[:, 1]) - mouth_y)
                    
                    # Add padding
                    padding = 20
                    mouth_x = max(0, mouth_x - padding)
                    mouth_y = max(0, mouth_y - padding)
                    mouth_w = min(face_region.shape[1] - mouth_x, mouth_w + 2 * padding)
                    mouth_h = min(face_region.shape[0] - mouth_y, mouth_h + 2 * padding)
                    
                    # Crop mouth region
                    mouth_crop = face_region[mouth_y:mouth_y+mouth_h, mouth_x:mouth_x+mouth_w]
                    
                    # Store global coordinates for later reconstruction
                    global_mouth_x = x + mouth_x
                    global_mouth_y = y + mouth_y
                    mouth_bboxes.append((global_mouth_x, global_mouth_y, mouth_w, mouth_h))
                    mouth_crops.append(mouth_crop)
                else:
                    logger.warning(f"Insufficient landmarks in frame {i + j}: {len(landmarks)}")
                    mouth_crops.append(None)
                    mouth_bboxes.append(None)
        
        logger.info(f"Extracted {len([c for c in mouth_crops if c is not None])} mouth regions")
        return mouth_crops, mouth_bboxes, frames

    def _run_mouth_only_processing(self, video_path, audio_path, face_restore_model, 
                                 mouth_mask_dilatation, mask_blur, code_former_weight, debug):
        """Run mouth-only processing pipeline"""
        logger.info("Starting mouth-only processing pipeline...")
        
        # Extract mouth regions
        mouth_crops, mouth_bboxes, original_frames = self._extract_mouth_regions(video_path)
        
        # Create mouth-only video from crops
        mouth_video_path = os.path.join(self.temp_dir, 'mouth_only_video.mp4')
        self._create_mouth_video(mouth_crops, mouth_bboxes, mouth_video_path)
        
        # Process mouth-only video with Wav2Lip (no resize factor needed)
        logger.info("Processing mouth regions with Wav2Lip...")
        w2l = W2l(
            face=mouth_video_path,
            audio=audio_path,
            checkpoint='wav2lip_gan',
            nosmooth=False,
            resize_factor=1,  # No resize needed for mouth crops
            pad_top=0,
            pad_bottom=0,
            pad_left=0,
            pad_right=0,
            face_swap_img=None,
            temp_dir=self.temp_dir
        )
        
        w2l.face_det_batch_size = self.face_det_batch_size
        w2l.wav2lip_batch_size = self.wav2lip_batch_size
        
        try:
            w2l.execute()
            wav2lip_result = w2l.outfile
            logger.info(f"Mouth-only Wav2Lip processing completed: {wav2lip_result}")
        except Exception as e:
            logger.error(f"Mouth-only Wav2Lip processing failed: {str(e)}")
            raise
        
        # Enhance mouth regions
        logger.info("Enhancing mouth regions...")
        enhanced_mouth_crops = self._enhance_mouth_crops(mouth_crops, face_restore_model, 
                                                       code_former_weight)
        
        # Reconstruct final video
        final_video = self._reconstruct_video(original_frames, enhanced_mouth_crops, 
                                            mouth_bboxes, video_path)
        
        return final_video

    def _create_mouth_video(self, mouth_crops, mouth_bboxes, output_path):
        """Create a video from mouth crops"""
        if not mouth_crops or all(crop is None for crop in mouth_crops):
            raise ValueError("No valid mouth crops to create video")
        
        # Find dimensions of the largest mouth crop
        valid_crops = [crop for crop in mouth_crops if crop is not None]
        if not valid_crops:
            raise ValueError("No valid mouth crops found")
        
        # Use the first valid crop dimensions
        h, w = valid_crops[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 25.0, (w, h))
        
        for crop in mouth_crops:
            if crop is not None:
                # Resize crop to standard dimensions
                resized_crop = cv2.resize(crop, (w, h))
                out.write(resized_crop)
            else:
                # Write black frame for missing crops
                black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                out.write(black_frame)
        
        out.release()
        logger.info(f"Created mouth-only video: {output_path}")

    def _enhance_mouth_crops(self, mouth_crops, face_restore_model, code_former_weight):
        """Enhance mouth crops using face restoration"""
        if not self.face_restoration:
            self._setup_face_restoration(face_restore_model, code_former_weight)
        
        enhanced_crops = []
        for i, crop in enumerate(mouth_crops):
            if crop is not None:
                try:
                    # Enhance the mouth crop
                    enhanced = self.face_restoration.restore_faces(
                        crop, device=self.device, low_vram=self.low_vram
                    )
                    enhanced_crops.append(enhanced)
                except Exception as e:
                    logger.warning(f"Failed to enhance mouth crop {i}: {str(e)}")
                    enhanced_crops.append(crop)
            else:
                enhanced_crops.append(None)
        
        return enhanced_crops

    def _reconstruct_video(self, original_frames, enhanced_mouth_crops, mouth_bboxes, 
                         original_video_path):
        """Reconstruct final video by pasting enhanced mouth regions back"""
        logger.info("Reconstructing final video...")
        
        # Get video properties
        cap = cv2.VideoCapture(original_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create output video
        output_path = os.path.join(self.temp_dir, 'final_output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame, enhanced_crop, bbox in zip(original_frames, enhanced_mouth_crops, mouth_bboxes):
            if enhanced_crop is not None and bbox is not None:
                x, y, w, h = bbox
                # Resize enhanced crop to fit the bbox
                resized_crop = cv2.resize(enhanced_crop, (w, h))
                # Paste back to original frame
                frame[y:y+h, x:x+w] = resized_crop
            
            out.write(frame)
        
        out.release()
        
        # Add original audio
        final_path = os.path.join(self.temp_dir, 'final_with_audio.mp4')
        self._add_audio_to_video(output_path, original_video_path, final_path)
        
        return final_path

    def _add_audio_to_video(self, video_path, original_video_path, output_path):
        """Add audio from original video to the enhanced video"""
        import subprocess
        
        # Extract audio from original video
        audio_path = os.path.join(self.temp_dir, 'original_audio.wav')
        subprocess.run([
            'ffmpeg', '-i', original_video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', '-y', audio_path
        ], check=True, capture_output=True)
        
        # Add audio to enhanced video
        subprocess.run([
            'ffmpeg', '-i', video_path, '-i', audio_path, '-c:v', 'copy', 
            '-c:a', 'aac', '-strict', 'experimental', '-y', output_path
        ], check=True, capture_output=True)

    def process_video(self, video_path, audio_path, checkpoint='wav2lip_gan', 
                     nosmooth=False, resize_factor=1, pad_top=0, pad_bottom=10, 
                     pad_left=0, pad_right=0, face_swap_img=None, 
                     face_restore_model='GFPGAN', mouth_mask_dilatation=0, 
                     erode_face_mask=0, mask_blur=0, only_mouth=False, 
                     code_former_weight=0.5, debug=False):
        """Main processing function"""
        
        try:
            # Setup
            self._setup_temp_dir()
            self._validate_inputs(video_path, audio_path)
            self._setup_face_restoration(face_restore_model, code_former_weight)
            
            # Choose processing mode
            if self.mouth_only_mode:
                logger.info("Using mouth-only processing mode")
                final_video = self._run_mouth_only_processing(
                    video_path, audio_path, face_restore_model,
                    mouth_mask_dilatation, mask_blur, code_former_weight, debug
                )
            else:
                logger.info("Using standard processing mode")
                # Run Wav2Lip
                wav2lip_result = self._run_wav2lip(
                    video_path, audio_path, checkpoint, nosmooth, resize_factor,
                    pad_top, pad_bottom, pad_left, pad_right, face_swap_img
                )
                
                # Run enhancement
                final_video = self._run_enhancement(
                    video_path, wav2lip_result, face_restore_model,
                    mouth_mask_dilatation, erode_face_mask, mask_blur, only_mouth,
                    face_swap_img, resize_factor, code_former_weight, debug
                )
            
            return final_video
            
        except Exception as e:
            logger.error(f"Video processing failed: {str(e)}")
            raise
        finally:
            # Cleanup
            if self.face_restoration:
                self.face_restoration.model = None
            torch.cuda.empty_cache()
            self._cleanup_temp_dir()
