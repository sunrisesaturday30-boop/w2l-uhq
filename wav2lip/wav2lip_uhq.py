import os
import numpy as np
import cv2
import dlib
import json
import torch
import logging

# Use relative imports with fallback
try:
    from . import face_detection
except ImportError:
    import face_detection

from imutils import face_utils
import subprocess
from pkg_resources import resource_filename

# Import from core.face_restoration instead of modules.face_restoration
try:
    from core.face_restoration import restore_faces
except ImportError:
    # Fallback for different import contexts
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.face_restoration import restore_faces

logger = logging.getLogger(__name__)


class Wav2LipUHQ:
    def __init__(self, face, face_restore_model, mouth_mask_dilatation, erode_face_mask, mask_blur, only_mouth,
                 face_swap_img, resize_factor, code_former_weight, debug=False, temp_dir=None):
        self.wav2lip_folder = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
        self.original_wav2lip_folder = self.wav2lip_folder  # Store original path for model weights
        
        # Use temp_dir if provided, otherwise use default
        if temp_dir:
            self.temp_dir = temp_dir
            self.w2l_video = os.path.join(temp_dir, "wav2lip_result.mp4")
        else:
            self.temp_dir = self.wav2lip_folder + '/temp'
            self.w2l_video = self.wav2lip_folder + '/results/result_voice.mp4'
            
        self.original_video = face
        self.face_restore_model = face_restore_model
        self.mouth_mask_dilatation = mouth_mask_dilatation
        self.erode_face_mask = erode_face_mask
        self.mask_blur = mask_blur
        self.only_mouth = only_mouth
        self.face_swap_img = face_swap_img
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ffmpeg_binary = self.find_ffmpeg_binary()
        self.resize_factor = resize_factor
        self.code_former_weight = code_former_weight
        self.debug = debug

    def find_ffmpeg_binary(self):
        for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
            try:
                package_path = resource_filename(package, 'binaries')
                files = [os.path.join(package_path, f) for f in os.listdir(package_path) if f.startswith("ffmpeg-")]
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return files[0] if files else 'ffmpeg'
            except:
                return 'ffmpeg'

    def assure_path_exists(self, path):
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)

    def get_framerate(self, video_file):
        video = cv2.VideoCapture(video_file)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        return fps

    def execute_command(self, command):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(stderr)

    def create_video_from_images(self, nb_frames):
        fps = str(self.get_framerate(self.w2l_video))
        output_dir = os.path.join(self.temp_dir, 'output')
        command = [self.ffmpeg_binary, "-y", "-framerate", fps, "-start_number", "0", "-i",
                   os.path.join(output_dir, "final", "output_%05d.png"), "-vframes",
                   str(nb_frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "8000k",
                   os.path.join(output_dir, "video.mp4")]

        self.execute_command(command)

        command = [self.ffmpeg_binary, "-y", "-framerate", fps, "-start_number", "0", "-i",
                   os.path.join(output_dir, "face_enhanced", "face_restore_%05d.png"), "-vframes",
                   str(nb_frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-b:v", "8000k",
                   os.path.join(output_dir, "video_enhanced.mp4")]

        self.execute_command(command)

    def extract_audio_from_video(self):
        output_dir = os.path.join(self.temp_dir, 'output')
        command = [self.ffmpeg_binary, "-y", "-i", self.w2l_video, "-vn", "-acodec", "copy",
                   os.path.join(output_dir, "output_audio.aac")]
        self.execute_command(command)

    def add_audio_to_video(self):
        output_dir = os.path.join(self.temp_dir, 'output')
        command = [self.ffmpeg_binary, "-y", "-i", os.path.join(output_dir, "video.mp4"), "-i",
                   os.path.join(output_dir, "output_audio.aac"), "-c:v", "copy", "-c:a", "aac", "-strict",
                   "experimental", os.path.join(output_dir, "output_video.mp4")]
        self.execute_command(command)

        command = [self.ffmpeg_binary, "-y", "-i", os.path.join(output_dir, "video_enhanced.mp4"), "-i",
                   os.path.join(output_dir, "output_audio.aac"), "-c:v", "copy", "-c:a", "aac", "-strict",
                   "experimental", os.path.join(output_dir, "output_video_enhanced.mp4")]
        self.execute_command(command)

    def initialize_dlib_predictor(self):
        print("[INFO] Loading the predictor...")
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                flip_input=False, device=self.device)
        # Use original_wav2lip_folder for model weights, not temp_dir
        weights_dir = os.path.dirname(self.original_wav2lip_folder)
        predictor_path = os.path.join(weights_dir, 'weights', 'predicator', 'shape_predictor_68_face_landmarks.dat')
        predictor = dlib.shape_predictor(predictor_path)
        return detector, predictor

    def initialize_video_streams(self):
        print("[INFO] Loading File...")
        vs = cv2.VideoCapture(self.w2l_video)
        vi = cv2.VideoCapture(self.original_video)
        return vs, vi

    def dilate_mouth(self, mouth, w, h):
        mask = np.zeros((w, h), dtype=np.uint8)
        cv2.fillPoly(mask, [mouth], 255)
        kernel = np.ones((self.mouth_mask_dilatation, self.mouth_mask_dilatation), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if contours is empty
        if len(contours) == 0:
            return mouth  # Return original mouth points if no contours found
            
        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        dilated_points = largest_contour.squeeze()
        return dilated_points

    def execute(self, resume=False):
        output_dir = os.path.join(self.temp_dir, 'output')
        debug_path = os.path.join(output_dir, "debug")
        face_enhanced_path = os.path.join(output_dir, "face_enhanced")
        final_path = os.path.join(output_dir, 'final')
        
        # Create directories
        os.makedirs(debug_path, exist_ok=True)
        os.makedirs(face_enhanced_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)
        
        detector, predictor = self.initialize_dlib_predictor()
        vs, vi = self.initialize_video_streams()
        (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        (jstart, jend) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        (nstart, nend) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

        max_frame = str(int(vs.get(cv2.CAP_PROP_FRAME_COUNT)))

        frame_number = 0
        if resume:
            resume_path = os.path.join(self.temp_dir, "resume.json")
            if os.path.exists(resume_path):
                with open(resume_path, "r") as f:
                    parameters = json.load(f)
                # Read frame
                for f in range(parameters["frame"]):
                    _, _ = vs.read()
                    ret, _ = vi.read()
                    if not ret:
                        vi.release()
                        vi = cv2.VideoCapture(self.original_video)
                        _, _ = vi.read()
                frame_number = parameters["frame"]
        print("Face Restoration model: " + str(self.face_restore_model))

        while True:
            print("[INFO] Processing frame: " + str(frame_number) + " of " + max_frame + " - ", end="\r")
            f_number = str(frame_number).rjust(5, '0')

            # Read frame
            ret, w2l_frame = vs.read()
            if not ret:
                break

            ret, original_frame = vi.read()
            if not ret:
                vi.release()
                vi = cv2.VideoCapture(self.original_video)
                ret, original_frame = vi.read()

            if w2l_frame.shape != original_frame.shape:
                if self.resize_factor > 1 and self.face_swap_img is None:
                    original_frame = cv2.resize(original_frame, (w2l_frame.shape[1], w2l_frame.shape[0]))
                else:
                    w2l_frame = cv2.resize(w2l_frame, (original_frame.shape[1], original_frame.shape[0]))

            # Convert to gray
            original_gray = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)

            # Restore face
            w2l_frame_to_restore = cv2.cvtColor(w2l_frame, cv2.COLOR_BGR2RGB)
            image_restored = restore_faces(w2l_frame_to_restore, self.face_restore_model, self.code_former_weight, device=self.device, low_vram=False)

            image_restored2 = cv2.cvtColor(image_restored, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(face_enhanced_path, "face_restore_" + f_number + ".png"), image_restored2)
            image_restored_gray = cv2.cvtColor(image_restored2, cv2.COLOR_RGB2GRAY)

            # Detect faces
            rects = detector.get_detections_for_batch(np.array([np.array(image_restored2)]))

            # Initialize mask
            mask = np.zeros_like(original_gray)

            # Process each detected face
            if not rects or all(rect is None for rect in rects):
                # No faces detected, skip this frame
                frame_number += 1
                continue
                
            for (i, rect) in enumerate(rects):
                if rect is None:
                    continue
                    
                # Get face coordinates
                if not self.only_mouth:
                    shape = predictor(original_gray, dlib.rectangle(*rect))
                    shape = face_utils.shape_to_np(shape)
                    
                    # Add bounds checking for landmarks
                    if len(shape) >= jend:
                        jaw = shape[jstart:jend][1:-1]
                    else:
                        logger.warning(f"Not enough facial landmarks for jaw detection: {len(shape)} points")
                        jaw = shape[jstart:] if len(shape) > jstart else []
                        
                    if len(shape) >= nend:
                        nose = shape[nstart:nend][2]
                    else:
                        logger.warning(f"Not enough facial landmarks for nose detection: {len(shape)} points")
                        nose = shape[nstart] if len(shape) > nstart else shape[-1] if len(shape) > 0 else np.array([0, 0])

                # Get mouth coordinates
                shape = predictor(image_restored_gray, dlib.rectangle(*rect))
                shape = face_utils.shape_to_np(shape)

                # Add bounds checking for mouth landmarks
                if len(shape) >= mend:
                    mouth = shape[mstart:mend][:-8]
                else:
                    logger.warning(f"Not enough facial landmarks for mouth detection: {len(shape)} points")
                    mouth = shape[mstart:] if len(shape) > mstart else shape
                    
                mouth = np.delete(mouth, [3], axis=0)
                if self.mouth_mask_dilatation > 0:
                    mouth = self.dilate_mouth(mouth, original_gray.shape[0], original_gray.shape[1])

                # Create mask for face
                if not self.only_mouth:
                    external_shape = np.append(jaw, [nose], axis=0)
                    external_shape_pts = external_shape.reshape((-1, 1, 2))
                    mask = cv2.fillPoly(mask, [external_shape_pts], 255)
                    if self.erode_face_mask > 0:
                        kernel = np.ones((self.erode_face_mask, self.erode_face_mask), np.uint8)
                        mask = cv2.erode(mask, kernel, iterations=1)
                    # Calculate diff between frames and apply threshold
                    diff = np.abs(original_gray - image_restored_gray)
                    diff[diff > 10] = 255
                    diff[diff <= 10] = 0
                    masked_diff = cv2.bitwise_and(diff, diff, mask=mask)
                else:
                    masked_diff = mask

                # Create mask for mouth
                cv2.fillConvexPoly(masked_diff, mouth, 255)

                # Save mask
                if self.mask_blur > 0:
                    blur = self.mask_blur if self.mask_blur % 2 == 1 else self.mask_blur - 1
                    masked_save = cv2.GaussianBlur(masked_diff, (blur, blur), 0)
                else:
                    masked_save = masked_diff

                original = original_frame.copy()

                # Apply restored face to original image with mask attention
                extended_mask = np.stack([masked_save] * 3, axis=-1)
                normalized_mask = extended_mask / 255.0
                dst = image_restored2 * normalized_mask
                original = original * (1 - normalized_mask) + dst
                original = original.astype(np.uint8)

                # Save final image
                cv2.imwrite(os.path.join(final_path, "output_" + f_number + ".png"), original)

                if self.debug:
                    clone = w2l_frame.copy()
                    if not self.only_mouth:
                        for (x, y) in np.concatenate((jaw, mouth, [nose])):
                            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    else:
                        for (x, y) in mouth:
                            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    if not self.only_mouth:
                        cv2.imwrite(os.path.join(debug_path, "diff_" + f_number + ".png"), diff)
                    cv2.imwrite(os.path.join(debug_path, "points_" + f_number + ".png"), clone)
                    cv2.imwrite(os.path.join(debug_path, 'mask_' + f_number + '.png'), masked_save)
                    cv2.imwrite(os.path.join(debug_path, 'original_' + f_number + '.png'), original_frame)
                    cv2.imwrite(os.path.join(debug_path, "face_restore_" + f_number + ".png"), image_restored2)
                    cv2.imwrite(os.path.join(debug_path, "dst_" + f_number + ".png"), dst)

            frame_number += 1
        torch.cuda.empty_cache()
        if frame_number > 1:
            vs.release()
            vi.release()

            print("[INFO] Create Videos output!")
            self.create_video_from_images(frame_number - 1)
            print("[INFO] Extract Audio from input!")
            self.extract_audio_from_video()
            print("[INFO] Add Audio to Videos!")
            self.add_audio_to_video()
            print("[INFO] Done! file save in output/video_output.mp4")

            if str(frame_number) != max_frame:
                parameters = {"frame": frame_number}
                resume_path = os.path.join(self.temp_dir, "resume.json")
                with open(resume_path, 'w') as f:
                    json.dump(parameters, f)
            else:
                resume_path = os.path.join(self.temp_dir, "resume.json")
                if os.path.exists(resume_path):
                    os.remove(resume_path)
            if self.face_swap_img is None:
                face_swap_output = None
            else:
                face_swap_output = os.path.join(self.temp_dir, "output", "faceswap", "video.mp4")
            return [face_swap_output,
                    self.w2l_video,
                    os.path.join(self.temp_dir, "output", "output_video_enhanced.mp4"),
                    os.path.join(self.temp_dir, "output", "output_video.mp4")]
        else:
            print("[INFO] Interrupted!")
            return None
