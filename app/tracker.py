import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import supervision as sv
import os
from transformers import CLIPProcessor, CLIPModel
from app.config import settings

class ObjectTracker:
    def __init__(self, processor, model, device):
        """
        Initialize tracker with CLIP model components
        
        Args:
            processor: CLIPProcessor instance
            model: CLIPModel instance
            device: torch device (cuda or cpu)
        """
        self.yolo = YOLO(settings.yolo_model)
        self.yolo.to(device)  # Move YOLO to GPU if available
        self.byte_tracker = sv.ByteTrack()
        self.processor = processor
        self.model = model
        self.device = device
        self.reference_embedding = None
        self.target_id = None
        self.target_timestamps = []
        self.last_confirmed_frame = -1  # Track last confirmed target sighting
        self.tracking_confidence = 1.0  # Confidence in current track

    def _get_embedding(self, crop):
        """Extract CLIP embedding from person crop"""
        if crop.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            inputs = self.processor(images=[pil], return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                return features / features.norm(p=2, dim=-1, keepdim=True)
        except Exception as e:
            print(f"[WARN] Embedding extraction error: {e}")
            return None

    def _get_embeddings_batch(self, crops):
        """Batch process multiple crops for faster CLIP inference"""
        if not crops or len(crops) == 0:
            return []
        
        try:
            pil_images = []
            valid_indices = []
            
            # Convert crops to PIL images
            for i, crop in enumerate(crops):
                if crop.size > 0:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    pil_images.append(pil)
                    valid_indices.append(i)
            
            if not pil_images:
                return [None] * len(crops)
            
            # Process all images at once
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            # Map back to original positions
            result = [None] * len(crops)
            for out_idx, orig_idx in enumerate(valid_indices):
                result[orig_idx] = features[out_idx:out_idx+1]  # Keep batch dimension
            
            return result
        except Exception as e:
            print(f"[WARN] Batch embedding error: {e}")
            return [None] * len(crops)

    def _similarity_check(self, embedding):
        """Fast similarity check without redundant embedding extraction"""
        if embedding is None or self.reference_embedding is None:
            return 0.0
        try:
            sim = torch.cosine_similarity(self.reference_embedding, embedding).item()
            return sim
        except Exception as e:
            print(f"[WARN] Similarity check error: {e}")
            return 0.0

    def process_frame(self, frame, frame_idx):
        """Process single frame for tracking with adaptive accuracy boost"""
        results = self.yolo.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = self.byte_tracker.update_with_detections(detections)

        annotated = frame.copy()
        target_found = False

        # Handle empty detections
        if len(detections) == 0:
            return annotated, target_found

        # Collect all person data and crops for batch processing
        person_data = []
        crops_list = []
        confidence_scores = []
        
        for i in range(len(detections)):
            try:
                xyxy = detections.xyxy[i]
                conf = detections.confidence[i] if detections.confidence is not None else 0.0
                class_id = detections.class_id[i] if detections.class_id is not None else -1
                track_id = detections.tracker_id[i] if detections.tracker_id is not None else None
                
                class_id = int(class_id)
                if class_id != 0:  # Only persons (class 0)
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                crop = frame[y1:y2, x1:x2]
                track_id_int = int(track_id) if track_id is not None else None
                
                if crop.size == 0:
                    continue
                
                person_data.append({
                    'xyxy': (x1, y1, x2, y2),
                    'track_id': track_id_int,
                    'conf': float(conf)
                })
                crops_list.append(crop)
                confidence_scores.append(float(conf))
            except Exception as e:
                print(f"[ERROR] Detection collection error at {i}: {e}")
                continue

        # Initialize reference on first frame
        if self.reference_embedding is None and len(crops_list) > 0:
            self.reference_embedding = self._get_embedding(crops_list[0])
            self.target_id = person_data[0]['track_id']

        # AGGRESSIVE CLIP checking: Check EVERY frame to catch track switches
        # Previous strategy (every 2 frames) missed boy/man switches mid-scene
        min_confidence = min(confidence_scores) if confidence_scores else 1.0
        is_low_confidence = min_confidence < 0.7
        target_missing_recently = (frame_idx - self.last_confirmed_frame) > 0 and (frame_idx - self.last_confirmed_frame) < 3
        should_check_embeddings = len(crops_list) > 0  # Check EVERY frame for interview-quality accuracy
        
        embeddings = []
        if should_check_embeddings:
            embeddings = self._get_embeddings_batch(crops_list)
        
        # Check each person - with improved re-detection logic
        best_match_idx = -1
        best_similarity = 0.0
        
        for idx, person in enumerate(person_data):
            x1, y1, x2, y2 = person['xyxy']
            track_id_int = person['track_id']
            conf = person['conf']
            
            if track_id_int is not None and self.target_id is not None:
                # Fast check: use track ID directly (ByteTrack maintains identity)
                is_target = (track_id_int == self.target_id)
                
                # Use CLIP embedding verification when:
                # 1. Track ID doesn't match but we have embeddings (re-detection)
                # 2. Confidence is low (uncertain detection)
                # 3. Target recently lost (occlusion recovery)
                if should_check_embeddings and idx < len(embeddings):
                    embedding = embeddings[idx]
                    sim_score = self._similarity_check(embedding) if embedding is not None else 0.0
                    
                    # Track best match for when target is temporarily lost
                    if sim_score > best_similarity:
                        best_similarity = sim_score
                        best_match_idx = idx
                    
                    # Mark as target if similarity is high
                    if sim_score >= settings.reid_threshold:
                        is_target = True
                
                if is_target:
                    self.target_id = track_id_int
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(annotated, "TARGET", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    self.target_timestamps.append(frame_idx)
                    target_found = True
        
        # Fallback: if target not found but have a STRONG CLIP match, re-detect
        # Use same threshold (0.55) to prevent re-detecting wrong person
        if not target_found and best_match_idx >= 0 and best_similarity >= settings.reid_threshold:
            person = person_data[best_match_idx]
            x1, y1, x2, y2 = person['xyxy']
            track_id_int = person['track_id']
            
            # Re-acquire target
            self.target_id = track_id_int
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(annotated, "TARGET [RE-DETECTED]", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self.target_timestamps.append(frame_idx)
            target_found = True

        return annotated, target_found

    def _is_target(self, crop):
        """Check if crop matches reference embedding"""
        emb = self._get_embedding(crop)
        if emb is None or self.reference_embedding is None:
            return False
        try:
            sim = torch.cosine_similarity(self.reference_embedding, emb).item()
            return sim >= settings.reid_threshold
        except Exception as e:
            print(f"⚠️ Similarity check error: {e}")
            return False

    def generate_tracked_video(self, original_video_path: str, video_id: str, query_text: str):
        """Generate video with persistent tracking"""
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(settings.frame_storage_path, f"{video_id}_tracked.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated, _ = self.process_frame(frame, frame_idx)
            out.write(annotated)
            
            if (frame_idx + 1) % 30 == 0:
                print(f"  Processing: {frame_idx + 1}/{total_frames} frames...")
            
            frame_idx += 1

        cap.release()
        out.release()
        print(f"✅ Tracked video saved: {output_path}")
        return output_path

    def generate_all_detections_video(self, original_video_path: str, video_id: str):
        """Generate video with YOLO bounding boxes on ALL detected persons"""
        cap = cv2.VideoCapture(original_video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join(settings.frame_storage_path, f"{video_id}_yolo_boxes.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO detection
            results = self.yolo(frame, verbose=False)
            detections = sv.Detections.from_ultralytics(results[0])
            
            annotated = frame.copy()
            
            # Draw bounding boxes for ALL detected persons
            if len(detections) > 0:
                for i in range(len(detections)):
                    try:
                        xyxy = detections.xyxy[i]
                        conf = detections.confidence[i] if detections.confidence is not None else 0.0
                        class_id = detections.class_id[i] if detections.class_id is not None else -1
                        
                        class_id = int(class_id)
                        if class_id != 0:  # Only persons (class 0)
                            continue
                        
                        x1, y1, x2, y2 = map(int, xyxy)
                        conf_score = float(conf)
                        
                        # Draw rectangle in blue for all persons
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
                        # Add label with confidence
                        label = f"Person {conf_score:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 4), 
                                    (x1 + label_size[0], y1), (255, 0, 0), -1)
                        cv2.putText(annotated, label, (x1, y1 - 4), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    except Exception as e:
                        print(f"[ERROR] YOLO detection error at index {i}: {e}")
                        continue
            
            # Add timestamp in corner
            timestamp_text = f"Frame: {frame_idx} | Time: {frame_idx/fps:.2f}s"
            cv2.putText(annotated, timestamp_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(annotated)
            
            if (frame_idx + 1) % 30 == 0:
                print(f"  Processing: {frame_idx + 1}/{total_frames} frames...")
            
            frame_idx += 1

        cap.release()
        out.release()
        print(f"✅ YOLO boxes video saved: {output_path}")
        return output_path
