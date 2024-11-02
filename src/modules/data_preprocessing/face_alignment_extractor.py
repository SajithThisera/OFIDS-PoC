import face_alignment
import torch
import numpy as np

class FaceAlignmentExtractor:
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

    def extract_landmarks_from_frame(self, frame):
        landmarks = self.fa.get_landmarks(frame)
        if landmarks is not None and len(landmarks) > 0:
            return np.array(landmarks[0]).flatten().tolist()
        return None
