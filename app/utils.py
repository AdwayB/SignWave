import os
import torch
from model.model_definition import GestureClassifier

LABELS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

INPUT_SIZE = 63

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, '..', 'model')
model_location = os.path.join(model_dir, 'best_model.pth')
model_location = os.path.normpath(model_location)

PATH_TO_BEST_MODEL = model_location


def load_model(model_path: str, input_size: int, num_classes: int, device: torch.device):
  model = GestureClassifier(input_size=input_size, num_classes=num_classes)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.to(device)
  model.eval()
  return model


def preprocess_landmarks(landmarks: list[float]):
  if not landmarks or len(landmarks) % 3 != 0:
    raise ValueError("Invalid landmark data provided.")

  x_values = landmarks[0::3]
  y_values = landmarks[1::3]
  z_values = landmarks[2::3]

  min_x, max_x = min(x_values), max(x_values)
  min_y, max_y = min(y_values), max(y_values)
  min_z, max_z = min(z_values), max(z_values)

  normalized_landmarks = []
  for i in range(0, len(landmarks), 3):
    normalized_x = (landmarks[i] - min_x) / (max_x - min_x) if max_x != min_x else 0.0
    normalized_y = (landmarks[i + 1] - min_y) / (max_y - min_y) if max_y != min_y else 0.0
    normalized_z = (landmarks[i + 2] - min_z) / (max_z - min_z) if max_z != min_z else 0.0

    normalized_landmarks.extend([normalized_x, normalized_y, normalized_z])

  return normalized_landmarks


def predict_landmark(model, device, landmark_data: list[list[float]]):
  predictions_buffer = []
  for landmarks in landmark_data:
    normalised_landmarks = preprocess_landmarks(landmarks)
    input_tensor = torch.tensor([normalised_landmarks], dtype=torch.float32).to(device)

    with torch.no_grad():
      output = model(input_tensor)
      _, predicted = torch.max(output, 1)

    predictions_buffer.append(LABELS[predicted.item()][0])

  prediction = "".join(predictions_buffer)
  return prediction


def initialize_translation_model():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  num_classes = len(LABELS)

  model = load_model(PATH_TO_BEST_MODEL, INPUT_SIZE, num_classes, device)
  return model, device
