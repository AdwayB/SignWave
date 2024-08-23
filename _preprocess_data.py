import cv2
import mediapipe as mp
import os
import json
import uuid
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.65)


def process_image(image_info):
    image_path, label, unique_id = image_info
    print(f"Processing image at: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)
            all_landmarks.append(landmarks)
        return {'id': unique_id, 'filename': os.path.basename(image_path), 'landmarks': all_landmarks,
                'label': label}
    return None


def load_images_from_directory(base_dir):
    images = []
    for sub_dir in os.listdir(base_dir):
        sub_dir_path = os.path.join(base_dir, sub_dir)
        if os.path.isdir(sub_dir_path):
            label = sub_dir
            for filename in os.listdir(sub_dir_path):
                if filename.endswith('.jpg'):
                    unique_id = str(uuid.uuid4())
                    images.append((os.path.join(sub_dir_path, filename), label, unique_id))
    return images


def normalize_landmarks(data):
    if len(data) == 0 or len(data) == 1:
        return data

    normalized_data = []
    for entry in data:
        for landmarks in entry['landmarks']:
            x_values = landmarks[0::3]
            y_values = landmarks[1::3]
            z_values = landmarks[2::3]

            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)
            min_z, max_z = min(z_values), max(z_values)

            normalized_landmarks = []
            for i in range(0, len(landmarks), 3):
                normalized_landmarks.append((landmarks[i] - min_x) / (max_x - min_x))  # Normalized x
                normalized_landmarks.append((landmarks[i + 1] - min_y) / (max_y - min_y))  # Normalized y
                normalized_landmarks.append((landmarks[i + 2] - min_z) / (max_z - min_z))  # Normalized z

            entry['landmarks'] = normalized_landmarks
        normalized_data.append(entry)
    return normalized_data


def save_to_csv(data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['id', 'filename', 'label'] + [f'landmark_{i}' for i in range(1, 64)]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for entry in data:
            row = {
                'id': entry['id'],
                'filename': entry['filename'],
                'label': entry['label'],
            }
            for i, value in enumerate(entry['landmarks']):
                row[f'landmark_{i + 1}'] = value
            writer.writerow(row)


real_life_images = './datasets/real_life'
staged_images = './datasets/staged'

output_file_real_life = './datasets/landmarks_dataset_a.json'
output_file_staged = './datasets/landmarks_dataset_b.json'

normalized_output_file_real_life = './datasets/normalized_landmarks_dataset_real_life.json'
normalized_output_file_staged = './datasets/normalized_landmarks_dataset_staged.json'

if __name__ == '__main__':
    image_info_list_real_life = load_images_from_directory(real_life_images)
    image_info_list_staged = load_images_from_directory(staged_images)

    data_real_life = [entry for entry in [process_image(info) for info in image_info_list_real_life] if
                      entry is not None]
    data_staged = [entry for entry in [process_image(info) for info in image_info_list_staged] if entry is not None]

    normalized_data_real_life = []
    normalized_data_staged = []

    if data_real_life:
        normalized_data_real_life = normalize_landmarks(data_real_life)

    if data_staged:
        normalized_data_staged = normalize_landmarks(data_staged)

    with open(output_file_real_life, 'w') as f:
        json.dump(data_real_life, f, indent=4)

    with open(output_file_staged, 'w') as f:
        json.dump(data_staged, f, indent=4)

    if data_real_life:
        with open(normalized_output_file_real_life, 'w') as f:
            json.dump(normalized_data_real_life, f, indent=4)

    if data_staged:
        with open(normalized_output_file_staged, 'w') as f:
            json.dump(normalized_data_staged, f, indent=4)

    if data_real_life:
        save_to_csv(data_real_life, './datasets/landmarks_dataset_a.csv')
        save_to_csv(normalized_data_real_life, './datasets/normalized_landmarks_dataset_a.csv')

    if data_staged:
        save_to_csv(data_staged, './datasets/landmarks_dataset_b.csv')
        save_to_csv(normalized_data_staged, './datasets/normalized_landmarks_dataset_b.csv')
