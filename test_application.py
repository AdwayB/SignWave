import uuid
import cv2
import mediapipe as mp
# import requests
import asyncio
import websockets
import json
import time
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

api_url = "http://localhost:8000"

cap = cv2.VideoCapture(0)

BUFFER_SIZE = 10
SEND_INTERVAL = 0.6


async def send_landmarks():
    buffer = deque(maxlen=BUFFER_SIZE)
    last_send_time = time.time()

    try:
        async with websockets.connect(f"{api_url}/ws/translate") as websocket:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = hands.process(rgb_frame)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                        buffer.append({
                            "uuid": str(uuid.uuid4()),
                            "landmarks": landmarks
                        })

                current_time = time.time()
                if buffer and current_time - last_send_time > SEND_INTERVAL:
                    try:
                        start = time.time()
                        # response = await requests.post(api_url, json={"landmarks": landmarks})
                        await websocket.send(json.dumps(list(buffer)))

                        response = await websocket.recv()

                        end = time.time()
                        latency_ms = (end - start) * 1000

                        response_data = json.loads(response)
                        predicted_label = response_data.get("predicted_label")
                        print(f"Predicted label: {predicted_label} \n Latency: {latency_ms: .2f}ms")

                    except websockets.exceptions.ConnectionClosedError as e:
                        print(f"WebSocket connection closed while sending data: {e}")
                        break

                    except Exception as e:
                        print(f"An error occurred while sending or receiving data: {e}")

                    buffer.clear()
                    last_send_time = current_time

                cv2.imshow('Hand Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


asyncio.run(send_landmarks())
