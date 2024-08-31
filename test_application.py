import cv2
import mediapipe as mp
# import requests
import asyncio
import websockets
import json
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

api_url = "http://localhost:8000"

cap = cv2.VideoCapture(0)


async def send_landmarks():
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

                        start = time.time()
                        # response = await requests.post(api_url, json={"landmarks": landmarks})
                        await websocket.send(json.dumps({"landmarks": landmarks}))

                        response = await websocket.recv()

                        end = time.time()
                        latency_ms = (end - start) * 1000

                        response_data = json.loads(response)
                        predicted_label = response_data.get("predicted_label")
                        print(f"Predicted label: {predicted_label} \n Latency: {latency_ms: .2f}ms")

                cv2.imshow('Hand Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"WebSocket connection closed: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


asyncio.run(send_landmarks())
