from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from backend.utils import initialize_translation_model, predict_landmark
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

model, device = initialize_translation_model()


@app.websocket("/ws/translate")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            landmarks = data["landmarks"]
            predicted_label = predict_landmark(model, device, landmarks)
            await websocket.send_json({"predicted_label": predicted_label})
    except WebSocketDisconnect:
        print("Client disconnected")


@app.post("/reload_model")
async def reload_model():
    global model, device
    try:
        model, device = await initialize_translation_model()
        logging.info("Model reloaded successfully.")
        return {"message": "model reloaded"}
    except (TimeoutError, ResourceWarning) as e:
        logging.error(f"Failed to reload model: {e}")
        return {"message": "failed to reload model", "error": str(e)}


# IDK, just in case
@app.post("/predict")
async def predict(landmarks: list):
    predicted_label = predict_landmark(model, device, landmarks)
    return {"predicted_label": predicted_label}
