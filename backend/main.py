from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from backend.utils import initialize_translation_model, predict_landmark
import logging

logging.basicConfig(level=logging.WARNING)

app = FastAPI()

model, device = initialize_translation_model()


@app.websocket("/ws/translate")
async def websocket_endpoint(websocket: WebSocket):
  await websocket.accept()
  try:
    while True:
      data_buffer = await websocket.receive_json()
      landmarks_array = [data["landmarks"] for data in data_buffer]
      last_uuid = data_buffer[-1]['uuid']
      predicted_label = predict_landmark(model, device, landmarks_array)
      await websocket.send_json({"last_uuid": last_uuid, "predicted_label": predicted_label})
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
async def predict(data: list):
  landmarks_array = [d["landmarks"] for d in data]
  last_uuid = data[-1]['uuid']
  predicted_label = predict_landmark(model, device, landmarks_array)
  return {"last_uuid": last_uuid, "predicted_label": predicted_label}
