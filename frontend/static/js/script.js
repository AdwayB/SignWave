const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
const outputDiv = document.getElementById('output');

let websocket;

const hands = new Hands({locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`;
}});

hands.setOptions({
    maxNumHands: 2,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    },
    width: 640,
    height: 480
});
camera.start();

function initWebSocket() {
    websocket = new WebSocket(`ws://${window.location.host.replace('8080', '8000')}/ws/translate`);

    websocket.onopen = () => {
        console.log('WebSocket connection established');
    };

    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received from server:', data);
        outputDiv.innerText = `Prediction: ${data.predicted_label}`;
    };

    websocket.onclose = () => {
        console.log('WebSocket connection closed, retrying in 1 second');
        setTimeout(initWebSocket, 1000);
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

initWebSocket();

function generateUUID() {
    let d = new Date().getTime();
    let d2 = (performance && performance.now && (performance.now() * 1000)) || 0;
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        let r = Math.random() * 16;
        if(d > 0){
            r = (d + r)%16 | 0;
            d = Math.floor(d/16);
        } else {
            r = (d2 + r)%16 | 0;
            d2 = Math.floor(d2/16);
        }
        return (c==='x' ? r :(r&0x3|0x8)).toString(16);
    });
}

let dataBuffer = [];
const BUFFER_SIZE = 10;
const SEND_INTERVAL = 600;
const ADD_INTERVAL = 800;

let lastSendTime = Date.now();
let lastAddTime = Date.now();

function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(
        results.image, 0, 0, canvasElement.width, canvasElement.height);

    // Draw hand landmarks
    if (results.multiHandLandmarks && (Date.now() - lastAddTime > ADD_INTERVAL)) {
        for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                           {color: '#00FF00', lineWidth: 5});
            drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
        }

        const landmarksArray = results.multiHandLandmarks.map(hand => {
            return hand.map(landmark => [landmark.x, landmark.y, landmark.z]);
        });

        const uuid = generateUUID();

        dataBuffer.push({
            uuid: uuid,
            landmarks: landmarksArray
        });

        lastAddTime = Date.now();
    }
    canvasCtx.restore();

    if (dataBuffer.length > 0 && (Date.now() - lastSendTime > SEND_INTERVAL || dataBuffer.length >= BUFFER_SIZE)) {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify(dataBuffer));
            dataBuffer = [];
            lastSendTime = Date.now();
        }
    }
}
