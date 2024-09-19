import React, { useRef, useEffect } from "react";
import {
  Hands,
  HAND_CONNECTIONS,
  NormalizedLandmarkList,
} from "@mediapipe/hands";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import * as cam from "@mediapipe/camera_utils";
import { Box, useMediaQuery, useTheme } from "@mui/material";

interface VideoCaptureProps {
  onLandmarksDetected: (landmarks: NormalizedLandmarkList) => void;
}

const VideoCapture: React.FC<VideoCaptureProps> = ({ onLandmarksDetected }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("sm"));

  const videoWidth = isMobile ? 320 : 640;
  const videoHeight = isMobile ? 240 : 480;

  useEffect(() => {
    const videoElement = videoRef.current!;
    const canvasElement = canvasRef.current!;
    const canvasCtx = canvasElement.getContext("2d")!;

    let camera: cam.Camera;

    const hands = new Hands({
      locateFile: (file: string) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      },
    });

    hands.setOptions({
      maxNumHands: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    hands.onResults(
      (results: {
        image: CanvasImageSource;
        multiHandLandmarks: NormalizedLandmarkList[];
      }) => {
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(
          results.image,
          0,
          0,
          canvasElement.width,
          canvasElement.height
        );

        if (
          results.multiHandLandmarks &&
          results.multiHandLandmarks.length > 0
        ) {
          for (const landmarks of results.multiHandLandmarks) {
            drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
              color: "#00FF00",
              lineWidth: 5,
            });
            drawLandmarks(canvasCtx, landmarks, {
              color: "#FF0000",
              lineWidth: 2,
            });

            onLandmarksDetected(landmarks);
          }
        }

        canvasCtx.restore();
      }
    );

    if (typeof videoElement !== "undefined") {
      camera = new cam.Camera(videoElement, {
        onFrame: async () => {
          await hands.send({ image: videoElement });
        },
        width: videoWidth,
        height: videoHeight,
      });
      camera.start();
    }

    return () => {
      hands.close();
      if (camera) {
        camera.stop();
      }
    };
  }, [onLandmarksDetected, videoWidth, videoHeight]);

  return (
    <Box
      className="video-container"
      position="relative"
      margin="0 auto"
      width={videoWidth}
      height={videoHeight}
    >
      <video ref={videoRef} style={{ display: "none" }} />
      <canvas
        ref={canvasRef}
        width={videoWidth}
        height={videoHeight}
        style={{ border: "1px solid black" }}
      />
    </Box>
  );
};

export default VideoCapture;
