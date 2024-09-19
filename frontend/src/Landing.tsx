import { FC, useEffect, useRef, useState } from "react";
import { NormalizedLandmarkList } from "@mediapipe/hands";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Container,
  Paper,
  Snackbar,
  TextField,
  ThemeProvider,
  Typography,
} from "@mui/material";
import VideoCapture from "./components/VideoCapture";
import { theme } from "./theme";

const Landing: FC = () => {
  const [prediction, setPrediction] = useState("");
  const [editableText, setEditableText] = useState("");
  const [wsConnected, setWsConnected] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  const BUFFER_SIZE = 10;
  const SEND_INTERVAL = 600; // milliseconds
  const ADD_INTERVAL = 800; // milliseconds

  const lastAddTime = useRef<number>(Date.now());
  const lastSendTime = useRef<number>(Date.now());
  const buffer = useRef<number[][]>([]);

  useEffect(() => {
    // Initialize WebSocket connection
    ws.current = new WebSocket("ws://localhost:8000/ws");

    ws.current.onopen = () => {
      console.log("WebSocket connection established.");
      setWsConnected(true);
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setPrediction(data.prediction);
      setEditableText(data.prediction);
    };

    ws.current.onclose = () => {
      console.log("WebSocket connection closed.");
      setWsConnected(false);
    };

    ws.current.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    return () => {
      ws.current?.close();
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      const currentTime = Date.now();
      if (
        buffer.current.length > 0 &&
        currentTime - lastSendTime.current > SEND_INTERVAL
      ) {
        // Send buffer via WebSocket
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ landmarks: buffer.current }));
          console.log("Sent buffer:", buffer.current);
          buffer.current = []; // Clear buffer
          lastSendTime.current = currentTime;
        }
      }
    }, 100); // Check every 100ms

    return () => clearInterval(interval);
  }, []);

  const preprocessLandmarks = (landmarks: NormalizedLandmarkList): number[] => {
    return landmarks.flatMap((landmark) => [
      landmark.x,
      landmark.y,
      landmark.z,
    ]);
  };

  const handleLandmarksDetected = (landmarks: NormalizedLandmarkList) => {
    const currentTime = Date.now();
    if (currentTime - lastAddTime.current > ADD_INTERVAL) {
      const processedLandmarks = preprocessLandmarks(landmarks);
      buffer.current.push(processedLandmarks);
      lastAddTime.current = currentTime;

      // Ensure buffer does not exceed BUFFER_SIZE
      if (buffer.current.length > BUFFER_SIZE) {
        buffer.current.shift(); // Remove oldest item
      }
    }
  };

  const copyText = () => {
    navigator.clipboard.writeText(editableText).then(
      () => {
        setCopySuccess(true);
      },
      (err) => {
        console.error("Could not copy text: ", err);
      }
    );
  };

  return (
    <ThemeProvider theme={theme}>
      <Container maxWidth="md">
        <Paper elevation={3} style={{ padding: "20px", marginTop: "20px" }}>
          <Typography variant="h4" component="h1" gutterBottom align="center">
            ISL to Text Translator
          </Typography>
          {!wsConnected ? (
            <Box
              display="flex"
              justifyContent="center"
              alignItems="center"
              minHeight="400px"
            >
              <CircularProgress />
            </Box>
          ) : (
            <>
              <Box display="flex" justifyContent="center" marginBottom={2}>
                <VideoCapture onLandmarksDetected={handleLandmarksDetected} />
              </Box>
              <Typography variant="h6" component="div" gutterBottom>
                {prediction}
              </Typography>
              <TextField
                label="Editable Text"
                multiline
                rows={4}
                value={editableText}
                onChange={(e) => setEditableText(e.target.value)}
                variant="outlined"
                fullWidth
                margin="normal"
              />
              <Button
                variant="contained"
                color="primary"
                onClick={copyText}
                style={{ marginTop: "10px" }}
              >
                Copy Text
              </Button>
            </>
          )}
        </Paper>
        <Snackbar
          open={copySuccess}
          autoHideDuration={3000}
          onClose={() => setCopySuccess(false)}
        >
          <Alert onClose={() => setCopySuccess(false)} severity="success">
            Text copied to clipboard!
          </Alert>
        </Snackbar>
      </Container>
    </ThemeProvider>
  );
};

export { Landing };
