import cv2
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import joblib
import subprocess
import os
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from ultralytics import YOLO


MODEL_DIR = "models"
OUTPUT_DIR = "output"
ALERT_DIR = os.path.join(OUTPUT_DIR, "alerts")

os.makedirs(ALERT_DIR, exist_ok=True)


yolo_model = YOLO(os.path.join(MODEL_DIR, "yolo11n.pt"))

gunshot_model = models.Sequential([
    layers.Input(shape=(1024,)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")
])

gunshot_model.load_weights(os.path.join(MODEL_DIR, "gunshot_detection_model.h5"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")



def extract_audio(video_path, audio_path=os.path.join(OUTPUT_DIR, "temp_audio.wav")):

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        audio_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return audio_path


def predict_gunshot(audio_segment):

    scores, embeddings, spectrogram = yamnet_model(audio_segment)
    embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    embedding = embedding.reshape(1, -1)

    pred = gunshot_model.predict(embedding, verbose=0)
    label = label_encoder.inverse_transform([np.argmax(pred)])

    return label[0]


def process_video(video_path):

    audio_path = extract_audio(video_path)
    audio, sr = librosa.load(audio_path, sr=16000)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    audio_pointer = 0
    window_size = int(sr * 1)

    gunshot_flag = False

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        human_detected = False
        results = yolo_model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    human_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cv2.rectangle(frame,
                                  (x1, y1),
                                  (x2, y2),
                                  (0,255,0),
                                  2)

                    cv2.putText(frame,
                                "Human",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0,255,0),
                                2)


        if frame_count % int(fps) == 0:
            audio_segment = audio[audio_pointer:audio_pointer+window_size]

            if len(audio_segment) == window_size:
                label = predict_gunshot(audio_segment)

                if label == "gunshot":
                    gunshot_flag = True
                else:
                    gunshot_flag = False

            audio_pointer += window_size

        if human_detected and gunshot_flag:
            cv2.putText(frame,
                        "HIGH RISK: HUMAN + GUNSHOT",
                        (40,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

            cv2.imwrite(os.path.join(ALERT_DIR, f"high_risk_{frame_count}.jpg"), frame)


        elif gunshot_flag:

            cv2.putText(frame,
                        "GUNSHOT DETECTED",
                        (40,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,165,255),
                        3)

            cv2.imwrite(os.path.join(ALERT_DIR, f"gunshot_{frame_count}.jpg"), frame)


        elif human_detected:

            cv2.putText(frame,
                        "HUMAN DETECTED",
                        (40,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,255,0),
                        2)

        cv2.imshow("Poaching Detection System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_count += 1


    cap.release()
    cv2.destroyAllWindows()


process_video("input.mp4")