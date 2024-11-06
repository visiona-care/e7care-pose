from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
from PIL import Image
import uuid
import os
from ultralytics import YOLO  # Modelo YOLOv5

app = FastAPI()

# Inicializamos YOLO y MediaPipe
model = YOLO('yolov8n.pt')  # Puedes ajustar el modelo aquí
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def detect_people_and_poses(image: np.array, output_path: str) -> list:
    results = model(image)  # Detectar personas en la imagen con YOLO
    formatted_pose_data = []

    for detection in results[0].boxes:
        if detection.cls[0] == 0:  # Clase 0 en YOLO normalmente es 'persona'
            # Extraer coordenadas del bounding box
            x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])

            # Recortar la imagen a la región de la persona detectada
            person_image = image[y_min:y_max, x_min:x_max]
            rgb_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_image)

            if result.pose_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    person_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                # Definir el bounding box y los puntos de pose en el formato deseado
                box = {
                    "width": x_max - x_min,
                    "height": y_max - y_min,
                    "xMax": x_max,
                    "xMin": x_min,
                    "yMax": y_max,
                    "yMin": y_min,
                }

                keypoints = []
                specific_points = {}

                for idx, landmark in enumerate(result.pose_landmarks.landmark):
                    point_data = {
                        "x": landmark.x * (x_max - x_min) + x_min,
                        "y": landmark.y * (y_max - y_min) + y_min,
                        "confidence": landmark.visibility,
                        "name": mp_pose.PoseLandmark(idx).name.lower(),
                    }
                    keypoints.append(point_data)

                    # Guardar puntos específicos
                    specific_points[point_data["name"]] = {
                        "x": point_data["x"],
                        "y": point_data["y"],
                        "confidence": landmark.visibility,
                    }

                formatted_pose_data.append({
                    "box": box,
                    "id": len(formatted_pose_data) + 1,  # ID único para cada detección
                    "keypoints": keypoints,
                    **specific_points,
                    "confidence": min([lm.visibility for lm in result.pose_landmarks.landmark])
                })

    # Guardar la imagen con las poses detectadas
    cv2.imwrite(output_path, image)

    return formatted_pose_data

@app.post("/detect_pose_with_link/")
async def detect_pose_with_link(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        image_np = np.array(image)

        # Generar un nombre único para la imagen de salida
        output_filename = f"pose_{uuid.uuid4()}.png"
        output_path = f"/tmp/{output_filename}"

        # Detectar poses y formatear la salida
        pose_data = detect_people_and_poses(image_np, output_path)

        # Generar el link a la imagen procesada (asumimos un servidor local)
        image_url = f"http://localhost:8000/images/{output_filename}"

        return JSONResponse(content={
            "image_url": image_url,
            "pose_data": pose_data
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint adicional para servir las imágenes procesadas
from fastapi.staticfiles import StaticFiles
app.mount("/images", StaticFiles(directory="/tmp"), name="images")
