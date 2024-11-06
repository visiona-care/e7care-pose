from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
from PIL import Image
import uuid
import os

app = FastAPI()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def detect_pose_and_format(image: np.array, output_path: str) -> list:
    # Convertir imagen a RGB y luego a BGR para asegurar que tiene tres canales
    if image.shape[-1] == 4:  # Si la imagen tiene un canal alfa (RGBA), convertir a RGB primero
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[-1] == 1:  # Si la imagen está en escala de grises, convertir a BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Convertir RGB a BGR (el formato esperado por OpenCV)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Procesar la imagen con MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_image)

    formatted_pose_data = []

    if result.pose_landmarks:
        # Dibujar los puntos clave de la pose en la imagen
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Crear un diccionario con el formato especificado
        box = {
            "width": image.shape[1],
            "height": image.shape[0],
            "xMax": max([lm.x for lm in result.pose_landmarks.landmark]),
            "xMin": min([lm.x for lm in result.pose_landmarks.landmark]),
            "yMax": max([lm.y for lm in result.pose_landmarks.landmark]),
            "yMin": min([lm.y for lm in result.pose_landmarks.landmark]),
        }

        keypoints = []
        specific_points = {}

        for idx, landmark in enumerate(result.pose_landmarks.landmark):
            point_data = {
                "x": landmark.x,
                "y": landmark.y,
                "confidence": landmark.visibility,
                "name": mp_pose.PoseLandmark(idx).name.lower(),
            }
            keypoints.append(point_data)

            # Guardar puntos específicos como left_ankle, left_ear, etc.
            specific_points[point_data["name"]] = {
                "x": landmark.x,
                "y": landmark.y,
                "confidence": landmark.visibility,
            }

        formatted_pose_data.append({
            "box": box,
            "id": 1,  # ID único para la detección actual
            "keypoints": keypoints,
            **specific_points,
            "confidence": min([lm.visibility for lm in result.pose_landmarks.landmark])
        })

    # Guardar la imagen con la pose detectada
    cv2.imwrite(output_path, image)

    return formatted_pose_data

@app.post("/detect_pose_with_link/")
async def detect_pose_with_link(file: UploadFile = File(...)):
    try:
        # Leer la imagen de entrada
        image = Image.open(file.file)
        image_np = np.array(image)

        # Generar un nombre único para la imagen de salida
        output_filename = f"pose_{uuid.uuid4()}.png"
        output_path = f"/tmp/{output_filename}"

        # Detectar la pose y formatear la salida
        pose_data = detect_pose_and_format(image_np, output_path)

        # Generar el link a la imagen procesada (asumimos un servidor local)
        image_url = f"http://localhost:8000/images/{output_filename}"

        return JSONResponse(content={
            "image_url": image_url,
            "pose_data": pose_data
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/redirect_to_pose_image/")
async def redirect_to_pose_image(file: UploadFile = File(...)):
    try:
        # Leer la imagen de entrada
        image = Image.open(file.file)
        image_np = np.array(image)

        # Generar un nombre único para la imagen de salida
        output_filename = f"pose_{uuid.uuid4()}.png"
        output_path = f"/tmp/{output_filename}"

        # Detectar la pose y guardar la imagen procesada
        pose_data = detect_pose_and_format(image_np, output_path)

        # Generar el link a la imagen procesada
        image_url = f"http://localhost:8000/images/{output_filename}"

        # Redirigir directamente a la URL de la imagen
        return RedirectResponse(url=image_url)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Endpoint adicional para servir las imágenes procesadas
from fastapi.staticfiles import StaticFiles

# Montar la carpeta temporal como ruta estática
app.mount("/images", StaticFiles(directory="/tmp"), name="images")
