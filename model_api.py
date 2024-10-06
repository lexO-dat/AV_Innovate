from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import json
import requests
import numpy as np
from gensim.models import Word2Vec
import os

app = FastAPI(title="Sistema de Recomendación de Artistas", version="1.0")

# Habilitar CORS para permitir cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir cualquier encabezado
)

# Modelos de Datos
class Artista(BaseModel):
    artist_id: int
    artist_name: str
    genres: List[str]

class TrainData(BaseModel):
    artistas: List[Artista]

class RecommendRequest(BaseModel):
    artistas_favoritos: List[str]
    generos_explicitos: List[str]

class RecommendResponse(BaseModel):
    recomendaciones: List[str]

# Funciones de Recomendación
def obtener_generos_artista(artista_name, df):
    generos = df.loc[df['artist_name'].str.lower() == artista_name.lower(), 'genres'].values
    if len(generos) > 0:
        return generos[0]
    else:
        return []

class RecommenderSistemaConSimilitudGeneros:
    def __init__(self, artistas_df, model, df_artistas_csv):
        self.artistas = artistas_df.copy()
        self.model = model
        self.df_artistas_csv = df_artistas_csv
    
    def _calcular_similitud_entre_generos(self, generos_usuario, generos_artista):
        similitudes = []
        for genero_usuario in generos_usuario:
            for genero_artista in generos_artista:
                if genero_usuario in self.model.wv and genero_artista in self.model.wv:
                    similitud = self.model.wv.similarity(genero_usuario, genero_artista)
                    similitudes.append(similitud)
        return np.mean(similitudes) if similitudes else 0
    
    def recomendar(self, artistas_favoritos, generos_explicitos, top_n=10):
        generos_de_artistas = []
        for artista in artistas_favoritos:
            generos_artista = obtener_generos_artista(artista, self.df_artistas_csv)
            generos_de_artistas.extend(generos_artista)
        generos_usuario_combinados = list(set(generos_explicitos + generos_de_artistas))
        recomendaciones = []
        for _, artista in self.artistas.iterrows():
            similitud_generos = self._calcular_similitud_entre_generos(generos_usuario_combinados, artista['genres'])
            recomendaciones.append((artista['artist_name'], similitud_generos))
        recomendaciones.sort(key=lambda x: x[1], reverse=True)
        top_n_recomendaciones = [rec[0] for rec in recomendaciones[:top_n]]
        return top_n_recomendaciones

# Variables Globales
modelo_word2vec = None
sistema_recomendacion = None
df_artistas = None

# Endpoint para entrenar el modelo
@app.post("/train", summary="Entrena el modelo con los datos de artistas proporcionados.")
def train_model(data: TrainData):
    global modelo_word2vec, sistema_recomendacion, df_artistas
    
    try:
        # Convertir los datos recibidos en un DataFrame
        df_artistas = pd.DataFrame([{
            'artist_id': artista.artist_id,
            'artist_name': artista.artist_name,
            'genres': artista.genres
        } for artista in data.artistas])
        
        # Preparar los géneros para el entrenamiento del modelo Word2Vec
        genres_list = df_artistas['genres'].tolist()
        modelo_word2vec = Word2Vec(sentences=genres_list, vector_size=100, window=5, min_count=1, workers=4)
        
        # Inicializar el sistema de recomendación
        sistema_recomendacion = RecommenderSistemaConSimilitudGeneros(df_artistas, modelo_word2vec, df_artistas)
        
        return {"mensaje": "Modelo entrenado exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

json_recomendados = "json_recomendado.json"
# Endpoint para generar recomendaciones basado en un POST
@app.post("/recommend", response_model=RecommendResponse, summary="Genera recomendaciones de artistas basadas en las preferencias del usuario.")
def recommend(recommend_request: RecommendRequest):
    global sistema_recomendacion, df_artistas, modelo_word2vec

    if sistema_recomendacion is None:
        raise HTTPException(status_code=400, detail="El modelo no ha sido entrenado. Por favor, entrena el modelo usando el endpoint /train primero.")
    
    try:
        recomendaciones = sistema_recomendacion.recomendar(
            artistas_favoritos=recommend_request.artistas_favoritos,
            generos_explicitos=recommend_request.generos_explicitos
        )
        
        data_a_guardar = {"artistas": recomendaciones}

        # Guardar el diccionario en un archivo JSON
        with open("json_recomendado.json", "w", encoding="utf-8") as archivo_json:
            json.dump(data_a_guardar, archivo_json, ensure_ascii=False, indent=4)
        
        return RecommendResponse(recomendaciones=recomendaciones)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al generar recomendaciones: {str(e)}")

# Supongamos que el archivo JSON principal está en el mismo directorio que este script
json_principal_path = "artistas_web.json"

@app.get("/recomendaciones", summary="Devuelve recomendaciones de artistas previamente calculadas", response_model=dict)
def obtener_recomendaciones():
    
    #recomendaciones = json guardado
    
    try:
        with open(json_recomendados, "r", encoding="utf-8") as archivo_json:
            recomendaciones_json = json.load(archivo_json)
            recomendaciones = recomendaciones_json.get("artistas", [])
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="No se pudo encontrar el archivo JSON de recomendaciones.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error al decodificar el archivo JSON de recomendaciones.")
    
    try:
        # Cargar el JSON principal desde el archivo
        try:
            with open(json_principal_path, "r", encoding="utf-8") as infile:
                json_principal = json.load(infile)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="No se pudo encontrar el archivo JSON principal.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Error al decodificar el archivo JSON principal.")

        # Asegurarse de que la estructura de datos es la esperada
        if not isinstance(json_principal, dict) or "artistas" not in json_principal:
            raise HTTPException(status_code=500, detail="El formato del archivo JSON principal es inválido.")
        
        # Filtrar los artistas recomendados del JSON principal
        artistas_filtrados = [
            artista for artista in json_principal["artistas"]
            if artista.get("artist_name") in recomendaciones
        ]

        # Tomar los 2 primeros artistas recomendados
        artistas_a_enviar = artistas_filtrados[:2]

        # Crear el cuerpo de la solicitud para el endpoint /recommend
        payload = {
            "mail": "Usuario",
            "subject": "Recomendaciones musicales personalizadas",
            "body": "Basado en tus preferencias musicales, estos son algunos artistas recomendados.",
            "artists": [
                {
                    "nombre": artista["artist_name"],
                    "fecha_evento": artista["concert_date"],
                    "hora_evento": artista["concert_time"],
                    "imagen": artista["photo_url"]
                } for artista in artistas_a_enviar
            ],
            "destinationEmail": "lucasabello4@gmail.com"  # Modifica el correo según sea necesario
        }

        # Crear el nuevo JSON con los artistas recomendados
        nuevo_json = {
            "artistas": artistas_filtrados
        }

        # Devolver el nuevo JSON con los artistas recomendados como respuesta
        return nuevo_json
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al obtener recomendaciones: {str(e)}")

@app.get("/json", summary="Devuelve el JSON principal de artistas.")
def obtener_artistas_web():
    try:
        # Cargar el JSON principal desde el archivo
        with open(json_principal_path, "r", encoding="utf-8") as infile:
            json_principal = json.load(infile)
        
        # Asegurarse de que la estructura de datos es la esperada
        if not isinstance(json_principal, dict) or "artistas" not in json_principal:
            raise HTTPException(status_code=500, detail="El formato del archivo JSON principal es inválido.")

        # Retornar el JSON principal
        return json_principal
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="No se pudo encontrar el archivo JSON principal.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error al decodificar el archivo JSON principal.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el archivo JSON: {str(e)}")