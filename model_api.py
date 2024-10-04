from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import os

app = FastAPI(title="Sistema de Recomendación de Artistas", version="1.0")

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

# Endpoints
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
        
        # Opcional: Guardar el modelo entrenado
        modelo_word2vec.save("word2vec_genres.model")
        
        # Inicializar el sistema de recomendación
        sistema_recomendacion = RecommenderSistemaConSimilitudGeneros(df_artistas, modelo_word2vec, df_artistas)
        
        return {"mensaje": "Modelo entrenado exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        return RecommendResponse(recomendaciones=recomendaciones)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
