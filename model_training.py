# train_model.py
import pandas as pd
from gensim.models import Word2Vec
import ast

# Cargar el archivo CSV
df = pd.read_csv("artists.csv")

# Función para parsear géneros usando ast.literal_eval
def parse_genres(x):
    if pd.notna(x):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return ['unknown']
    else:
        return ['unknown']

# Rellenar filas con géneros vacíos con 'unknown' y convertir cadenas a listas
df['genres'] = df['genres'].apply(parse_genres)

# Eliminar filas que no tienen géneros si prefieres
df = df[df['genres'].apply(lambda x: len(x) > 0)]

# Ver el formato final (opcional)
print(df.head())

# Obtener la lista de géneros
genres_list = df['genres'].tolist()

# Entrenar el modelo Word2Vec con los géneros
model = Word2Vec(sentences=genres_list, vector_size=100, window=5, min_count=1, workers=4)

# Guardar el modelo para uso posterior
model.save("word2vec_genres.model")

print("Modelo Word2Vec entrenado y guardado exitosamente.")
