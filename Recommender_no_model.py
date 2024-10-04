import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommenderSistema:
    def __init__(self, artistas_df):
        """
        Inicializa el sistema de recomendación con un DataFrame de artistas.

        Parámetros:
        - artistas_df: DataFrame de pandas que contiene 'artist_id', 'artist_name' y 'genres'.
        """
        self.artistas = artistas_df.copy()
        self._preprocesar_datos()
    
    def _preprocesar_datos(self):
        """Preprocesado de los datos de los artistas."""
        # Binarización de géneros
        self.mlb = MultiLabelBinarizer()
        generos_binarizados = self.mlb.fit_transform(self.artistas['genres'])
        generos_df = pd.DataFrame(generos_binarizados, columns=self.mlb.classes_)
        self.artistas_vect = pd.concat([self.artistas, generos_df], axis=1)
        
        # Vectorización de géneros con TF-IDF
        self.artistas_vect['generos_str'] = self.artistas_vect['genres'].apply(lambda x: ' '.join(x))
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(self.artistas_vect['generos_str'])
    
    def _obtener_generos_favoritos(self, artistas_favoritos):
        """Obtiene los géneros de los artistas favoritos del usuario."""
        generos_favoritos = set()
        for artista in artistas_favoritos:
            generos_artista = self.artistas_vect.loc[self.artistas_vect['artist_name'] == artista, 'genres'].values
            if len(generos_artista) > 0:
                generos_favoritos.update(generos_artista[0])
        return list(generos_favoritos)
    
    def recomendar(self, artistas_favoritos, generos_explicitos, top_n=10):
        """
        Genera recomendaciones basadas en los artistas y géneros favoritos del usuario.

        Parámetros:
        - artistas_favoritos: lista de nombres de artistas favoritos del usuario.
        - generos_explicitos: lista de géneros explícitamente preferidos por el usuario.
        - top_n: número de recomendaciones a retornar.

        Retorna:
        - Lista de nombres de artistas recomendados.
        """
        generos_de_artistas = self._obtener_generos_favoritos(artistas_favoritos)
        
        generos_usuario_combinados = list(set(generos_explicitos + generos_de_artistas))
        
        generos_usuario_str = ' '.join(generos_usuario_combinados)
        perfil_usuario_tfidf = self.tfidf.transform([generos_usuario_str])

        # cosine similarity        
        similitud_coseno = cosine_similarity(perfil_usuario_tfidf, self.tfidf_matrix).flatten()
        self.artistas_vect['Similitud'] = similitud_coseno
        
        recomendaciones = self.artistas_vect.sort_values(by='Similitud', ascending=False)
        top_n_recomendaciones = recomendaciones.head(top_n)['artist_name'].tolist()
        
        return top_n_recomendaciones

    
# Datos de los artistas disponibles en la web
artistas = pd.DataFrame({
    'artist_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    'artist_name': [
        'Radiohead', 'Drake', 'The Weeknd', 'Metallica', 'Taylor Swift', 
        'Beyoncé', 'Kendrick Lamar', 'Coldplay', 'Billie Eilish', 'Eminem', 
        'Arctic Monkeys', 'Travis Scott', 'Adele', 'Ed Sheeran', 'Dua Lipa', 
        'Imagine Dragons', 'Bruno Mars', 'Daft Punk', 'Nirvana', 'Post Malone', 
        'Maroon 5', 'Justin Bieber', 'Linkin Park', 'Shakira', 'Bad Bunny', 
        'J Balvin', 'Ozuna', 'System of a Down', 'Slipknot', 'Camila Cabello'
    ],
    'genres': [
        ['Rock', 'Alternative'], ['Rap', 'Hip-hop'], ['Pop', 'R&B'], 
        ['Rock', 'Metal'], ['Pop'], ['Pop', 'R&B', 'Soul'], ['Rap', 'Hip-hop'], 
        ['Rock', 'Pop'], ['Alternative', 'Pop'], ['Rap', 'Hip-hop'], 
        ['Rock', 'Indie'], ['Rap', 'Hip-hop'], ['Pop', 'Soul'], ['Pop'], 
        ['Pop', 'Dance'], ['Rock', 'Pop', 'Alternative'], ['Pop', 'Funk', 'R&B'], 
        ['Electronic', 'Dance'], ['Rock', 'Grunge'], ['Hip-hop', 'Rap', 'Pop'], 
        ['Pop', 'Rock'], ['Pop'], ['Rock', 'Alternative', 'Nu-metal'], ['Pop', 'Latin'], 
        ['Reggaeton', 'Latin'], ['Reggaeton', 'Latin'], ['Reggaeton', 'Latin'], 
        ['Metal', 'Rock'], ['Metal', 'Nu-metal'], ['Pop', 'Latin']
    ]
})

# Inicializacion
sistema_recomendacion = RecommenderSistema(artistas)

# Info extraida del usuario
artistas_favoritos_usuario = ['Ozuna', 'Bad Bunny']
generos_usuario = ['Pop', 'Latin']

# Generacion de las recomendaciones
top_recomendaciones = sistema_recomendacion.recomendar(
    artistas_favoritos=artistas_favoritos_usuario,
    generos_explicitos=generos_usuario,
    top_n=10
)

print("Top Recomendaciones de Artistas:")
for idx, artista in enumerate(top_recomendaciones, start=1):
    print(f"{idx}. {artista}")
