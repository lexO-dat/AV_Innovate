# Usar una imagen base de Python
FROM python:3.9

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de requisitos
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]

#docker build -t recomendacion-artistas .
#docker run -d -p 8000:8000 recomendacion-artistas
#http://localhost:8000/docs -> fastapi gui
#post to http://localhost:8000/train or http://localhost:8000/recommend