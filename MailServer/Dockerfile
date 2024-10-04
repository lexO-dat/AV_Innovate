# Etapa de compilación
FROM golang:1.19-alpine AS builder

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar go.mod y go.sum primero para aprovechar el caché de Docker
COPY go.mod go.sum ./

# Descargar dependencias
RUN go mod download

# Copiar el resto de los archivos del proyecto al contenedor
COPY . .

# Construir la aplicación
RUN go build -o server main.go

# Etapa de ejecución
FROM alpine:latest

# Crear un directorio de trabajo
WORKDIR /app

# Copiar el binario compilado desde la etapa de construcción
COPY --from=builder /app/server /app/server

# Copiar el archivo .env (si lo tienes) al contenedor
COPY .env /app/.env

# Exponer el puerto en el que se ejecutará el servidor
EXPOSE 8080

# Comando para ejecutar la aplicación cuando el contenedor arranca
CMD ["/app/server"]
