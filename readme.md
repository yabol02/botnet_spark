```bash
# 1. Construir la imagen
sudo docker build -f docker/Dockerfile -t spark-jupyter .

# 2. Crear el contenedor con la nueva ruta de volumen
sudo docker run -d --name spark-jupyter -p 8888:8888 -v "$PWD/docker/docker_volumen":/app/local spark-jupyter
# - Alternativamente en windows (power shell):
docker run -d --name spark-jupyter -p 8888:8888 -v "${PWD}\docker\docker_volumen:/app/data" spark-jupyter

# 3. Iniciar
sudo docker start spark-jupyter

# 4. Ver logs para obtener el token de acceso
sudo docker logs spark-jupyter