# NewsCheck frontend (Streamlit). Talks only to the Spring Boot backend
# (BACKEND_URL) -- never to the ML service or any model file directly.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# --server.address=0.0.0.0 is required in a container: Streamlit's default
# (localhost) would make it unreachable from outside the container.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
