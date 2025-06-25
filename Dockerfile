FROM python:3.10-slim

WORKDIR /app
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install mlflow

EXPOSE 5000
CMD ["mlflow", "ui", "--backend-store-uri", "./mlruns", "--host", "0.0.0.0", "--port", "5000"]
