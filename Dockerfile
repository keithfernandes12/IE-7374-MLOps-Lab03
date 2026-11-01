FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY model/ model/

# Run as a non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
