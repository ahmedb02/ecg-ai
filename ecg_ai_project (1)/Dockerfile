FROM python:3.11-slim

WORKDIR /app

# CPU-only torch wheel: much smaller/faster to build than the default
# CUDA-bundled wheel, and this app only needs CPU inference.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ecg_ai ./ecg_ai
COPY backend ./backend
COPY frontend ./frontend

# 7860 is the conventional port for container-based app platforms
# (Hugging Face Spaces' Docker SDK default among them); override PORT if
# your platform expects something else.
EXPOSE 7860
ENV PORT=7860

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT}"]
