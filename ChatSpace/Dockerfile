FROM python:3.12-slim

# HF Spaces runs containers as uid 1000; install into that user's home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user Requirements.txt .
RUN pip install --no-cache-dir --user -r Requirements.txt

COPY --chown=user App.py .
COPY --chown=user Static ./Static

EXPOSE 7860
CMD ["uvicorn", "App:app", "--host", "0.0.0.0", "--port", "7860"]
