# Smile-design analysis server for a free Hugging Face Space (Docker). Photos are analysed in memory only.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.5.3 opencv-python-headless==5.0.0.93 onnxruntime==1.29.0

# Hugging Face runs Spaces as user 1000
RUN useradd -m -u 1000 app
WORKDIR /app
COPY --chown=app core/dsd /app/core/dsd
COPY --chown=app server /app/server

# the tooth model, from the project's GitHub repository, checked against its fingerprint
ARG MODEL_URL=https://raw.githubusercontent.com/alnsyryfwzy55-source/digital-smile-design-ai/main/docs/teeth_unet_v2.onnx
ARG MODEL_SHA256=7f33b3a876054c264e0554acf2f81b4937d126972cbfddaca6bedfd26ca9e483
RUN mkdir -p /app/model && curl -fsSL "$MODEL_URL" -o /app/model/teeth.onnx \
    && echo "$MODEL_SHA256  /app/model/teeth.onnx" | sha256sum -c - \
    && chown -R app /app/model

USER app
ENV HOST=0.0.0.0 PORT=7860 DSD_MODEL=/app/model/teeth.onnx \
    DSD_ORIGINS=https://alnsyryfwzy55-source.github.io PYTHONUNBUFFERED=1
EXPOSE 7860
HEALTHCHECK CMD curl -fs http://localhost:7860/api/health || exit 1
CMD ["python", "server/http_api.py"]
