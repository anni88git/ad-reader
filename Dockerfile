# Hugging Face Spaces (Docker SDK) image for the Streamlit app. Also runs anywhere Docker does.
FROM python:3.12-slim

# Linux libraries OpenCV / OCR need (listed in packages.txt)
COPY packages.txt /tmp/packages.txt
RUN apt-get update \
 && xargs -a /tmp/packages.txt apt-get install -y --no-install-recommends \
 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000; the app writes jobs, the ops DB and downloaded models under its folder
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user --upgrade pip && pip install --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# CORS / XSRF off: needed for st.file_uploader inside the Spaces iframe (Hugging Face's Streamlit guidance)
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
