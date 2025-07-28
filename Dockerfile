# Stage 1: Builder
# This stage will download all dependencies and models.
FROM python:3.11-slim AS builder

WORKDIR /app

# Create and populate requirements.txt inside the container.
RUN echo "torch --index-url https://download.pytorch.org/whl/cpu" > requirements.txt && \
    echo "transformers" >> requirements.txt && \
    echo "sentence-transformers" >> requirements.txt && \
    echo "faiss-cpu" >> requirements.txt && \
    echo "pymupdf" >> requirements.txt && \
    echo "sentencepiece" >> requirements.txt && \
    echo "numpy<2" >> requirements.txt && \
    echo "scikit-learn" >> requirements.txt

# Install all Python packages.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the setup script and run it to download the models into the builder stage.
COPY setup_project.py .
RUN python setup_project.py


# Stage 2: Final Image
# This stage will be the final, lean application image.
FROM python:3.11-slim

WORKDIR /app

# Copy requirements.txt and install packages again for the final image.
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code from the host.
COPY main.py .

# Copy the downloaded models from the builder stage into the final image.
# This is the key step that ensures the models are always present.
COPY --from=builder /app/models ./models/

# Add this instruction to ensure the models directory is preserved
# even when the parent directory is mounted over.
VOLUME /app/models

# Define the entrypoint for the container.
ENTRYPOINT ["python", "main.py"]

# Define a default command (a default collection to run if none is specified).
CMD ["Collection 1"]
