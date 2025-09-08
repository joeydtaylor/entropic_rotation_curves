# Reproducible image for EGR rotation-curves (full plots + MCMC capable)
ARG PYTHON_VERSION=3.12-slim
FROM python:${PYTHON_VERSION} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

# Install Python deps first (cache-friendly)
COPY requirements.txt /work/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy repo and install package
COPY . /work
RUN pip install -e .

# Create default out dir
RUN mkdir -p /work/out

# Handy entrypoints:
#   egr-rc           -> original CLI (one page)
#   egr-rc-run-all   -> all pages, plots+RAR+MCMC on
#   egr-rc-selftest  -> synthetic quick run
# Default CMD: selftest (so image works out-of-the-box)
CMD ["egr-rc-selftest", "--outdir", "/work/out"]
