# =============================================================================
# RADAR Unified Docker Image
# Builds the full system (Angular frontend + FastAPI backend) from source.
#
# Usage (from S-RADAR-API root):
#   docker build -f docker/Dockerfile -t radar-unified .
#   docker run -d -p 80:80 --name radar radar-unified
# =============================================================================

# =============================================================================
# Stage 1: Clone & build Angular frontend
# =============================================================================
FROM node:20-alpine AS frontend-build

RUN apk add --no-cache git

WORKDIR /frontend

# Clone frontend repo (pinned to main — change to a tag for reproducibility)
RUN git clone --depth 1 https://github.com/ari-dasci/S-RADAR-frontend.git .

# Install dependencies and build for production
RUN npm ci --prefer-offline
RUN npm run build-prod


# =============================================================================
# Stage 2: Production image — Python + Nginx
# =============================================================================
FROM python:3.10-slim

# Install nginx + git in one layer, clean up apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends nginx git && \
    rm -rf /var/lib/apt/lists/*

# --------------- Backend setup ---------------
WORKDIR /app/backend

# Copy and install Python dependencies first (best layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code from build context (S-RADAR-API repo)
COPY main.py ./
COPY __init__.py ./
COPY routers/ ./routers/

# If RADAR directory exists locally, use it; otherwise clone from GitHub
COPY RADAR/ ./RADAR/
RUN if [ ! -d "RADAR" ] || [ -z "$(ls -A RADAR 2>/dev/null)" ]; then \
        git clone --depth 1 https://github.com/ari-dasci/S-RADAR.git /tmp/S-RADAR && \
        mv /tmp/S-RADAR/RADAR ./RADAR && \
        rm -rf /tmp/S-RADAR; \
    fi

# Remove git (no longer needed in production image)
RUN apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# --------------- Frontend setup ---------------
# Remove default nginx config
RUN rm -f /etc/nginx/sites-enabled/default /etc/nginx/conf.d/default.conf

# Copy custom nginx config
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# Copy compiled Angular assets from Stage 1
COPY --from=frontend-build /frontend/dist/s-adl-frontend/browser /usr/share/nginx/html

# --------------- Startup ---------------
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose port 80 (nginx serves frontend + proxies API)
EXPOSE 80

ENV PYTHONUNBUFFERED=1

CMD ["/app/start.sh"]
