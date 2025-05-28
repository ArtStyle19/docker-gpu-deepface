#!/bin/sh
echo "Starting DeepFace API..."
exec gunicorn --workers=1 --preload --timeout=7200 --bind=0.0.0.0:5000 --log-level=debug --access-logfile=- "app:create_app()"
# exec gunicorn --workers=2 --worker-class=gthread --threads=2 --timeout=7200 --bind=0.0.0.0:5000 --log-level=debug --access-logfile=- "app:create_app()"
echo "DeepFace API started successfully."
