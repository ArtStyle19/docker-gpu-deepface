#!/bin/sh
echo "Starting DeepFace API..."
exec gunicorn --workers=1 --timeout=7200 --bind=0.0.0.0:5000 --log-level=debug --access-logfile=- "app:create_app()"
echo "DeepFace API started successfully."
