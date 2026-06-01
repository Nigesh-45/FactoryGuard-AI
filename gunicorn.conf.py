import os
import multiprocessing

# Gunicorn production serving configurations
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("GUNICORN_WORKERS", 2))
threads = int(os.environ.get("GUNICORN_THREADS", 2))

# Timeout to prevent worker recycling during longer SHAP explanations
timeout = 120

# Logging parameters
accesslog = "-"
errorlog = "-"
loglevel = "info"
