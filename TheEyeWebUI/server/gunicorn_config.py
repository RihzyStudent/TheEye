# Gunicorn configuration optimized for Render free tier (500MB RAM)
import multiprocessing
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5001)}"
backlog = 64

# Worker processes - KEEP AT 1 FOR LOW MEMORY!
workers = 1  # Only 1 worker to minimize memory usage
worker_class = 'sync'
worker_connections = 10
timeout = 180  # 3 minutes for FITS processing
keepalive = 2

# Memory management
max_requests = 50  # Restart worker after 50 requests to prevent memory leaks
max_requests_jitter = 10
preload_app = True  # Load app before forking workers (saves memory)

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = 'theeye-backend'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Memory optimization
def pre_fork(server, worker):
    """Called just before a worker is forked"""
    import gc
    gc.collect()

def post_fork(server, worker):
    """Called just after a worker has been forked"""
    import gc
    gc.collect()

def worker_exit(server, worker):
    """Called when a worker exits"""
    import gc
    gc.collect()

