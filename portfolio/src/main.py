import sys
import os
import uvicorn
import logging
from fastapi import FastAPI
from portfolio.src.api import app
from portfolio.src.startup import on_startup
from portfolio.src.sync import start_sync_tasks
from portfolio.src.redis_subscriber import start_redis_subscriptions

# Configuration basique du log
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Attacher événements de démarrage
    app.add_event_handler("startup", on_startup)

    # Lancer Uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info"
    )
