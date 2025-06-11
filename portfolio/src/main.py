import uvicorn
from portfolio.src.api import app
from portfolio.src.startup import on_startup

@app.on_event("startup")
async def _run_startup():
    await on_startup()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Configuration du logging pour réduire la verbosité
    import logging
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info",
        access_log=False  # Désactiver les logs d'accès pour les health checks
    )
