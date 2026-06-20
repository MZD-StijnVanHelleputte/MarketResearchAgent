import uvicorn
from config.logging import configure_logging

configure_logging()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
