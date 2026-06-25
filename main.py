import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import router as chat_router
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sugar_main")

app = FastAPI(
    title="SUGAR API",
    description="Systematic Utility for Guidance, Automation, and Research - Backend",
    version="1.0.0"
)

# Robust CORS for Cloud Deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routes
app.include_router(chat_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "online", "system": "SUGAR Core"}

if __name__ == "__main__":
    # Standard Uvicorn startup
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
