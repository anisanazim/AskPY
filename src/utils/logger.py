import os
from loguru import logger
from src.config.settings import settings

# Create logs directory if it doesn't exist
os.makedirs(settings.logs_path, exist_ok=True)

# Configure logger
logger.add(
    os.path.join(settings.logs_path, "pythondocbot.log"),
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}"
)

def get_logger():
    return logger