# ============================================
# TRAFFIC VISION AI - CONFIGURATION
# ============================================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===== PROJECT PATHS =====
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads' / 'videos'
MODELS_FOLDER = BASE_DIR / 'models'
DATABASE_FOLDER = BASE_DIR / 'database'

# Create required directories
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
DATABASE_FOLDER.mkdir(parents=True, exist_ok=True)

# ===== DATABASE CONFIGURATION =====
class DatabaseConfig:
    # SQLite (default, development)
    SQLITE_DB = f"sqlite:///{DATABASE_FOLDER / 'traffic_system.db'}"
    
    # PostgreSQL (production)
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'traffic_user')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'traffic_password')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'traffic_db')
    
    POSTGRES_URL = f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
    
    # MongoDB (alternative)
    MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/traffic_db')
    
    # Select database
    DATABASE = os.getenv('DATABASE_TYPE', 'sqlite')  # 'sqlite', 'postgres', or 'mongodb'
    
    @staticmethod
    def get_database_url():
        if DatabaseConfig.DATABASE == 'postgres':
            return DatabaseConfig.POSTGRES_URL
        elif DatabaseConfig.DATABASE == 'mongodb':
            return DatabaseConfig.MONGODB_URL
        else:
            return DatabaseConfig.SQLITE_DB

# ===== YOLO CONFIGURATION =====
class YOLOConfig:
    # Model selection
    VEHICLE_MODEL = os.getenv('YOLO_VEHICLE_MODEL', 'yolov8n.pt')  # nano, small, medium, large
    AMBULANCE_MODEL = os.getenv('YOLO_AMBULANCE_MODEL', 'ambulance_model.pt')
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', '0.45'))
    
    # Vehicle class IDs (YOLOv8 COCO)
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }
    
    # Model paths
    VEHICLE_MODEL_PATH = MODELS_FOLDER / VEHICLE_MODEL
    AMBULANCE_MODEL_PATH = MODELS_FOLDER / AMBULANCE_MODEL
    
    # Device
    DEVICE = os.getenv('DEVICE', 'cpu')  # 'cpu' or 'cuda'
    
    # Inference size
    INFERENCE_SIZE = int(os.getenv('INFERENCE_SIZE', '640'))

# ===== VIDEO PROCESSING CONFIGURATION =====
class VideoProcessingConfig:
    # Frame processing
    FRAME_SKIP = int(os.getenv('FRAME_SKIP', '5'))  # Process every Nth frame
    TARGET_RESOLUTION = tuple(map(int, os.getenv('TARGET_RESOLUTION', '640,480').split(',')))
    
    # Video limits
    MAX_VIDEO_SIZE = int(os.getenv('MAX_VIDEO_SIZE', '500')) * 1024 * 1024  # MB to bytes
    ALLOWED_VIDEO_FORMATS = ['mp4', 'avi', 'flv', 'mov', 'mkv']
    
    # Processing
    ASYNC_PROCESSING = bool(os.getenv('ASYNC_PROCESSING', 'true').lower() == 'true')
    BATCH_PROCESSING = bool(os.getenv('BATCH_PROCESSING', 'false').lower() == 'true')
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))

# ===== TRAFFIC CONTROL CONFIGURATION =====
class TrafficControlConfig:
    # Signal timing (seconds)
    BASE_TIME = float(os.getenv('BASE_TIME', '10'))  # Base green signal duration
    MIN_TIME = float(os.getenv('MIN_TIME', '5'))     # Minimum signal duration
    MAX_TIME = float(os.getenv('MAX_TIME', '60'))    # Maximum signal duration
    AMBULANCE_TIME = float(os.getenv('AMBULANCE_TIME', '15'))
    
    # Timing calculation
    TIME_PER_VEHICLE = float(os.getenv('TIME_PER_VEHICLE', '0.5'))  # seconds per vehicle
    
    # Density thresholds
    LOW_DENSITY = int(os.getenv('LOW_DENSITY', '10'))
    MEDIUM_DENSITY = int(os.getenv('MEDIUM_DENSITY', '30'))
    HIGH_DENSITY = int(os.getenv('HIGH_DENSITY', '50'))
    
    # Lanes
    NUM_LANES = int(os.getenv('NUM_LANES', '4'))
    
    @staticmethod
    def calculate_signal_duration(vehicle_count):
        """Calculate green signal duration based on vehicle density"""
        if vehicle_count == 0:
            return TrafficControlConfig.MIN_TIME
        
        duration = TrafficControlConfig.BASE_TIME + (vehicle_count * TrafficControlConfig.TIME_PER_VEHICLE)
        return min(duration, TrafficControlConfig.MAX_TIME)

# ===== FLASK CONFIGURATION =====
class FlaskConfig:
    # Server
    DEBUG = bool(os.getenv('FLASK_DEBUG', 'false').lower() == 'true')
    TESTING = bool(os.getenv('FLASK_TESTING', 'false').lower() == 'true')
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', '5000'))
    
    # Database
    SQLALCHEMY_DATABASE_URI = DatabaseConfig.get_database_url()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File uploads
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_UPLOAD_SIZE', '500')) * 1024 * 1024
    UPLOAD_FOLDER = str(UPLOAD_FOLDER)
    
    # Session
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this')
    SESSION_COOKIE_SECURE = bool(os.getenv('SESSION_COOKIE_SECURE', 'false').lower() == 'true')
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

# ===== FASTAPI CONFIGURATION =====
class FastAPIConfig:
    # Server
    DEBUG = bool(os.getenv('FASTAPI_DEBUG', 'false').lower() == 'true')
    HOST = os.getenv('FASTAPI_HOST', '0.0.0.0')
    PORT = int(os.getenv('FASTAPI_PORT', '8000'))
    
    # Database
    DATABASE_URL = DatabaseConfig.get_database_url()
    
    # File uploads
    MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', '500')) * 1024 * 1024
    
    # API
    API_SECRET_KEY = os.getenv('FASTAPI_SECRET_KEY', 'your-secret-key-change-this')
    API_TITLE = 'Traffic Vision AI'
    API_VERSION = '1.0.0'
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

# ===== DJANGO CONFIGURATION =====
class DjangoConfig:
    # Security
    SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'your-secret-key-change-this')
    DEBUG = bool(os.getenv('DJANGO_DEBUG', 'false').lower() == 'true')
    ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
    
    # Database
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': str(DATABASE_FOLDER / 'db.sqlite3'),
        }
    }
    
    # Celery
    CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379')
    CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379')
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_TIMEZONE = 'UTC'

# ===== AMBULANCE DETECTION CONFIGURATION =====
class AmbulanceDetectionConfig:
    # Color range for ambulance detection (HSV)
    LOWER_BLUE = tuple(map(int, os.getenv('LOWER_BLUE', '100,100,100').split(',')))
    UPPER_BLUE = tuple(map(int, os.getenv('UPPER_BLUE', '130,255,255').split(',')))
    
    # Detection threshold
    MIN_AREA = int(os.getenv('AMBULANCE_MIN_AREA', '1000'))
    
    # Sound alert
    ENABLE_SOUND_ALERT = bool(os.getenv('ENABLE_SOUND_ALERT', 'false').lower() == 'true')

# ===== ANALYTICS CONFIGURATION =====
class AnalyticsConfig:
    # Traffic reduction baseline (traditional system)
    BASELINE_SIGNAL_TIME = float(os.getenv('BASELINE_SIGNAL_TIME', '30'))
    
    # Metrics to track
    TRACK_VEHICLE_COUNT = True
    TRACK_WAIT_TIME = True
    TRACK_THROUGHPUT = True
    TRACK_EMISSIONS = True
    
    # Database retention
    RETENTION_DAYS = int(os.getenv('RETENTION_DAYS', '30'))
    ARCHIVE_OLD_DATA = bool(os.getenv('ARCHIVE_OLD_DATA', 'true').lower() == 'true')

# ===== LOGGING CONFIGURATION =====
class LoggingConfig:
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / 'logs' / 'traffic_system.log'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @staticmethod
    def setup_logging():
        import logging
        LOG_FILE_PATH = LoggingConfig.LOG_FILE
        LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=LoggingConfig.LOG_LEVEL,
            format=LoggingConfig.LOG_FORMAT,
            handlers=[
                logging.FileHandler(LOG_FILE_PATH),
                logging.StreamHandler()
            ]
        )

# ===== SECURITY CONFIGURATION =====
class SecurityConfig:
    # HTTPS
    FORCE_HTTPS = bool(os.getenv('FORCE_HTTPS', 'false').lower() == 'true')
    
    # JWT
    JWT_SECRET = os.getenv('JWT_SECRET', 'your-jwt-secret')
    JWT_ALGORITHM = 'HS256'
    JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', '24'))
    
    # Password
    PASSWORD_MIN_LENGTH = int(os.getenv('PASSWORD_MIN_LENGTH', '8'))
    
    # Rate limiting
    RATE_LIMIT_ENABLED = bool(os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true')
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', '100'))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', '3600'))

# ===== ENVIRONMENT SELECTOR =====
ENV = os.getenv('ENVIRONMENT', 'development').lower()

if ENV == 'production':
    DATABASE_URL = DatabaseConfig.POSTGRES_URL
    DEBUG = False
elif ENV == 'testing':
    DEBUG = True
else:  # development
    DATABASE_URL = DatabaseConfig.SQLITE_DB
    DEBUG = True

# ===== CONFIGURATION SUMMARY =====
CONFIG_SUMMARY = {
    'Environment': ENV,
    'Database': DatabaseConfig.DATABASE,
    'YOLO Model': YOLOConfig.VEHICLE_MODEL,
    'Device': YOLOConfig.DEVICE,
    'Frame Skip': VideoProcessingConfig.FRAME_SKIP,
    'Base Signal Time': TrafficControlConfig.BASE_TIME,
    'Async Processing': VideoProcessingConfig.ASYNC_PROCESSING,
    'Debug Mode': DEBUG,
}

def print_config():
    """Print configuration summary"""
    print('\n' + '='*50)
    print('TRAFFIC VISION AI - CONFIGURATION')
    print('='*50)
    for key, value in CONFIG_SUMMARY.items():
        print(f'{key}: {value}')
    print('='*50 + '\n')

def get_flask_config():
    """Get Flask configuration object"""
    return FlaskConfig

def get_fastapi_config():
    """Get FastAPI configuration"""
    return FastAPIConfig

def get_django_config():
    """Get Django configuration"""
    return DjangoConfig

def get_yolo_config():
    """Get YOLO configuration"""
    return YOLOConfig

def get_traffic_control_config():
    """Get traffic control configuration"""
    return TrafficControlConfig

def get_video_processing_config():
    """Get video processing configuration"""
    return VideoProcessingConfig

# ===== QUICK ACCESS =====
# Use these for easy configuration access
flask_config = FlaskConfig()
fastapi_config = FastAPIConfig()
django_config = DjangoConfig()
yolo_config = YOLOConfig()
traffic_control = TrafficControlConfig()
video_processing = VideoProcessingConfig()
ambulance_detection = AmbulanceDetectionConfig()
analytics_config = AnalyticsConfig()
security_config = SecurityConfig()
logging_config = LoggingConfig()

if __name__ == '__main__':
    print_config()
    
    print("\n📋 DETAILED CONFIGURATION BREAKDOWN:\n")
    
    print("🗄️  DATABASE CONFIGURATION:")
    print(f"   Database Type: {DatabaseConfig.DATABASE}")
    print(f"   Database URL: {DatabaseConfig.get_database_url()}")
    print(f"   SQLite Path: {DatabaseConfig.SQLITE_DB}")
    
    print("\n🤖 YOLO CONFIGURATION:")
    print(f"   Vehicle Model: {YOLOConfig.VEHICLE_MODEL}")
    print(f"   Confidence Threshold: {YOLOConfig.CONFIDENCE_THRESHOLD}")
    print(f"   IOU Threshold: {YOLOConfig.IOU_THRESHOLD}")
    print(f"   Device: {YOLOConfig.DEVICE}")
    print(f"   Inference Size: {YOLOConfig.INFERENCE_SIZE}")
    
    print("\n🎥 VIDEO PROCESSING CONFIGURATION:")
    print(f"   Frame Skip: {VideoProcessingConfig.FRAME_SKIP}")
    print(f"   Target Resolution: {VideoProcessingConfig.TARGET_RESOLUTION}")
    print(f"   Max Video Size: {VideoProcessingConfig.MAX_VIDEO_SIZE / (1024*1024):.0f}MB")
    print(f"   Async Processing: {VideoProcessingConfig.ASYNC_PROCESSING}")
    print(f"   Max Workers: {VideoProcessingConfig.MAX_WORKERS}")
    
    print("\n🚦 TRAFFIC CONTROL CONFIGURATION:")
    print(f"   Base Signal Time: {TrafficControlConfig.BASE_TIME}s")
    print(f"   Min Signal Time: {TrafficControlConfig.MIN_TIME}s")
    print(f"   Max Signal Time: {TrafficControlConfig.MAX_TIME}s")
    print(f"   Time Per Vehicle: {TrafficControlConfig.TIME_PER_VEHICLE}s")
    print(f"   Ambulance Time: {TrafficControlConfig.AMBULANCE_TIME}s")
    print(f"   Number of Lanes: {TrafficControlConfig.NUM_LANES}")
    
    print("\n🌐 FLASK CONFIGURATION:")
    print(f"   Debug: {FlaskConfig.DEBUG}")
    print(f"   Host: {FlaskConfig.HOST}")
    print(f"   Port: {FlaskConfig.PORT}")
    print(f"   Max Upload Size: {FlaskConfig.MAX_CONTENT_LENGTH / (1024*1024):.0f}MB")
    
    print("\n⚡ FASTAPI CONFIGURATION:")
    print(f"   Debug: {FastAPIConfig.DEBUG}")
    print(f"   Host: {FastAPIConfig.HOST}")
    print(f"   Port: {FastAPIConfig.PORT}")
    
    print("\n🚨 AMBULANCE DETECTION CONFIGURATION:")
    print(f"   Lower Blue Range: {AmbulanceDetectionConfig.LOWER_BLUE}")
    print(f"   Upper Blue Range: {AmbulanceDetectionConfig.UPPER_BLUE}")
    print(f"   Min Area: {AmbulanceDetectionConfig.MIN_AREA}")
    print(f"   Sound Alert: {AmbulanceDetectionConfig.ENABLE_SOUND_ALERT}")
    
    print("\n📊 ANALYTICS CONFIGURATION:")
    print(f"   Baseline Signal Time: {AnalyticsConfig.BASELINE_SIGNAL_TIME}s")
    print(f"   Data Retention: {AnalyticsConfig.RETENTION_DAYS} days")
    print(f"   Archive Old Data: {AnalyticsConfig.ARCHIVE_OLD_DATA}")
    
    print("\n🔐 SECURITY CONFIGURATION:")
    print(f"   Force HTTPS: {SecurityConfig.FORCE_HTTPS}")
    print(f"   JWT Algorithm: {SecurityConfig.JWT_ALGORITHM}")
    print(f"   Rate Limiting: {SecurityConfig.RATE_LIMIT_ENABLED}")
    print(f"   Rate Limit Requests: {SecurityConfig.RATE_LIMIT_REQUESTS} per {SecurityConfig.RATE_LIMIT_WINDOW}s")
    
    print("\n📁 PROJECT PATHS:")
    print(f"   Base Directory: {BASE_DIR}")
    print(f"   Upload Folder: {UPLOAD_FOLDER}")
    print(f"   Models Folder: {MODELS_FOLDER}")
    print(f"   Database Folder: {DATABASE_FOLDER}")
    
    print("\n✅ Configuration loaded successfully!\n")
