# Configuration settings
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'karanka-secret-key-2025')
    DEBUG = False
    TESTING = False
    
    # Deriv API settings
    DERIV_APP_ID = os.environ.get('DERIV_APP_ID', '1089')  # Default Deriv app ID
    
    # Trading defaults
    DEFAULT_MAX_DAILY_TRADES = 25
    DEFAULT_MAX_HOURLY_TRADES = 6
    DEFAULT_MIN_SCORE = 65
    DEFAULT_RETEST_TOLERANCE = 2
    DEFAULT_LOT_SIZE = 0.01
    
    # Session config
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True