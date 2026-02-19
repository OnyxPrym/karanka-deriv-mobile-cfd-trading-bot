#!/usr/bin/env python3
"""
Configuration settings for Karanka Deriv Webapp
"""

class Config:
    """Application configuration"""
    
    # Default symbols for trading
    DEFAULT_SYMBOLS = [
        "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD",
        "US30", "USTEC", "AUDUSD", "BTCUSD", "NZDUSD",
        "USDCHF", "USDCAD", "EURGBP", "EURJPY"
    ]
    
    # All available symbols by category
    ALL_SYMBOLS = {
        "EURUSD": "EUR/USD",
        "GBPUSD": "GBP/USD",
        "USDJPY": "USD/JPY",
        "AUDUSD": "AUD/USD",
        "USDCAD": "USD/CAD",
        "USDCHF": "USD/CHF",
        "NZDUSD": "NZD/USD",
        "EURGBP": "EUR/GBP",
        "EURJPY": "EUR/JPY",
        "GBPJPY": "GBP/JPY",
        "AUDJPY": "AUD/JPY",
        "CHFJPY": "CHF/JPY",
        "EURAUD": "EUR/AUD",
        "GBPAUD": "GBP/AUD",
        "CADJPY": "CAD/JPY",
        "XAUUSD": "Gold",
        "XAGUSD": "Silver",
        "US30": "Dow Jones",
        "USTEC": "Nasdaq",
        "US100": "US 100",
        "BTCUSD": "Bitcoin"
    }
    
    # Symbol categories for UI
    SYMBOL_CATEGORIES = {
        "FOREX MAJORS": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"],
        "FOREX CROSSES": ["EURGBP", "EURJPY", "CHFJPY", "GBPJPY", "AUDJPY", "EURAUD", "GBPAUD", "CADJPY"],
        "COMMODITIES": ["XAUUSD", "XAGUSD"],
        "INDICES": ["US30", "USTEC", "US100"],
        "CRYPTO": ["BTCUSD"]
    }
    
    # Default settings
    DEFAULT_SETTINGS = {
        'dry_run': True,
        'fixed_amount': 1.0,
        'max_concurrent_trades': 5,
        'max_daily_trades': 25,
        'execution_threshold': 65,
        'max_signal_age': 20
    }
