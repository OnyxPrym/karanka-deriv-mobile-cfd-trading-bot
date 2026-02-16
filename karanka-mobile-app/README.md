# KARANKA MULTIVERSE ALGO AI - Deriv Trading Bot

![Version](https://img.shields.io/badge/version-2.0.0-gold)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🏆 Overview

Advanced automated trading bot for Deriv.com featuring a sophisticated 80%+ win rate strategy. The bot uses multiple market analysis engines to detect optimal entry points across 30+ CFD and synthetic markets.

### Key Features
- **Market State Engine** - Detects trend/range/breakout conditions
- **Continuation Strategy (SWAPPED)** - SELL in uptrends, BUY in downtrends
- **Quasimodo Strategy (SWAPPED)** - BUY from bearish patterns, SELL from bullish
- **Smart Strategy Selector** - Chooses best strategy for current market
- **2 Pip Retest** - Precision entry validation
- **HTF Structure** - Higher timeframe confirmation mandatory
- **Trailing Stop Loss** - 30% activation, locks 85% of profits
- **Trade Tracking** - Real-time monitoring with auto-close detection
- **Web UI** - Full control panel with live updates

## 📋 Prerequisites

- Python 3.11+
- Deriv.com account with API token
- GitHub account (for deployment)
- Render.com account (for hosting)

## 🚀 Quick Start

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/karanka-mobile-app.git
cd karanka-mobile-app
