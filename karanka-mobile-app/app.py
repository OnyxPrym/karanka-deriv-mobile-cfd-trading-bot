#!/usr/bin/env python3
"""
KARANKA MULTIVERSE ALGO AI TRADER - DERIV CFD WEBAPP
Mobile-optimized web application for Deriv CFD trading
"""

import os
import json
import time
import threading
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from deriv_client import DerivClient
from trading_engine import KarankaTradingEngine
from config import Config
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['SESSION_TYPE'] = 'filesystem'
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=60, ping_interval=25)

# Store active trading sessions
active_sessions = {}
session_locks = {}

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/connect', methods=['POST'])
def connect():
    """Connect to Deriv with API token"""
    try:
        data = request.json
        api_token = data.get('api_token')
        app_id = data.get('app_id', '1089')
        
        if not api_token:
            return jsonify({'success': False, 'error': 'API token required'})
        
        # Create Deriv client
        client = DerivClient(api_token, app_id)
        
        # Connect and authorize
        success, message, accounts = client.connect()
        
        if success and accounts:
            # Store client in session (but not the actual client object)
            session_id = secrets.token_hex(16)
            session['session_id'] = session_id
            session['api_token'] = api_token
            session['app_id'] = app_id
            
            # Store in active sessions
            active_sessions[session_id] = {
                'client': client,
                'connected': True,
                'accounts': accounts,
                'selected_account': None,
                'trading_engine': None,
                'running': False
            }
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'accounts': accounts,
                'message': message
            })
        else:
            return jsonify({'success': False, 'error': message})
            
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/select-account', methods=['POST'])
def select_account():
    """Select which account to trade on"""
    try:
        data = request.json
        session_id = data.get('session_id')
        account_id = data.get('account_id')
        account_type = data.get('account_type')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Invalid session'})
        
        session_data = active_sessions[session_id]
        client = session_data['client']
        
        # Switch to selected account
        success, message = client.select_account(account_id, account_type)
        
        if success:
            session_data['selected_account'] = {
                'id': account_id,
                'type': account_type
            }
            
            # Get account info
            balance = client.get_balance()
            
            # Initialize trading engine with settings
            settings = {
                'dry_run': data.get('dry_run', True),
                'fixed_amount': float(data.get('fixed_amount', 1.0)),  # USD amount
                'max_concurrent_trades': int(data.get('max_concurrent_trades', 5)),
                'max_daily_trades': int(data.get('max_daily_trades', 25)),
                'enabled_symbols': data.get('enabled_symbols', Config.DEFAULT_SYMBOLS),
                'execution_threshold': int(data.get('execution_threshold', 65)),
                'max_signal_age': int(data.get('max_signal_age', 20))
            }
            
            # Create trading engine
            engine = KarankaTradingEngine(settings, client)
            session_data['trading_engine'] = engine
            
            return jsonify({
                'success': True,
                'balance': balance,
                'account_info': client.account_info
            })
        else:
            return jsonify({'success': False, 'error': message})
            
    except Exception as e:
        logger.error(f"Account selection error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start-trading', methods=['POST'])
def start_trading():
    """Start the trading engine"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Invalid session'})
        
        session_data = active_sessions[session_id]
        
        if not session_data['trading_engine']:
            return jsonify({'success': False, 'error': 'No trading engine initialized'})
        
        # Update settings if provided
        if data.get('settings'):
            settings = data.get('settings')
            engine = session_data['trading_engine']
            engine.update_settings(settings)
        
        # Start trading in a background thread
        def run_trading():
            try:
                session_data['trading_engine'].start_trading()
            except Exception as e:
                logger.error(f"Trading error: {str(e)}")
                socketio.emit('trading_error', {'error': str(e)}, room=session_id)
        
        if not session_data.get('running', False):
            thread = threading.Thread(target=run_trading, daemon=True)
            thread.start()
            session_data['running'] = True
            session_data['thread'] = thread
            
            # Start status update thread
            def status_updater():
                while session_data.get('running', False):
                    try:
                        if session_data['trading_engine']:
                            status = session_data['trading_engine'].get_status()
                            signals = session_data['trading_engine'].get_latest_signals()
                            socketio.emit('status_update', {
                                'status': status,
                                'signals': signals
                            }, room=session_id)
                    except Exception as e:
                        logger.error(f"Status update error: {str(e)}")
                    time.sleep(2)
            
            updater_thread = threading.Thread(target=status_updater, daemon=True)
            updater_thread.start()
            session_data['updater_thread'] = updater_thread
        
        return jsonify({'success': True, 'message': 'Trading started'})
        
    except Exception as e:
        logger.error(f"Start trading error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop-trading', methods=['POST'])
def stop_trading():
    """Stop the trading engine"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Invalid session'})
        
        session_data = active_sessions[session_id]
        
        if session_data.get('trading_engine'):
            session_data['trading_engine'].stop_trading()
        
        session_data['running'] = False
        
        return jsonify({'success': True, 'message': 'Trading stopped'})
        
    except Exception as e:
        logger.error(f"Stop trading error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get-status', methods=['POST'])
def get_status():
    """Get current trading status"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({'success': False, 'error': 'Invalid session'})
        
        session_data = active_sessions[session_id]
        
        if session_data.get('trading_engine'):
            status = session_data['trading_engine'].get_status()
            signals = session_data['trading_engine'].get_latest_signals()
            return jsonify({
                'success': True,
                'status': status,
                'signals': signals,
                'running': session_data.get('running', False)
            })
        else:
            return jsonify({
                'success': True,
                'status': {'connected': True, 'running': False},
                'signals': []
            })
            
    except Exception as e:
        logger.error(f"Get status error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get-symbols', methods=['GET'])
def get_symbols():
    """Get available trading symbols"""
    return jsonify({
        'success': True,
        'symbols': Config.ALL_SYMBOLS,
        'categories': Config.SYMBOL_CATEGORIES
    })

@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    """Disconnect from Deriv"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id and session_id in active_sessions:
            # Stop trading if running
            if active_sessions[session_id].get('running'):
                active_sessions[session_id]['trading_engine'].stop_trading()
            
            # Disconnect client
            if active_sessions[session_id].get('client'):
                active_sessions[session_id]['client'].disconnect()
            
            # Remove session
            del active_sessions[session_id]
        
        return jsonify({'success': True, 'message': 'Disconnected'})
        
    except Exception as e:
        logger.error(f"Disconnect error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    logger.info(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    logger.info(f"Client disconnected: {request.sid}")

# Keep-alive endpoint to prevent sleeping on Render
@app.route('/ping')
def ping():
    """Ping endpoint to keep the service alive"""
    return jsonify({'status': 'alive', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
