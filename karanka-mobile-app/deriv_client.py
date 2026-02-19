#!/usr/bin/env python3
"""
Deriv API Client - Handles all Deriv API interactions
Supports multiple accounts (demo/real) and proper authentication
"""

import json
import time
import threading
import logging
import websocket
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DerivClient:
    """Deriv API Client with full account management"""
    
    def __init__(self, api_token: str, app_id: str = "1089"):
        self.api_token = api_token
        self.app_id = app_id
        self.ws = None
        self.connected = False
        self.authorized = False
        self.account_info = None
        self.accounts = []  # List of available accounts
        self.current_account = None
        self.active_symbols = []
        self.price_data = {}
        self.candle_data = {}
        self.pending_requests = {}
        self.request_id = 0
        self.subscriptions = set()
        
        # WebSocket thread
        self.ws_thread = None
        self.should_stop = False
        
        # Callbacks
        self.on_tick_callbacks = []
        self.on_candle_callbacks = []
        
    def connect(self) -> Tuple[bool, str, List[Dict]]:
        """
        Connect to Deriv WebSocket and authorize
        Returns: (success, message, accounts)
        """
        try:
            # Create WebSocket connection
            websocket_url = f"wss://ws.derivws.com/websockets/v3?app_id={self.app_id}&l=EN"
            
            self.ws = websocket.WebSocketApp(
                websocket_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Run WebSocket in thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.ws_thread.start()
            
            # Wait for connection
            timeout = time.time() + 10
            while time.time() < timeout and not self.connected:
                time.sleep(0.1)
            
            if not self.connected:
                return False, "Connection timeout", []
            
            # Authorize
            auth_success, auth_message = self._authorize()
            
            if not auth_success:
                return False, auth_message, []
            
            # Get available accounts
            accounts = self._get_accounts()
            
            return True, "Connected successfully", accounts
            
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False, str(e), []
    
    def _on_open(self, ws):
        """WebSocket opened"""
        logger.info("Deriv WebSocket connected")
        self.connected = True
    
    def _on_message(self, ws, message):
        """Handle incoming messages"""
        try:
            data = json.loads(message)
            
            # Handle ping
            if data.get("ping"):
                self.ws.send(json.dumps({"pong": data["ping"]}))
                return
            
            # Handle responses
            req_id = data.get("req_id")
            if req_id and req_id in self.pending_requests:
                self.pending_requests[req_id] = data
            
            # Handle authorization
            if "authorize" in data:
                self.authorized = True
                self.account_info = data["authorize"]
                logger.info(f"Authorized as: {self.account_info.get('email', 'Unknown')}")
            
            # Handle account list
            if "authorize" in data and "account_list" in data["authorize"]:
                self.accounts = data["authorize"]["account_list"]
            
            # Handle active symbols
            if "active_symbols" in data:
                self.active_symbols = data["active_symbols"]
            
            # Handle ticks
            if "tick" in data:
                symbol = data["tick"]["symbol"]
                self.price_data[symbol] = {
                    "time": data["tick"]["epoch"],
                    "quote": data["tick"]["quote"],
                    "ask": data["tick"].get("ask", data["tick"]["quote"]),
                    "bid": data["tick"].get("bid", data["tick"]["quote"])
                }
                
                # Trigger callbacks
                for callback in self.on_tick_callbacks:
                    try:
                        callback(symbol, self.price_data[symbol])
                    except:
                        pass
            
            # Handle candles
            if "ohlc" in data:
                symbol = data["ohlc"]["symbol"]
                if symbol not in self.candle_data:
                    self.candle_data[symbol] = []
                
                candle = {
                    "time": data["ohlc"]["epoch"],
                    "open": float(data["ohlc"]["open"]),
                    "high": float(data["ohlc"]["high"]),
                    "low": float(data["ohlc"]["low"]),
                    "close": float(data["ohlc"]["close"]),
                    "volume": data["ohlc"].get("volume", 0)
                }
                
                self.candle_data[symbol].append(candle)
                
                # Keep last 1000 candles
                if len(self.candle_data[symbol]) > 1000:
                    self.candle_data[symbol] = self.candle_data[symbol][-1000:]
                
                # Trigger callbacks
                for callback in self.on_candle_callbacks:
                    try:
                        callback(symbol, candle)
                    except:
                        pass
            
            # Handle buy/sell responses
            if "buy" in data:
                logger.info(f"Order placed: {data['buy']['contract_id']}")
            
            if "sell" in data:
                logger.info(f"Position closed: {data['sell']}")
            
        except Exception as e:
            logger.error(f"Message error: {str(e)}")
    
    def _on_error(self, ws, error):
        """WebSocket error"""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket closed"""
        logger.info("Deriv WebSocket closed")
        self.connected = False
        self.authorized = False
    
    def _send_request(self, request: Dict, timeout: int = 5) -> Optional[Dict]:
        """Send request and wait for response"""
        self.request_id += 1
        request["req_id"] = self.request_id
        self.pending_requests[self.request_id] = None
        
        try:
            self.ws.send(json.dumps(request))
        except:
            return None
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.pending_requests[self.request_id] is not None:
                response = self.pending_requests[self.request_id]
                del self.pending_requests[self.request_id]
                return response
            time.sleep(0.1)
        
        del self.pending_requests[self.request_id]
        return None
    
    def _authorize(self) -> Tuple[bool, str]:
        """Authorize with API token"""
        request = {
            "authorize": self.api_token
        }
        
        response = self._send_request(request, timeout=10)
        
        if response and "authorize" in response:
            self.authorized = True
            return True, "Authorized"
        elif response and "error" in response:
            return False, response["error"].get("message", "Authorization failed")
        else:
            return False, "No response"
    
    def _get_accounts(self) -> List[Dict]:
        """Get all available accounts (demo and real)"""
        accounts = []
        
        if not self.account_info:
            return accounts
        
        # Parse account list from authorize response
        account_list = self.account_info.get("account_list", [])
        
        for acc in account_list:
            accounts.append({
                "id": acc.get("loginid"),
                "type": "demo" if acc.get("is_virtual") else "real",
                "currency": acc.get("currency"),
                "balance": acc.get("balance", 0),
                "landing_company": acc.get("landing_company_name")
            })
        
        return accounts
    
    def select_account(self, account_id: str, account_type: str) -> Tuple[bool, str]:
        """Switch to selected account"""
        try:
            # For Deriv, we need to use a new token for each account
            # The token is account-specific, so we just need to ensure we're using the right token
            if account_type == "demo" and not self.account_info.get("is_virtual", False):
                logger.info("Using demo account token")
            elif account_type == "real" and self.account_info.get("is_virtual", False):
                logger.info("Using real account token")
            
            self.current_account = {
                "id": account_id,
                "type": account_type
            }
            
            return True, "Account selected"
            
        except Exception as e:
            logger.error(f"Account selection error: {str(e)}")
            return False, str(e)
    
    def get_balance(self) -> float:
        """Get current account balance"""
        if self.account_info:
            return self.account_info.get("balance", 0)
        return 0
    
    def subscribe_ticks(self, symbols: List[str]):
        """Subscribe to tick updates"""
        if not isinstance(symbols, list):
            symbols = [symbols]
        
        for symbol in symbols:
            if symbol in self.subscriptions:
                continue
                
            request = {
                "ticks": symbol,
                "subscribe": 1
            }
            try:
                self.ws.send(json.dumps(request))
                self.subscriptions.add(symbol)
                logger.info(f"Subscribed to {symbol} ticks")
            except Exception as e:
                logger.error(f"Tick subscription error: {e}")
    
    def subscribe_candles(self, symbol: str, granularity: int = 60):
        """Subscribe to candle updates"""
        request = {
            "ticks_history": symbol,
            "granularity": granularity,
            "style": "candles",
            "subscribe": 1
        }
        try:
            self.ws.send(json.dumps(request))
            logger.info(f"Subscribed to {symbol} candles")
        except Exception as e:
            logger.error(f"Candle subscription error: {e}")
    
    def get_candles(self, symbol: str, granularity: int = 60, count: int = 100) -> pd.DataFrame:
        """Get historical candles"""
        request = {
            "ticks_history": symbol,
            "granularity": granularity,
            "style": "candles",
            "count": count
        }
        
        response = self._send_request(request)
        
        if response and "candles" in response:
            candles = []
            for candle in response["candles"]:
                candles.append({
                    "time": candle["epoch"],
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": candle.get("volume", 0)
                })
            
            df = pd.DataFrame(candles)
            
            # Convert epoch to datetime
            if "time" in df.columns:
                df["datetime"] = pd.to_datetime(df["time"], unit="s")
            
            return df
        
        return pd.DataFrame()
    
    def place_order(self, symbol: str, contract_type: str, amount: float,
                   duration: int = 1, duration_unit: str = "m") -> Optional[Dict]:
        """
        Place a trade order
        
        Args:
            symbol: Trading symbol (e.g., "frxEURUSD")
            contract_type: "CALL" for buy, "PUT" for sell
            amount: Amount in USD
            duration: Contract duration
            duration_unit: "m" for minutes, "h" for hours, "d" for days
        """
        try:
            # Get current price
            if symbol not in self.price_data:
                logger.error(f"No price data for {symbol}")
                return None
            
            current_price = self.price_data[symbol].get("quote", 0)
            
            # Prepare order request
            request = {
                "buy": 1,
                "price": amount,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type.upper(),
                    "currency": "USD",
                    "duration": duration,
                    "duration_unit": duration_unit,
                    "symbol": symbol
                }
            }
            
            response = self._send_request(request, timeout=10)
            
            if response and "buy" in response:
                return {
                    "contract_id": response["buy"]["contract_id"],
                    "transaction_id": response["buy"]["transaction_id"],
                    "price": response["buy"]["price"],
                    "balance": response["buy"].get("balance_after", 0)
                }
            elif response and "error" in response:
                logger.error(f"Order error: {response['error']}")
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Order error: {str(e)}")
            return None
    
    def close_position(self, contract_id: str) -> bool:
        """Close an open position"""
        request = {
            "sell": contract_id,
            "price": 0  # Market price
        }
        
        response = self._send_request(request)
        
        return response is not None and "sell" in response
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        request = {
            "portfolio": 1
        }
        
        response = self._send_request(request)
        
        if response and "portfolio" in response:
            return response["portfolio"].get("contracts", [])
        
        return []
    
    def get_available_symbols(self, market_type: str = "all") -> List[Dict]:
        """Get available symbols for trading"""
        request = {
            "active_symbols": "brief",
            "product_type": "basic"
        }
        
        response = self._send_request(request)
        
        symbols = []
        if response and "active_symbols" in response:
            for symbol in response["active_symbols"]:
                if market_type == "all" or symbol.get("market") == market_type.upper():
                    symbols.append({
                        "symbol": symbol["symbol"],
                        "display_name": symbol["display_name"],
                        "market": symbol.get("market", ""),
                        "pip_size": symbol.get("pip_size", 4),
                        "min_contract": float(symbol.get("min_contract_size", 0.01)),
                        "max_contract": float(symbol.get("max_contract_size", 100))
                    })
        
        return symbols
    
    def register_tick_callback(self, callback):
        """Register callback for tick updates"""
        self.on_tick_callbacks.append(callback)
    
    def register_candle_callback(self, callback):
        """Register callback for candle updates"""
        self.on_candle_callbacks.append(callback)
    
    def disconnect(self):
        """Disconnect from Deriv"""
        self.should_stop = True
        if self.ws:
            self.ws.close()
