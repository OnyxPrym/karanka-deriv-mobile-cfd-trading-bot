# deriv_connector.py - Deriv API WebSocket Connection
import asyncio
import websockets
import json
import threading
import time
from datetime import datetime
import pandas as pd
import numpy as np

class DerivConnector:
    """
    Connects to Deriv API using user-provided token
    Handles WebSocket connection, authentication, and data streaming
    """
    
    def __init__(self, api_token, app_id=1089, account_type='demo'):
        self.api_token = api_token
        self.app_id = app_id
        self.account_type = account_type  # 'demo' or 'real'
        self.ws_url = "wss://ws.deriv.com/websockets/v3"
        self.connection = None
        self.connected = False
        self.authenticated = False
        self.user_data = {}
        self.balance = 0
        self.currency = "USD"
        self.loginid = ""
        self.email = ""
        self.symbol_info = {}
        self.price_cache = {}
        self.candle_data = {}
        self.tick_subscriptions = {}
        self.candle_subscriptions = {}
        self.message_queue = []
        self.last_message_time = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
    def connect(self):
        """Connect to Deriv WebSocket API"""
        try:
            # Run async connection in thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Start connection in background thread
            thread = threading.Thread(target=self._run_async_connect, args=(loop,), daemon=True)
            thread.start()
            
            # Wait for connection result (with timeout)
            timeout = 10
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.connected and self.authenticated:
                    return True, "Connected successfully"
                time.sleep(0.5)
            
            if self.connected and not self.authenticated:
                return False, "Authentication failed - check token"
            elif not self.connected:
                return False, "Connection timeout"
            
            return True, "Connected"
            
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def _run_async_connect(self, loop):
        """Run async connection in thread"""
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._async_connect())
    
    async def _async_connect(self):
        """Async connection establishment"""
        try:
            # Connect websocket
            self.connection = await websockets.connect(self.ws_url)
            
            # Authenticate
            auth_success = await self._authenticate()
            
            if auth_success:
                self.connected = True
                self.authenticated = True
                
                # Get account info
                await self._get_account_info()
                
                # Get available symbols
                await self._get_symbols()
                
                # Start message handler
                asyncio.create_task(self._handle_messages())
                
                # Ping to keep connection alive
                asyncio.create_task(self._ping_loop())
                
                print("✅ Deriv connected successfully")
            else:
                print("❌ Authentication failed")
                
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
    
    async def _authenticate(self):
        """Authenticate with API token"""
        try:
            auth_request = {
                "authorize": self.api_token
            }
            await self.connection.send(json.dumps(auth_request))
            response = await self.connection.recv()
            response_data = json.loads(response)
            
            if 'error' in response_data:
                print(f"Auth error: {response_data['error']['message']}")
                return False
            
            # Store user data
            if 'authorize' in response_data:
                auth_data = response_data['authorize']
                self.user_data = auth_data
                self.loginid = auth_data.get('loginid', '')
                self.email = auth_data.get('email', '')
                self.currency = auth_data.get('currency', 'USD')
                
                # Check account type
                if 'is_virtual' in auth_data:
                    self.account_type = 'demo' if auth_data['is_virtual'] else 'real'
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Auth exception: {e}")
            return False
    
    async def _get_account_info(self):
        """Get account balance and info"""
        try:
            balance_request = {
                "balance": 1,
                "subscribe": 1
            }
            await self.connection.send(json.dumps(balance_request))
            response = await self.connection.recv()
            response_data = json.loads(response)
            
            if 'balance' in response_data:
                balance_data = response_data['balance']
                self.balance = float(balance_data['balance'])
                self.currency = balance_data.get('currency', self.currency)
                
        except Exception as e:
            print(f"Balance error: {e}")
    
    async def _get_symbols(self):
        """Get available trading symbols"""
        try:
            symbols_request = {
                "active_symbols": "brief",
                "product_type": "basic"
            }
            await self.connection.send(json.dumps(symbols_request))
            response = await self.connection.recv()
            response_data = json.loads(response)
            
            if 'active_symbols' in response_data:
                for symbol in response_data['active_symbols']:
                    if symbol.get('exchange_is_open', 0) == 1:
                        self.symbol_info[symbol['symbol']] = {
                            'name': symbol.get('display_name', ''),
                            'market': symbol.get('market', ''),
                            'pip': symbol.get('pip', 0.0001),
                            'min_contract': symbol.get('min_contract_size', 0.01)
                        }
                        
        except Exception as e:
            print(f"Symbols error: {e}")
    
    async def _handle_messages(self):
        """Handle incoming messages"""
        while self.connected:
            try:
                message = await self.connection.recv()
                data = json.loads(message)
                
                # Store in queue for processing
                self.message_queue.append(data)
                
                # Process different message types
                await self._process_message(data)
                
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                self.connected = False
                break
            except Exception as e:
                print(f"Message handler error: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, data):
        """Process different message types"""
        
        # Tick data
        if 'tick' in data:
            tick = data['tick']
            symbol = tick['symbol']
            self.price_cache[symbol] = {
                'ask': float(tick.get('ask', tick['quote'])),
                'bid': float(tick.get('bid', tick['quote'])),
                'price': float(tick['quote']),
                'time': datetime.fromtimestamp(tick['epoch']),
                'symbol': symbol
            }
        
        # Candle data
        elif 'candle' in data:
            candle = data['candle']
            symbol = candle['symbol']
            
            # Extract granularity from subscription key
            # This would need proper tracking in production
            granularity = 60  # default
            
            cache_key = f"{symbol}_{granularity}"
            
            if cache_key not in self.candle_data:
                self.candle_data[cache_key] = []
            
            self.candle_data[cache_key].append({
                'time': datetime.fromtimestamp(candle['epoch']),
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': int(candle.get('volume', 0))
            })
            
            # Keep last 1000 candles
            if len(self.candle_data[cache_key]) > 1000:
                self.candle_data[cache_key] = self.candle_data[cache_key][-1000:]
        
        # Candle history response
        elif 'candles' in data:
            msg_id = data.get('req_id', 0)
            candles = data['candles']
            
            # We'd need to map req_id to symbol/granularity in production
            # For now, we'll store generically
            pass
        
        # Balance update
        elif 'balance' in data:
            self.balance = float(data['balance']['balance'])
    
    async def _ping_loop(self):
        """Send ping to keep connection alive"""
        while self.connected:
            try:
                await asyncio.sleep(30)
                ping_request = {"ping": 1}
                await self.connection.send(json.dumps(ping_request))
            except:
                break
    
    def get_candles(self, symbol, granularity, count=300):
        """
        Get candle data
        granularity: 60 (1m), 300 (5m), 900 (15m), 1800 (30m), 3600 (1h)
        """
        if not self.connected:
            return None
        
        cache_key = f"{symbol}_{granularity}"
        
        # Return from cache if available
        if cache_key in self.candle_data and len(self.candle_data[cache_key]) >= count:
            df = pd.DataFrame(self.candle_data[cache_key][-count:])
            return df
        
        # Otherwise fetch history
        async def fetch():
            request = {
                "ticks_history": symbol,
                "granularity": granularity,
                "style": "candles",
                "count": count,
                "req_id": hash(f"{symbol}{granularity}") % 10000
            }
            await self.connection.send(json.dumps(request))
            response = await self.connection.recv()
            return json.loads(response)
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(fetch())
            
            if 'candles' in result:
                candles = result['candles']
                df_data = []
                for c in candles:
                    df_data.append({
                        'time': datetime.fromtimestamp(c['epoch']),
                        'open': float(c['open']),
                        'high': float(c['high']),
                        'low': float(c['low']),
                        'close': float(c['close']),
                        'volume': int(c.get('volume', 0))
                    })
                
                df = pd.DataFrame(df_data)
                
                # Update cache
                if cache_key not in self.candle_data:
                    self.candle_data[cache_key] = []
                self.candle_data[cache_key].extend(df_data)
                
                # Keep last 1000
                if len(self.candle_data[cache_key]) > 1000:
                    self.candle_data[cache_key] = self.candle_data[cache_key][-1000:]
                
                # Subscribe for updates
                self.subscribe_candles(symbol, granularity)
                
                return df
            
            return None
            
        except Exception as e:
            print(f"Get candles error: {e}")
            return None
    
    def subscribe_candles(self, symbol, granularity):
        """Subscribe to live candle updates"""
        if not self.connected:
            return False
        
        async def subscribe():
            request = {
                "ticks_history": symbol,
                "granularity": granularity,
                "style": "candles",
                "subscribe": 1,
                "req_id": hash(f"{symbol}{granularity}_sub") % 10000
            }
            await self.connection.send(json.dumps(request))
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(subscribe())
            
            sub_key = f"{symbol}_{granularity}"
            self.candle_subscriptions[sub_key] = True
            return True
            
        except:
            return False
    
    def get_current_price(self, symbol):
        """Get current price for symbol"""
        if symbol in self.price_cache:
            return self.price_cache[symbol]
        
        # Request tick if not cached
        if self.connected:
            async def get_tick():
                request = {"ticks": symbol}
                await self.connection.send(json.dumps(request))
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(get_tick())
                
                # Wait a moment for response
                time.sleep(0.5)
                
                if symbol in self.price_cache:
                    return self.price_cache[symbol]
            except:
                pass
        
        return None
    
    def place_order(self, symbol, direction, volume, order_type="market", price=None):
        """
        Place a CFD trade order
        """
        if not self.connected:
            return False, "Not connected"
        
        # Convert volume to contract size
        contract_size = volume  # Deriv uses lots directly
        
        async def place():
            # For CFD trading, we use "buy" or "sell" with contract params
            proposal_request = {
                "proposal": 1,
                "amount": contract_size,
                "basis": "stake",
                "contract_type": "CALL" if direction == "BUY" else "PUT",
                "currency": self.currency,
                "symbol": symbol
            }
            
            await self.connection.send(json.dumps(proposal_request))
            response = await self.connection.recv()
            proposal_data = json.loads(response)
            
            if 'error' in proposal_data:
                return False, proposal_data['error']['message']
            
            if 'proposal' in proposal_data:
                proposal_id = proposal_data['proposal']['id']
                
                # Execute the trade
                buy_request = {
                    "buy": proposal_id,
                    "price": proposal_data['proposal']['ask_price']
                }
                
                await self.connection.send(json.dumps(buy_request))
                buy_response = await self.connection.recv()
                buy_data = json.loads(buy_response)
                
                if 'buy' in buy_data:
                    return True, {
                        'transaction_id': buy_data['buy']['transaction_id'],
                        'price': buy_data['buy']['buy_price'],
                        'contract_id': buy_data['buy']['contract_id']
                    }
                else:
                    return False, buy_data.get('error', {}).get('message', 'Unknown error')
            
            return False, "Proposal failed"
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success, result = loop.run_until_complete(place())
            
            if success:
                return True, result
            else:
                return False, result
                
        except Exception as e:
            return False, str(e)
    
    def disconnect(self):
        """Disconnect from Deriv"""
        self.connected = False
        if self.connection:
            async def close():
                await self.connection.close()
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(close())
            except:
                pass

# Helper function to map timeframes to granularity
def timeframe_to_granularity(timeframe):
    """Convert timeframe string to Deriv granularity"""
    mapping = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        'H1': 3600,
        'M15': 900,
        'M30': 1800
    }
    return mapping.get(timeframe, 900)  # Default 15m