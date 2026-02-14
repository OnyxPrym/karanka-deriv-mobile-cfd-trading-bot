<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes">
    <title>KARANKA MULTIVERSE ALGO AI</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>KARANKA MULTIVERSE ALGO AI</h1>
            <h2>TRIPLE STRATEGY TRADING SYSTEM</h2>
            <div class="status-bar" id="headerStatus">
                Not Connected
            </div>
        </div>
        
        <!-- Tab Navigation -->
        <div class="tabs">
            <button class="tab-btn active" data-tab="dashboard">📊</button>
            <button class="tab-btn" data-tab="markets">📈</button>
            <button class="tab-btn" data-tab="settings">⚙️</button>
            <button class="tab-btn" data-tab="connection">🔗</button>
            <button class="tab-btn" data-tab="monitor">👁️</button>
        </div>
        
        <!-- Dashboard Tab -->
        <div id="dashboard" class="tab-content active">
            <div class="live-analysis" id="liveAnalysis">
                <div class="analysis-header">🔍 LIVE MARKET SCAN</div>
                <div style="text-align: center; padding: 20px; color: #666;">
                    Connect to Deriv API to start scanning
                </div>
            </div>
            
            <div class="stats-panel" id="statsPanel">
                <div class="stat-row">
                    <span class="stat-label">Status:</span>
                    <span class="stat-value">Not Connected</span>
                </div>
            </div>
            
            <div style="display: flex; gap: 8px;">
                <button class="btn btn-primary" id="startBtn" disabled>🚀 Start Trading</button>
                <button class="btn btn-danger" id="stopBtn" disabled>🛑 Stop</button>
            </div>
        </div>
        
        <!-- Markets Tab -->
        <div id="markets" class="tab-content">
            <h3 style="color: #D4AF37; margin-bottom: 10px;">Select Markets</h3>
            
            <div class="market-grid" id="marketGrid">
                <!-- Forex Majors -->
                <div class="market-checkbox">
                    <input type="checkbox" value="EURUSD" checked> EURUSD
                </div>
                <div class="market-checkbox">
                    <input type="checkbox" value="GBPUSD" checked> GBPUSD
                </div>
                <div class="market-checkbox">
                    <input type="checkbox" value="USDJPY" checked> USDJPY
                </div>
                <div class="market-checkbox">
                    <input type="checkbox" value="AUDUSD" checked> AUDUSD
                </div>
                <div class="market-checkbox">
                    <input type="checkbox" value="USDCAD" checked> USDCAD
                </div>
                <div class="market-checkbox">
                    <input type="checkbox" value="USDCHF" checked> USDCHF
                </div>
                <div class="market-checkbox">
                    <input type="checkbox" value="NZDUSD" checked> NZDUSD
                </div>
                
                <!-- Commodities -->
                <div class="market-checkbox">
                    <input type="checkbox" value="XAUUSD" checked> XAUUSD
                </div>
                <div class="market-checkbox">
                    <input type="checkbox" value="XAGUSD"> XAGUSD
                </div>
                
                <!-- Indices -->
                <div class="market-checkbox">
                    <input type="checkbox" value="US30"> US30
                </div>
                <div class="market-checkbox">
                    <input type="checkbox" value="USTEC"> USTEC
                </div>
                
                <!-- Crypto -->
                <div class="market-checkbox">
                    <input type="checkbox" value="BTCUSD"> BTCUSD
                </div>
            </div>
            
            <div style="display: flex; gap: 8px; margin: 15px 0;">
                <button class="btn btn-primary" id="selectAllBtn">✅ Select All</button>
                <button class="btn btn-danger" id="deselectAllBtn">❌ Deselect All</button>
            </div>
        </div>
        
        <!-- Settings Tab -->
        <div id="settings" class="tab-content">
            <h3 style="color: #D4AF37; margin-bottom: 10px;">Trading Settings</h3>
            
            <div class="settings-grid">
                <div class="setting-item">
                    <label>Max Daily Trades</label>
                    <input type="number" id="maxDaily" value="25" min="1" max="50">
                </div>
                <div class="setting-item">
                    <label>Max Hourly Trades</label>
                    <input type="number" id="maxHourly" value="6" min="1" max="20">
                </div>
                <div class="setting-item">
                    <label>Seconds Between</label>
                    <input type="number" id="minSeconds" value="8" min="3" max="60">
                </div>
                <div class="setting-item">
                    <label>Max Concurrent</label>
                    <input type="number" id="maxConcurrent" value="5" min="1" max="10">
                </div>
                <div class="setting-item">
                    <label>Lot Size</label>
                    <input type="number" id="lotSize" value="0.01" min="0.01" max="1" step="0.01">
                </div>
            </div>
            
            <h3 style="color: #D4AF37; margin: 15px 0 10px;">Timeframes</h3>
            
            <div style="background: #111; padding: 10px; border-radius: 5px;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <label class="switch">
                        <input type="checkbox" id="enable5m" checked>
                        <span class="slider"></span>
                    </label>
                    <span style="color: #FFD700;">5M (Breakout)</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <label class="switch">
                        <input type="checkbox" id="enable15m" checked>
                        <span class="slider"></span>
                    </label>
                    <span style="color: #FFD700;">15M (Quasimodo)</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <label class="switch">
                        <input type="checkbox" id="enable30m" checked>
                        <span class="slider"></span>
                    </label>
                    <span style="color: #FFD700;">30M (Secondary)</span>
                </div>
            </div>
            
            <button class="btn btn-primary" id="saveSettingsBtn" style="margin-top: 15px;">💾 Save Settings</button>
        </div>
        
        <!-- Connection Tab -->
        <div id="connection" class="tab-content">
            <div class="connection-panel">
                <h3 style="color: #D4AF37; margin-bottom: 15px;">Deriv API Connection</h3>
                
                <div class="input-group">
                    <label>Account Type</label>
                    <select id="accountType">
                        <option value="demo">Demo Account</option>
                        <option value="real">Real Account</option>
                    </select>
                </div>
                
                <div class="input-group">
                    <label>API Token</label>
                    <input type="password" id="apiToken" placeholder="Enter your Deriv API token">
                </div>
                
                <button class="btn btn-primary" id="connectBtn">🔗 Connect</button>
                
                <div style="margin-top: 20px; padding: 10px; background: #000; border-radius: 5px; font-size: 0.8rem; color: #B8860B;">
                    <strong>How to get API token:</strong><br>
                    1. Log in to Deriv<br>
                    2. Go to Settings → API Token<br>
                    3. Generate new token with "Trade" permissions<br>
                    4. Copy and paste above
                </div>
                
                <div style="margin-top: 15px;" id="connectionStatus"></div>
            </div>
        </div>
        
        <!-- Monitor Tab -->
        <div id="monitor" class="tab-content">
            <h3 style="color: #D4AF37; margin-bottom: 10px;">Active Trades</h3>
            <div id="activeTrades" style="margin-bottom: 15px;">
                <div style="text-align: center; padding: 20px; color: #666;">
                    No active trades
                </div>
            </div>
            
            <h3 style="color: #D4AF37; margin-bottom: 10px;">Today's Trades</h3>
            <div id="tradeHistory" style="margin-bottom: 15px;">
                <div style="text-align: center; padding: 20px; color: #666;">
                    No trades yet today
                </div>
            </div>
        </div>
    </div>
    
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>