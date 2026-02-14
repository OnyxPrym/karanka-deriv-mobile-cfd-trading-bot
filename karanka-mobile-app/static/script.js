// Frontend JavaScript for Karanka Mobile App

let socket = null;
let currentAccountType = 'demo';
let isConnected = false;
let isTrading = false;
let updateInterval = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    // Setup tab switching
    setupTabs();
    
    // Load saved settings
    loadSettings();
    
    // Setup event listeners
    setupEventListeners();
    
    // Start periodic updates
    startUpdates();
});

function setupTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab
            tab.classList.add('active');
            
            // Show corresponding content
            const tabId = tab.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

function setupEventListeners() {
    // Connect button
    document.getElementById('connectBtn').addEventListener('click', connectToDeriv);
    
    // Start/Stop trading buttons
    document.getElementById('startBtn').addEventListener('click', startTrading);
    document.getElementById('stopBtn').addEventListener('click', stopTrading);
    
    // Save settings button
    document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
    
    // Account type toggle
    document.getElementById('accountType').addEventListener('change', function(e) {
        currentAccountType = e.target.value;
    });
    
    // Market selection
    document.querySelectorAll('.market-checkbox input').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            updateSelectedMarkets();
        });
    });
    
    // Select/Deselect all buttons
    document.getElementById('selectAllBtn')?.addEventListener('click', selectAllMarkets);
    document.getElementById('deselectAllBtn')?.addEventListener('click', deselectAllMarkets);
}

function startUpdates() {
    // Update status every 2 seconds
    updateInterval = setInterval(() => {
        if (isConnected) {
            fetchStatus();
            fetchAnalysis();
        }
    }, 2000);
}

function connectToDeriv() {
    const token = document.getElementById('apiToken').value.trim();
    const accountType = document.getElementById('accountType').value;
    
    if (!token) {
        showToast('Please enter your API token', 'error');
        return;
    }
    
    // Show loading
    document.getElementById('connectBtn').disabled = true;
    document.getElementById('connectBtn').textContent = 'Connecting...';
    
    // Send connection request
    fetch('/api/connect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            token: token,
            account_type: accountType
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            isConnected = true;
            showToast('Connected successfully!', 'success');
            document.getElementById('connectionStatus').innerHTML = `
                <span class="stat-label">Account:</span>
                <span class="stat-value">${data.loginid || 'N/A'}</span>
                <span class="stat-label">Balance:</span>
                <span class="stat-value">${data.balance || 0} ${data.currency || 'USD'}</span>
            `;
            document.getElementById('startBtn').disabled = false;
        } else {
            showToast('Connection failed: ' + data.error, 'error');
        }
    })
    .catch(error => {
        showToast('Error: ' + error.message, 'error');
    })
    .finally(() => {
        document.getElementById('connectBtn').disabled = false;
        document.getElementById('connectBtn').textContent = '🔗 Connect';
    });
}

function startTrading() {
    if (!isConnected) {
        showToast('Please connect first', 'error');
        return;
    }
    
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    
    fetch('/api/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            isTrading = true;
            showToast('Trading started', 'success');
            document.getElementById('tradingStatus').textContent = 'ACTIVE';
            document.getElementById('tradingStatus').style.color = '#00FF00';
        } else {
            showToast('Failed to start: ' + data.error, 'error');
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
    })
    .catch(error => {
        showToast('Error: ' + error.message, 'error');
        document.getElementById('startBtn').disabled = false;
    });
}

function stopTrading() {
    fetch('/api/stop', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            isTrading = false;
            showToast('Trading stopped', 'success');
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('tradingStatus').textContent = 'STOPPED';
            document.getElementById('tradingStatus').style.color = '#FF4444';
        }
    });
}

function fetchStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            updateStatusDisplay(data);
        })
        .catch(error => console.error('Status error:', error));
}

function fetchAnalysis() {
    fetch('/api/analysis')
        .then(response => response.json())
        .then(data => {
            updateAnalysisDisplay(data);
        })
        .catch(error => console.error('Analysis error:', error));
}

function updateStatusDisplay(status) {
    // Update header status
    document.getElementById('headerStatus').innerHTML = `
        Balance: ${status.balance || 0} ${status.currency || 'USD'} | 
        Trades: ${status.trades_today || 0} |
        Active: ${status.active_trades || 0}
    `;
    
    // Update stats panel
    const statsHtml = `
        <div class="stat-row">
            <span class="stat-label">Connection:</span>
            <span class="stat-value">${status.connected ? '✅' : '❌'}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Trading:</span>
            <span class="stat-value">${status.running ? '✅' : '❌'}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Account Type:</span>
            <span class="stat-value">${status.account_type || 'N/A'}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Balance:</span>
            <span class="stat-value">${status.balance || 0} ${status.currency || 'USD'}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Trades Today:</span>
            <span class="stat-value">${status.trades_today || 0}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Active Trades:</span>
            <span class="stat-value">${status.active_trades || 0}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Markets Scanned:</span>
            <span class="stat-value">${status.scanned_markets || 0}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Ready Signals:</span>
            <span class="stat-value">${status.ready_signals || 0}</span>
        </div>
    `;
    
    document.getElementById('statsPanel').innerHTML = statsHtml;
}

function updateAnalysisDisplay(data) {
    let html = '<div class="analysis-header">🔍 LIVE MARKET SCAN</div>';
    
    // Ready signals section
    if (data.ready_signals && Object.keys(data.ready_signals).length > 0) {
        html += '<div style="color: #00FF00; margin: 10px 0;">🟢 READY TO TRADE:</div>';
        
        Object.values(data.ready_signals).forEach(signal => {
            const strategyIcon = {
                'QUASIMODO': '🔻',
                'PULLBACK': '📈',
                'BREAKOUT': '💥'
            }[signal.strategy] || '📊';
            
            html += `
                <div class="ready-signal">
                    <div style="display: flex; justify-content: space-between;">
                        <span>${strategyIcon} ${signal.symbol} ${signal.direction}</span>
                        <span>Score: ${signal.score}</span>
                    </div>
                    <div style="font-size: 0.7rem; color: #B8860B;">
                        ${signal.strategy} | Entry: ${signal.entry.toFixed(5)}
                    </div>
                    <div style="font-size: 0.65rem; color: #666;">
                        ${signal.reasons?.join(', ') || ''}
                    </div>
                </div>
            `;
        });
    }
    
    // Market scans section
    html += '<div style="color: #D4AF37; margin: 10px 0;">🔍 MARKETS BEING SCANNED:</div>';
    
    if (data.scan_results) {
        Object.entries(data.scan_results).forEach(([symbol, scan]) => {
            const hasPatterns = scan.qm_patterns > 0 || scan.pb_setups > 0 || scan.breakout;
            const icon = hasPatterns ? '🟡' : '⚪';
            
            html += `
                <div class="scan-card">
                    <div class="scan-header">
                        <span class="symbol">${icon} ${symbol}</span>
                        <span class="price">${scan.price.toFixed(5)}</span>
                    </div>
                    <div class="scan-details">
                        <span class="scan-badge">${scan.regime}</span>
                        <span class="scan-badge">${scan.trend}</span>
                        <span class="scan-badge">Vol:${scan.volatility}</span>
                    </div>
                    <div style="font-size: 0.7rem; margin-top: 5px;">
                        ${scan.qm_patterns > 0 ? `<span class="scan-badge">QM:${scan.qm_patterns}</span>` : ''}
                        ${scan.pb_setups > 0 ? '<span class="scan-badge">PB</span>' : ''}
                        ${scan.breakout ? '<span class="scan-badge">BO</span>' : ''}
                    </div>
                </div>
            `;
        });
    }
    
    document.getElementById('liveAnalysis').innerHTML = html;
}

function saveSettings() {
    const settings = {
        max_daily_trades: parseInt(document.getElementById('maxDaily').value) || 25,
        max_hourly_trades: parseInt(document.getElementById('maxHourly').value) || 6,
        min_seconds_between: parseInt(document.getElementById('minSeconds').value) || 8,
        max_concurrent: parseInt(document.getElementById('maxConcurrent').value) || 5,
        fixed_lot_size: parseFloat(document.getElementById('lotSize').value) || 0.01,
        enable_5m: document.getElementById('enable5m').checked,
        enable_15m: document.getElementById('enable15m').checked,
        enable_30m: document.getElementById('enable30m').checked,
        enabled_symbols: getSelectedMarkets()
    };
    
    fetch('/api/settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Settings saved', 'success');
        } else {
            showToast('Error saving settings', 'error');
        }
    });
}

function loadSettings() {
    fetch('/api/settings')
        .then(response => response.json())
        .then(settings => {
            if (settings) {
                document.getElementById('maxDaily').value = settings.max_daily_trades || 25;
                document.getElementById('maxHourly').value = settings.max_hourly_trades || 6;
                document.getElementById('minSeconds').value = settings.min_seconds_between || 8;
                document.getElementById('maxConcurrent').value = settings.max_concurrent || 5;
                document.getElementById('lotSize').value = settings.fixed_lot_size || 0.01;
                document.getElementById('enable5m').checked = settings.enable_5m !== false;
                document.getElementById('enable15m').checked = settings.enable_15m !== false;
                document.getElementById('enable30m').checked = settings.enable_30m !== false;
            }
        });
}

function getSelectedMarkets() {
    const selected = [];
    document.querySelectorAll('.market-checkbox input:checked').forEach(cb => {
        selected.push(cb.value);
    });
    return selected;
}

function updateSelectedMarkets() {
    // Just updates the internal state
}

function selectAllMarkets() {
    document.querySelectorAll('.market-checkbox input').forEach(cb => {
        cb.checked = true;
    });
}

function deselectAllMarkets() {
    document.querySelectorAll('.market-checkbox input').forEach(cb => {
        cb.checked = false;
    });
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (updateInterval) {
        clearInterval(updateInterval);
    }
});