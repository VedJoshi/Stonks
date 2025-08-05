from flask import Flask, render_template, jsonify
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

class WebDashboard:
    def __init__(self, db_path="trading_bot_production.db"):
        self.db_path = db_path
    
    def get_dashboard_data(self):
        """Get all dashboard data"""
        if not os.path.exists(self.db_path):
            return {
                'account': {'error': 'Database not found'},
                'trades': [],
                'performance': [],
                'sentiment': [],
                'risk_events': [],
                'last_updated': datetime.now().isoformat()
            }
        
        conn = sqlite3.connect(self.db_path)
        
        # Account summary (mock data - in production this would come from trading engine)
        try:
            account_data = {
                'portfolio_value': 1050.25,
                'cash': 948.75,
                'day_pnl': 15.50,
                'total_pnl': 50.25,
                'buying_power': 948.75,
                'positions_count': 2
            }
        except:
            account_data = {'error': 'Unable to fetch account data'}
        
        # Recent trades
        try:
            trades_df = pd.read_sql_query('''
                SELECT symbol, side, quantity, price, timestamp, pnl, status
                FROM trades 
                WHERE timestamp >= datetime('now', '-7 days')
                ORDER BY timestamp DESC
                LIMIT 20
            ''', conn)
        except:
            trades_df = pd.DataFrame()
        
        # Performance metrics
        try:
            performance_df = pd.read_sql_query('''
                SELECT date, total_pnl, total_trades, win_rate, max_drawdown
                FROM daily_performance
                WHERE date >= date('now', '-30 days')
                ORDER BY date DESC
            ''', conn)
        except:
            performance_df = pd.DataFrame()
        
        # Sentiment data
        try:
            sentiment_df = pd.read_sql_query('''
                SELECT symbol, AVG(sentiment_score) as avg_sentiment, 
                       AVG(confidence) as avg_confidence, COUNT(*) as data_points
                FROM sentiment_data
                WHERE timestamp >= datetime('now', '-24 hours')
                GROUP BY symbol
                ORDER BY avg_sentiment DESC
            ''', conn)
        except:
            sentiment_df = pd.DataFrame()
        
        # Risk events
        try:
            risk_df = pd.read_sql_query('''
                SELECT timestamp, event_type, description, severity
                FROM risk_events
                WHERE timestamp >= datetime('now', '-7 days')
                ORDER BY timestamp DESC
                LIMIT 10
            ''', conn)
        except:
            risk_df = pd.DataFrame()
        
        conn.close()
        
        return {
            'account': account_data,
            'trades': trades_df.to_dict('records') if not trades_df.empty else [],
            'performance': performance_df.to_dict('records') if not performance_df.empty else [],
            'sentiment': sentiment_df.to_dict('records') if not sentiment_df.empty else [],
            'risk_events': risk_df.to_dict('records') if not risk_df.empty else [],
            'last_updated': datetime.now().isoformat()
        }

dashboard = WebDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/dashboard')
def api_dashboard():
    """API endpoint for dashboard data"""
    try:
        data = dashboard.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'trading_bot_dashboard'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
