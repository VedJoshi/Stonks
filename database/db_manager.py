import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import os
import shutil

class ProductionDatabaseManager:
    def __init__(self, db_path="trading_bot_production.db"):
        self.db_path = db_path
        self.backup_dir = "database_backups"
        self.init_database()
        self.setup_backup_system()
    
    def setup_backup_system(self):
        """Setup automatic database backups"""
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def backup_database(self):
        """Create a backup of the database"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"trading_bot_backup_{timestamp}.db")
            shutil.copy2(self.db_path, backup_path)
            
            # Keep only last 7 days of backups
            cutoff_date = datetime.now() - timedelta(days=7)
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("trading_bot_backup_"):
                    file_path = os.path.join(self.backup_dir, filename)
                    if os.path.getctime(file_path) < cutoff_date.timestamp():
                        os.remove(file_path)
                        
            logging.info(f"Database backed up to {backup_path}")
        except Exception as e:
            logging.error(f"Database backup failed: {e}")
    
    def init_database(self):
        """Initialize production database with enhanced schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced sentiment data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                text_content TEXT,
                post_count INTEGER DEFAULT 1,
                upvotes INTEGER DEFAULT 0,
                comments_count INTEGER DEFAULT 0,
                author_karma INTEGER DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        # Enhanced trading signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                signal_type TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence REAL NOT NULL,
                technical_score REAL DEFAULT 0,
                combined_score REAL NOT NULL,
                volume_factor REAL DEFAULT 1.0,
                executed BOOLEAN DEFAULT FALSE,
                execution_price REAL,
                execution_time DATETIME,
                reason TEXT
            )
        ''')
        
        # Enhanced trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                order_id TEXT UNIQUE,
                status TEXT DEFAULT 'pending',
                filled_price REAL,
                filled_quantity INTEGER DEFAULT 0,
                commission REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                pnl_percent REAL DEFAULT 0,
                hold_time_minutes INTEGER DEFAULT 0,
                exit_reason TEXT,
                sentiment_at_entry REAL,
                sentiment_at_exit REAL
            )
        ''')
        
        # Risk management log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                symbol TEXT,
                description TEXT NOT NULL,
                severity TEXT DEFAULT 'INFO',
                account_impact REAL DEFAULT 0,
                action_taken TEXT
            )
        ''')
        
        # Performance tracking with detailed metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE DEFAULT CURRENT_DATE,
                starting_balance REAL NOT NULL,
                ending_balance REAL NOT NULL,
                total_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                largest_win REAL DEFAULT 0,
                largest_loss REAL DEFAULT 0,
                avg_win REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                sharpe_ratio REAL DEFAULT 0,
                total_commission REAL DEFAULT 0,
                win_rate REAL DEFAULT 0
            )
        ''')
        
        # Account snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                account_value REAL NOT NULL,
                buying_power REAL NOT NULL,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                day_trade_count INTEGER DEFAULT 0,
                positions_json TEXT,
                market_value REAL DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info("Production database initialized successfully")
    
    def insert_sentiment_data(self, symbol, source, sentiment_score, text_content="", post_count=1, upvotes=0, comments_count=0, author_karma=0, confidence=0.5):
        """Insert sentiment data into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sentiment_data (symbol, source, sentiment_score, confidence, text_content, post_count, upvotes, comments_count, author_karma)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, source, sentiment_score, confidence, text_content, post_count, upvotes, comments_count, author_karma))
        
        conn.commit()
        conn.close()
    
    def insert_trading_signal(self, symbol, signal_type, sentiment_score, confidence, technical_score=0, combined_score=None, volume_factor=1.0, reason=""):
        """Insert trading signal into database"""
        if combined_score is None:
            combined_score = sentiment_score
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trading_signals (symbol, signal_type, sentiment_score, confidence, technical_score, combined_score, volume_factor, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, signal_type, sentiment_score, confidence, technical_score, combined_score, volume_factor, reason))
        
        conn.commit()
        conn.close()
    
    def insert_trade(self, symbol, side, quantity, price, order_id, status='pending', filled_price=None, filled_quantity=0, commission=0):
        """Insert trade record into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (symbol, side, quantity, price, order_id, status, filled_price, filled_quantity, commission)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, side, quantity, price, order_id, status, filled_price, filled_quantity, commission))
        
        conn.commit()
        conn.close()
    
    def log_risk_event(self, event_type, description, symbol=None, severity="INFO", account_impact=0, action_taken=None):
        """Log risk management events"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO risk_events (event_type, symbol, description, severity, account_impact, action_taken)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (event_type, symbol, description, severity, account_impact, action_taken))
        
        conn.commit()
        conn.close()
        
        if severity in ["WARNING", "ERROR", "CRITICAL"]:
            logging.warning(f"Risk Event - {event_type}: {description}")
    
    def update_daily_performance(self, performance_data):
        """Update daily performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        # Check if today's record exists
        cursor.execute('SELECT id FROM daily_performance WHERE date = ?', (today,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            cursor.execute('''
                UPDATE daily_performance SET
                ending_balance = ?, total_pnl = ?, realized_pnl = ?, unrealized_pnl = ?,
                total_trades = ?, winning_trades = ?, losing_trades = ?,
                largest_win = ?, largest_loss = ?, avg_win = ?, avg_loss = ?,
                max_drawdown = ?, total_commission = ?, win_rate = ?
                WHERE date = ?
            ''', (
                performance_data['ending_balance'], performance_data['total_pnl'],
                performance_data['realized_pnl'], performance_data['unrealized_pnl'],
                performance_data['total_trades'], performance_data['winning_trades'],
                performance_data['losing_trades'], performance_data['largest_win'],
                performance_data['largest_loss'], performance_data['avg_win'],
                performance_data['avg_loss'], performance_data['max_drawdown'],
                performance_data['total_commission'], performance_data['win_rate'],
                today
            ))
        else:
            # Insert new record
            cursor.execute('''
                INSERT INTO daily_performance (
                    date, starting_balance, ending_balance, total_pnl, realized_pnl,
                    unrealized_pnl, total_trades, winning_trades, losing_trades,
                    largest_win, largest_loss, avg_win, avg_loss, max_drawdown,
                    total_commission, win_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today, performance_data.get('starting_balance', 1000), performance_data['ending_balance'],
                performance_data['total_pnl'], performance_data['realized_pnl'],
                performance_data['unrealized_pnl'], performance_data['total_trades'],
                performance_data['winning_trades'], performance_data['losing_trades'],
                performance_data['largest_win'], performance_data['largest_loss'],
                performance_data['avg_win'], performance_data['avg_loss'],
                performance_data['max_drawdown'], performance_data['total_commission'],
                performance_data['win_rate']
            ))
        
        conn.commit()
        conn.close()
    
    def get_performance_summary(self, days=30):
        """Get comprehensive performance summary"""
        conn = sqlite3.connect(self.db_path)
        
        # Get daily performance
        daily_perf = pd.read_sql_query('''
            SELECT * FROM daily_performance 
            WHERE date >= date('now', '-{} days')
            ORDER BY date DESC
        '''.format(days), conn)
        
        # Get trade statistics
        trade_stats = pd.read_sql_query('''
            SELECT 
                COUNT(*) as total_trades,
                AVG(pnl) as avg_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade,
                AVG(hold_time_minutes) as avg_hold_time
            FROM trades 
            WHERE timestamp >= datetime('now', '-{} days')
            AND status = 'filled'
        '''.format(days), conn)
        
        # Get recent risk events
        risk_events = pd.read_sql_query('''
            SELECT * FROM risk_events 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days), conn)
        
        conn.close()
        
        return {
            'daily_performance': daily_perf,
            'trade_statistics': trade_stats,
            'risk_events': risk_events
        }
    
    def get_trades_by_date(self, date):
        """Get trades for a specific date"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trades 
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp DESC
        ''', (date,))
        
        trades = cursor.fetchall()
        conn.close()
        
        return [dict(zip([column[0] for column in cursor.description], trade)) for trade in trades] if trades else []
    
    def get_recent_trades(self, count=10):
        """Get recent trades"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trades 
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (count,))
        
        trades = cursor.fetchall()
        conn.close()
        
        return [dict(zip([column[0] for column in cursor.description], trade)) for trade in trades] if trades else []
