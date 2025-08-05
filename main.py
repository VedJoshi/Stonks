"""
PRODUCTION TRADING BOT - MAIN APPLICATION
Complete automated trading system with sentiment analysis and risk management
"""

import os
import sys
import time
import logging
import schedule
import signal
import threading
from datetime import datetime, timedelta
from contextlib import contextmanager

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import with error handling
def safe_import():
    """Safely import modules with proper error handling"""
    try:
        from database.db_manager import ProductionDatabaseManager
        from sentiment.sentiment_analyzer import SentimentAnalyzer
        from risk_management.risk_manager import RiskManager, DailyRiskCheck
        from monitoring.dashboard import NotificationSystem, KPITracker, WeeklyReview
        from monitoring.web_dashboard import WebDashboard  # Fixed: was LiveDashboard
        
        # Try to import trading engine - this will need API keys
        try:
            from trading.trading_engine import ProductionTradingEngine
        except Exception as e:
            print(f"Warning: Trading engine import failed: {e}")
            print("This is expected if API keys are not configured yet.")
            ProductionTradingEngine = None
        
        # Try to import config
        try:
            from config.credentials import get_config
        except ImportError:
            print("Warning: config/credentials.py not found. Using default config.")
            def get_config():
                return {
                    'MAX_DAILY_TRADES': 10,
                    'DAILY_LOSS_LIMIT': 100,
                    'MAX_SINGLE_TRADE': 500,
                    'POSITION_SIZE_PERCENT': 0.1,
                    'STOP_LOSS_PERCENT': 0.05,
                    'TAKE_PROFIT_PERCENT': 0.10,
                    'CONFIDENCE_THRESHOLD': 0.6,
                    'SENTIMENT_BUY_THRESHOLD': 0.7,
                    'SENTIMENT_SELL_THRESHOLD': -0.7,
                    'TRADING_COOLDOWN': 300,
                    'EMAIL_NOTIFICATIONS': False,
                    'TELEGRAM_NOTIFICATIONS': False,
                    # Add default Reddit credentials for demo
                    'REDDIT_CLIENT_ID': 'demo_client_id',
                    'REDDIT_CLIENT_SECRET': 'demo_client_secret',
                    'REDDIT_USER_AGENT': 'TradingBot/1.0 by YourUsername'
                }
        
        return (ProductionDatabaseManager, SentimentAnalyzer, ProductionTradingEngine, 
                RiskManager, DailyRiskCheck, NotificationSystem, KPITracker, 
                WeeklyReview, WebDashboard, get_config)
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required packages are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)

class ProductionTradingBot:
    def __init__(self):
        """Initialize the complete production trading bot"""
        print("Starting Production Trading Bot...")
        
        # Safe import of modules
        (self.DatabaseManager, self.SentimentAnalyzer, self.TradingEngine, 
         self.RiskManager, self.DailyRiskCheck, self.NotificationSystem, 
         self.KPITracker, self.WeeklyReview, self.WebDashboard, 
         self.get_config) = safe_import()
        
        self.config = self.get_config()
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Initialize logging
        self.setup_production_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.db_manager = self.DatabaseManager()
        
        # Initialize sentiment analyzer with error handling
        try:
            self.sentiment_analyzer = self.SentimentAnalyzer(
                reddit_client_id=self.config.get('REDDIT_CLIENT_ID', 'dummy'),
                reddit_client_secret=self.config.get('REDDIT_CLIENT_SECRET', 'dummy'),
                reddit_user_agent=self.config.get('REDDIT_USER_AGENT', 'TradingBot/1.0')
            )
            self.logger.info("Sentiment analyzer initialized")
        except Exception as e:
            self.logger.warning(f"Sentiment analyzer initialization failed: {e}")
            self.sentiment_analyzer = None
        
        # Initialize trading engine with error handling
        if self.TradingEngine:
            try:
                self.trading_engine = self.TradingEngine(
                    api_key=self.config.get('ALPACA_API_KEY', ''),
                    secret_key=self.config.get('ALPACA_SECRET_KEY', ''),
                    base_url=self.config.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
                    config=self.config
                )
                self.logger.info("Trading engine initialized")
            except Exception as e:
                self.logger.warning(f"Trading engine initialization failed: {e}")
                self.trading_engine = None
        else:
            self.trading_engine = None
        
        # Initialize risk management
        if self.trading_engine:
            try:
                self.risk_manager = self.RiskManager(
                    max_daily_loss=self.config.get('DAILY_LOSS_LIMIT', 100),
                    max_position_size=self.config.get('MAX_SINGLE_TRADE', 500),
                    max_positions=self.config.get('MAX_POSITIONS', 3)
                )
                self.daily_risk_check = self.DailyRiskCheck(self.trading_engine, self.db_manager)
                self.logger.info("Risk management initialized")
            except Exception as e:
                self.logger.warning(f"Risk management initialization failed: {e}")
                self.risk_manager = None
                self.daily_risk_check = None
        else:
            self.risk_manager = None
            self.daily_risk_check = None
        
        # Initialize monitoring components
        try:
            self.notification_system = self.NotificationSystem(self.config)
            self.kpi_tracker = self.KPITracker(self.db_manager)
            self.weekly_review = self.WeeklyReview(self.db_manager)
            self.logger.info("Monitoring systems initialized")
        except Exception as e:
            self.logger.error(f"Monitoring initialization failed: {e}")
            self.notification_system = None
            self.kpi_tracker = None
            self.weekly_review = None
        
        # Initialize web dashboard
        try:
            self.web_dashboard = self.WebDashboard(self.db_manager.db_path)
            self.logger.info("Web dashboard initialized")
        except Exception as e:
            self.logger.warning(f"Web dashboard initialization failed: {e}")
            self.web_dashboard = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Production Trading Bot initialized successfully")
    
    def setup_production_logging(self):
        """Setup comprehensive production logging"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # File handlers for different log levels
        handlers = []
        
        # Main application log
        main_handler = logging.FileHandler(f"{log_dir}/trading_bot.log", encoding='utf-8')
        main_handler.setFormatter(detailed_formatter)
        main_handler.setLevel(logging.INFO)
        handlers.append(main_handler)
        
        # Error log
        error_handler = logging.FileHandler(f"{log_dir}/errors.log", encoding='utf-8')
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)
        
        # Trade log
        trade_handler = logging.FileHandler(f"{log_dir}/trades.log", encoding='utf-8')
        trade_handler.setFormatter(detailed_formatter)
        trade_handler.setLevel(logging.INFO)
        trade_handler.addFilter(lambda record: "TRADE" in record.getMessage())
        handlers.append(trade_handler)
        
        # Console output with UTF-8 encoding for Windows
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(detailed_formatter)
        console_handler.setLevel(logging.INFO)
        # Force UTF-8 encoding on Windows
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8')
        handlers.append(console_handler)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add all handlers
        for handler in handlers:
            root_logger.addHandler(handler)
    
    def start_production_bot(self):
        """Start the production trading bot with full monitoring"""
        try:
            self.logger.info("STARTING PRODUCTION TRADING BOT")
            self.running = True
            
            # Perform startup checks
            if not self.perform_startup_checks():
                self.logger.error("Startup checks failed - running in limited mode")
                return self.run_limited_mode()
            
            # Initialize database
            self.db_manager.init_database()
            
            # Schedule trading operations (only if trading engine available)
            if self.trading_engine:
                self.schedule_trading_operations()
            else:
                self.logger.info("Running in monitoring-only mode (no trading engine)")
            
            # Schedule monitoring and maintenance
            self.schedule_monitoring_operations()
            
            # Start main execution loop
            self.run_main_loop()
            return True
            
        except Exception as e:
            self.logger.error(f"Critical error in production bot: {e}")
            if self.notification_system:
                self.send_critical_alert(f"Production bot crashed: {e}")
            return False
    
    def run_limited_mode(self):
        """Run in limited mode without trading functionality"""
        self.logger.info("Running in limited mode (monitoring only)")
        
        try:
            # Schedule only monitoring operations
            self.schedule_monitoring_operations()
            
            # Start limited main loop
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Run scheduled jobs
                    schedule.run_pending()
                    
                    # System health check every 30 minutes
                    if datetime.now().minute % 30 == 0:
                        self.system_health_check()
                    
                    # Sleep for 1 minute before next iteration
                    time.sleep(60)
                    
                except Exception as e:
                    self.logger.error(f"Error in limited mode loop: {e}")
                    time.sleep(60)
            
            self.logger.info("Limited mode stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Limited mode failed: {e}")
            return False
    
    def perform_startup_checks(self):
        """Perform comprehensive startup checks"""
        self.logger.info("Performing startup checks...")
        
        checks = [
            ("Database Access", self.check_database_access),
            ("Configuration", self.check_configuration),
        ]
        
        # Only add trading-specific checks if trading engine is available
        if self.trading_engine:
            checks.extend([
                ("API Connection", self.check_api_connection),
                ("Account Status", self.check_account_status),
            ])
        
        passed_checks = 0
        total_checks = len(checks)
        
        for check_name, check_func in checks:
            try:
                if check_func():
                    self.logger.info(f"{check_name}: PASSED")
                    passed_checks += 1
                else:
                    self.logger.warning(f"{check_name}: FAILED")
            except Exception as e:
                self.logger.error(f"{check_name}: ERROR - {e}")
        
        # Allow partial success for limited mode
        if passed_checks >= 2:  # At least database and config
            self.logger.info(f"Startup checks: {passed_checks}/{total_checks} passed")
            return True
        else:
            self.logger.error(f"Startup checks: Only {passed_checks}/{total_checks} passed")
            return False
    
    def check_api_connection(self):
        """Check API connection"""
        if not self.trading_engine:
            return False
        try:
            # Try to get account info to test connection
            account = self.trading_engine.api.get_account()
            return account is not None
        except:
            return False
    
    def check_database_access(self):
        """Check database access"""
        try:
            # Test database by trying to backup
            self.db_manager.backup_database()
            return True
        except Exception as e:
            self.logger.error(f"Database check failed: {e}")
            return False
    
    def check_configuration(self):
        """Check configuration"""
        required_configs = ['MAX_DAILY_TRADES', 'DAILY_LOSS_LIMIT', 'MAX_SINGLE_TRADE']
        for config_key in required_configs:
            if config_key not in self.config:
                return False
        return True
    
    def check_account_status(self):
        """Check account status and buying power"""
        if not self.trading_engine:
            return False
        try:
            account_info = self.trading_engine.get_account_info()
            if not account_info:
                return False
            buying_power = account_info.get('buying_power', 0)
            return buying_power > 100  # Minimum required buying power
        except:
            return False
    
    def schedule_trading_operations(self):
        """Schedule all trading operations"""
        if not self.trading_engine:
            return
            
        # Pre-market sentiment analysis
        schedule.every().day.at("08:00").do(self.pre_market_analysis)
        
        # Market open preparation
        schedule.every().day.at("09:25").do(self.market_open_preparation)
        
        # Main trading loop (every 5 minutes during market hours)
        schedule.every(5).minutes.do(self.trading_cycle)
        
        # End of day processing
        schedule.every().day.at("16:30").do(self.end_of_day_processing)
        
        self.logger.info("Trading operations scheduled")
    
    def schedule_monitoring_operations(self):
        """Schedule monitoring and maintenance operations"""
        # Risk checks every 15 minutes
        if self.daily_risk_check:
            schedule.every(15).minutes.do(self.risk_monitoring_cycle)
        
        # Database backup daily
        schedule.every().day.at("02:00").do(self.db_manager.backup_database)
        
        # Weekly performance review
        if self.weekly_review:
            schedule.every().monday.at("06:00").do(self.weekly_performance_review)
        
        # Monthly KPI calculation - use first day of month instead of .month
        if self.kpi_tracker:
            schedule.every().day.at("06:30").do(self.check_monthly_kpi_review)
        
        self.logger.info("Monitoring operations scheduled")
    
    def check_monthly_kpi_review(self):
        """Check if it's time for monthly KPI review (first day of month)"""
        today = datetime.now()
        if today.day == 1:  # First day of the month
            self.monthly_kpi_review()
    
    def run_main_loop(self):
        """Main execution loop"""
        self.logger.info("Starting main execution loop")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Run scheduled jobs
                schedule.run_pending()
                
                # Check system health
                if datetime.now().minute % 30 == 0:  # Every 30 minutes
                    self.system_health_check()
                
                # Sleep for 1 minute before next iteration
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Continue after error
        
        self.logger.info("Main execution loop stopped")
    
    def trading_cycle(self):
        """Main trading cycle - runs every 5 minutes during market hours"""
        if not self.trading_engine or not self.sentiment_analyzer:
            return
        
        try:
            # Check if market is open (simplified check)
            current_hour = datetime.now().hour
            if current_hour < 9 or current_hour > 16:  # Market hours approximation
                return
        
            self.logger.info("Starting trading cycle")
            
            # 1. Check daily risk limits
            if self.daily_risk_check:
                risk_status = self.daily_risk_check.pre_market_check()
                if not risk_status:
                    self.logger.warning("Daily risk limits exceeded - skipping trading cycle")
                    return
            
            # 2. Analyze sentiment for watchlist stocks
            watchlist = self.get_watchlist()
            sentiment_results = {}
            
            for symbol in watchlist:
                try:
                    sentiment_score = self.sentiment_analyzer.get_combined_sentiment(symbol)
                    sentiment_results[symbol] = sentiment_score
                    
                    # Store sentiment data
                    self.db_manager.insert_sentiment_data(
                        symbol, 'combined', sentiment_score.get('combined_sentiment', 0),
                        confidence=sentiment_score.get('combined_confidence', 0)
                    )
                except Exception as e:
                    self.logger.error(f"Error analyzing sentiment for {symbol}: {e}")
                    continue
            
            # 3. Generate trading signals
            trading_signals = self.generate_trading_signals(sentiment_results)
            
            # 4. Execute trades based on signals
            for signal in trading_signals:
                self.execute_trade_signal(signal)
            
            self.logger.info(f"Trading cycle completed - {len(trading_signals)} signals processed")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def generate_trading_signals(self, sentiment_results):
        """Generate trading signals based on sentiment analysis"""
        signals = []
        
        for symbol, sentiment_data in sentiment_results.items():
            try:
                if not self.trading_engine:
                    continue
                    
                # Get current position
                current_positions = self.trading_engine.get_positions()
                current_position = next((pos for pos in current_positions if pos['symbol'] == symbol), None)
                
                sentiment_score = sentiment_data.get('combined_sentiment', 0)
                confidence = sentiment_data.get('combined_confidence', 0)
                
                # Generate signal using trading engine
                signal_info = self.trading_engine.generate_trading_signal(symbol, sentiment_data)
                
                if signal_info['signal'] == 'buy' and not current_position:
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'sentiment_score': sentiment_score,
                        'confidence': confidence,
                        'strength': signal_info['strength'],
                        'reason': signal_info['reason']
                    })
                
                elif signal_info['signal'] == 'sell' and current_position:
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'sentiment_score': sentiment_score,
                        'confidence': confidence,
                        'strength': signal_info['strength'],
                        'reason': signal_info['reason']
                    })
                    
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals
    
    def execute_trade_signal(self, signal):
        """Execute a trading signal with full risk management"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            sentiment_score = signal['sentiment_score']
            confidence = signal['confidence']
            
            if not self.trading_engine:
                self.logger.warning("Trading engine not available")
                return False
            
            # Log the signal
            self.db_manager.insert_trading_signal(
                symbol, action, sentiment_score, confidence, 
                reason=signal.get('reason', '')
            )
            
            # Execute trade
            if action == 'BUY':
                trade_result = self.trading_engine.place_buy_order_with_stops(
                    symbol, sentiment_score, confidence
                )
            else:
                trade_result = self.trading_engine.place_sell_order(
                    symbol, sentiment_score, confidence
                )
            
            if trade_result:
                # Log successful trade
                self.logger.info(f"TRADE EXECUTED: {trade_result}")
                
                # Store trade in database
                self.db_manager.insert_trade(
                    symbol, action.lower(), trade_result['quantity'], 
                    trade_result['price'], trade_result['order_id'],
                    status='filled'
                )
                
                # Send notification
                if self.notification_system:
                    self.notification_system.send_trade_alert(trade_result)
                
                return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade signal: {e}")
            return False
    
    def pre_market_analysis(self):
        """Pre-market sentiment analysis and preparation"""
        self.logger.info("Starting pre-market analysis")
        
        try:
            if not self.sentiment_analyzer:
                self.logger.warning("Sentiment analyzer not available")
                return
                
            # Get watchlist
            watchlist = self.get_watchlist()
            
            # Analyze pre-market sentiment
            pre_market_sentiment = {}
            for symbol in watchlist:
                try:
                    sentiment = self.sentiment_analyzer.get_combined_sentiment(symbol)
                    pre_market_sentiment[symbol] = sentiment
                except Exception as e:
                    self.logger.error(f"Error analyzing pre-market sentiment for {symbol}: {e}")
                    continue
            
            # Store pre-market data
            for symbol, sentiment in pre_market_sentiment.items():
                self.db_manager.insert_sentiment_data(
                    symbol, 'pre_market', sentiment.get('combined_sentiment', 0),
                    confidence=sentiment.get('combined_confidence', 0)
                )
            
            self.logger.info(f"Pre-market analysis completed for {len(pre_market_sentiment)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error in pre-market analysis: {e}")
    
    def market_open_preparation(self):
        """Prepare for market open"""
        self.logger.info("Market open preparation")
        
        try:
            if self.trading_engine:
                # Check account status
                account_info = self.trading_engine.get_account_info()
                if account_info:
                    self.logger.info(f"Account equity: ${account_info['equity']:,.2f}")
                    self.logger.info(f"Buying power: ${account_info['buying_power']:,.2f}")
            
            # Reset daily risk counters
            if self.daily_risk_check:
                self.daily_risk_check.reset_daily_tracking()
            
            self.logger.info("Market open preparation completed")
            
        except Exception as e:
            self.logger.error(f"Error in market open preparation: {e}")
    
    def end_of_day_processing(self):
        """End of day processing and reporting"""
        self.logger.info("Starting end-of-day processing")
        
        try:
            # Calculate daily performance
            daily_performance = self.calculate_daily_performance()
            
            if daily_performance:
                # Store daily performance
                self.db_manager.update_daily_performance(daily_performance)
                
                # Generate daily report
                self.generate_daily_report(daily_performance)
            
            self.logger.info("End-of-day processing completed")
            
        except Exception as e:
            self.logger.error(f"Error in end-of-day processing: {e}")
    
    def risk_monitoring_cycle(self):
        """Risk monitoring cycle - runs every 15 minutes"""
        try:
            if not self.daily_risk_check:
                return
                
            # Check for risk violations
            risk_status = self.daily_risk_check.pre_market_check()
            
            if not risk_status:
                self.logger.warning("Risk limits violated")
                if self.notification_system:
                    self.notification_system.send_risk_alert(
                        "Daily Limits Exceeded",
                        "Trading has been suspended due to risk limit violations"
                    )
            
            # Monitor individual positions
            self.monitor_position_risks()
            
        except Exception as e:
            self.logger.error(f"Error in risk monitoring: {e}")
    
    def system_health_check(self):
        """Perform system health check"""
        try:
            health_status = {
                'timestamp': datetime.now(),
                'database_status': self.check_database_access(),
                'memory_usage': self.get_memory_usage(),
                'active_positions': len(self.trading_engine.get_positions()) if self.trading_engine else 0
            }
            
            if self.trading_engine:
                health_status['api_connection'] = self.check_api_connection()
            
            # Log health status
            self.logger.info(f"System Health: {health_status}")
            
        except Exception as e:
            self.logger.error(f"Error in system health check: {e}")
    
    def weekly_performance_review(self):
        """Generate weekly performance review"""
        try:
            if not self.weekly_review:
                return
                
            self.logger.info("Generating weekly performance review")
            
            weekly_report = self.weekly_review.generate_weekly_report()
            
            # Send weekly report
            self.send_weekly_report(weekly_report)
            
        except Exception as e:
            self.logger.error(f"Error in weekly performance review: {e}")
    
    def monthly_kpi_review(self):
        """Generate monthly KPI review"""
        try:
            if not self.kpi_tracker:
                return
                
            self.logger.info("Generating monthly KPI review")
            
            monthly_kpis = self.kpi_tracker.calculate_monthly_kpis()
            recommendations = self.kpi_tracker.generate_optimization_recommendations(monthly_kpis)
            
            # Send monthly report
            self.send_monthly_report(monthly_kpis, recommendations)
            
        except Exception as e:
            self.logger.error(f"Error in monthly KPI review: {e}")
    
    def get_watchlist(self):
        """Get the current watchlist"""
        # Default watchlist - can be loaded from config or database
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    
    def calculate_daily_performance(self):
        """Calculate daily performance metrics"""
        try:
            if not self.trading_engine:
                return None
                
            account_info = self.trading_engine.get_account_info()
            if not account_info:
                return None
                
            positions = self.trading_engine.get_positions()
            
            total_equity = account_info['equity']
            total_pnl = account_info.get('total_pnl', 0)
            
            return {
                'date': datetime.now().date(),
                'starting_balance': 1000,  # This should be tracked properly
                'ending_balance': total_equity,
                'total_pnl': total_pnl,
                'realized_pnl': 0,  # This should be calculated from closed trades
                'unrealized_pnl': sum(pos.get('unrealized_pl', 0) for pos in positions),
                'total_trades': 0,  # This should be counted from today's trades
                'winning_trades': 0,
                'losing_trades': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'total_commission': 0,
                'win_rate': 0
            }
        except Exception as e:
            self.logger.error(f"Error calculating daily performance: {e}")
            return None
    
    def monitor_position_risks(self):
        """Monitor individual position risks"""
        try:
            if not self.trading_engine:
                return
                
            positions = self.trading_engine.get_positions()
            
            for position in positions:
                unrealized_pnl_pct = position.get('unrealized_plpc', 0)
                
                # Check for significant losses
                if unrealized_pnl_pct < -0.10:  # 10% loss
                    self.logger.warning(f"Large loss detected: {position['symbol']} down {unrealized_pnl_pct:.2%}")
                    
                    # Consider automated stop-loss
                    if unrealized_pnl_pct < -0.15:  # 15% loss
                        self.consider_stop_loss(position)
        
        except Exception as e:
            self.logger.error(f"Error monitoring position risks: {e}")
    
    def consider_stop_loss(self, position):
        """Consider executing stop-loss for a position"""
        try:
            if not self.trading_engine:
                return
                
            symbol = position['symbol']
            
            self.logger.warning(f"Considering stop-loss for {symbol}")
            
            # Execute stop-loss
            trade_result = self.trading_engine.place_sell_order(symbol, 0, 0.5)
            
            if trade_result:
                self.logger.info(f"Stop-loss executed for {symbol}")
                if self.notification_system:
                    self.notification_system.send_risk_alert(
                        "Stop-Loss Executed",
                        f"Stop-loss order executed for {symbol} due to excessive losses"
                    )
        
        except Exception as e:
            self.logger.error(f"Error considering stop-loss: {e}")
    
    def get_memory_usage(self):
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except:
            return 0
    
    def send_weekly_report(self, report):
        """Send weekly performance report"""
        if not self.notification_system:
            return
            
        message = f"""
WEEKLY TRADING REPORT
Period: {report.get('period', 'Last 7 days')}
Total Trades: {report.get('total_trades', 0)}
Win Rate: {report.get('win_rate', 0):.2%}
Avg Return/Trade: ${report.get('avg_return_per_trade', 0):.2f}
Max Drawdown: {report.get('max_drawdown', 0):.2%}
Risk-Adjusted Return: {report.get('risk_adjusted_return', 0):.3f}
        """
        
        self.notification_system.send_notification("Weekly Report", message)
    
    def send_monthly_report(self, kpis, recommendations):
        """Send monthly KPI report"""
        if not self.notification_system:
            return
            
        message = f"""
MONTHLY KPI REPORT
Total Return: {kpis.get('total_return_pct', 0):.2f}%
Win Rate: {kpis.get('win_rate', 0):.2%}
Sharpe Ratio: {kpis.get('sharpe_ratio', 0):.3f}
Max Drawdown: {kpis.get('max_drawdown_pct', 0):.2f}%

Top Recommendations:
{chr(10).join([f"• {rec.get('recommendation', 'N/A')}" for rec in recommendations[:3]])}
        """
        
        self.notification_system.send_notification("Monthly KPI Report", message)
    
    def generate_daily_report(self, performance):
        """Generate and send daily report"""
        if not self.notification_system:
            return
            
        message = f"""
DAILY TRADING REPORT
Date: {performance.get('date', 'N/A')}
Total Equity: ${performance.get('ending_balance', 0):,.2f}
Daily P&L: ${performance.get('total_pnl', 0):,.2f}
Unrealized P&L: ${performance.get('unrealized_pnl', 0):,.2f}
Total Trades: {performance.get('total_trades', 0)}
        """
        
        self.notification_system.send_notification("Daily Report", message)
    
    def send_critical_alert(self, message):
        """Send critical system alert"""
        if self.notification_system:
            self.notification_system.send_risk_alert("CRITICAL SYSTEM ERROR", message)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum} - initiating graceful shutdown")
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of the trading bot"""
        self.logger.info("Initiating graceful shutdown...")
        
        try:
            # Set shutdown flag
            self.running = False
            self.shutdown_event.set()
            
            # Cancel any open orders
            if self.trading_engine:
                try:
                    # This would need to be implemented in trading engine
                    # self.trading_engine.cancel_all_orders()
                    pass
                except:
                    pass
            
            # Generate final report
            if self.trading_engine:
                final_performance = self.calculate_daily_performance()
                if final_performance:
                    self.generate_daily_report(final_performance)
            
            # Send shutdown notification
            if self.notification_system:
                self.notification_system.send_notification(
                    "Trading Bot Shutdown",
                    "Production trading bot has been shut down gracefully"
                )
            
            self.logger.info("Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def main():
    """Main entry point for the production trading bot"""
    try:
        # Create and start the trading bot
        bot = ProductionTradingBot()
        
        print("Starting Production Trading Bot...")
        print("Press Ctrl+C to stop the bot gracefully")
        
        # Start the bot
        success = bot.start_production_bot()
        
        if not success:
            print("Failed to start trading bot in full mode")
            print("Bot may be running in limited monitoring mode")
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()