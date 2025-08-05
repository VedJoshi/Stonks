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

from database.db_manager import ProductionDatabaseManager
from sentiment.sentiment_analyzer import SentimentAnalyzer
from trading.trading_engine import ProductionTradingEngine
from risk_management.risk_manager import RiskManager, DailyRiskCheck
from monitoring.dashboard import NotificationSystem, KPITracker, WeeklyReview
from monitoring.web_dashboard import LiveDashboard
from config.credentials import get_config

class ProductionTradingBot:
    def __init__(self):
        """Initialize the complete production trading bot"""
        self.config = get_config()
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Initialize logging
        self.setup_production_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize core components
        self.db_manager = ProductionDatabaseManager()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trading_engine = ProductionTradingEngine()
        self.risk_manager = RiskManager(self.trading_engine)
        self.daily_risk_check = DailyRiskCheck(self.trading_engine, self.db_manager)
        
        # Initialize monitoring components
        self.notification_system = NotificationSystem(self.config)
        self.kpi_tracker = KPITracker(self.db_manager)
        self.weekly_review = WeeklyReview(self.db_manager)
        self.live_dashboard = LiveDashboard(self.db_manager)
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("🚀 Production Trading Bot initialized successfully")
    
    def setup_production_logging(self):
        """Setup comprehensive production logging"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # File handlers for different log levels
        handlers = [
            # Main application log
            logging.FileHandler(f"{log_dir}/trading_bot.log"),
            # Error log
            logging.FileHandler(f"{log_dir}/errors.log"),
            # Trade log
            logging.FileHandler(f"{log_dir}/trades.log"),
            # Console output
            logging.StreamHandler(sys.stdout)
        ]
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        for handler in handlers:
            handler.setFormatter(detailed_formatter)
            if "errors" in str(handler.baseFilename):
                handler.setLevel(logging.ERROR)
            elif "trades" in str(handler.baseFilename):
                handler.setLevel(logging.INFO)
                handler.addFilter(lambda record: "TRADE" in record.getMessage())
            root_logger.addHandler(handler)
    
    def start_production_bot(self):
        """Start the production trading bot with full monitoring"""
        try:
            self.logger.info("🔥 STARTING PRODUCTION TRADING BOT")
            self.running = True
            
            # Perform startup checks
            if not self.perform_startup_checks():
                self.logger.error("❌ Startup checks failed - aborting")
                return False
            
            # Initialize database
            self.db_manager.initialize_database()
            
            # Schedule trading operations
            self.schedule_trading_operations()
            
            # Schedule monitoring and maintenance
            self.schedule_monitoring_operations()
            
            # Start main execution loop
            self.run_main_loop()
            
        except Exception as e:
            self.logger.error(f"❌ Critical error in production bot: {e}")
            self.send_critical_alert(f"Production bot crashed: {e}")
            return False
    
    def perform_startup_checks(self):
        """Perform comprehensive startup checks"""
        self.logger.info("🔍 Performing startup checks...")
        
        checks = [
            ("API Connection", self.check_api_connection),
            ("Database Access", self.check_database_access),
            ("Risk Limits", self.check_risk_limits),
            ("Market Status", self.check_market_status),
            ("Account Status", self.check_account_status)
        ]
        
        for check_name, check_func in checks:
            try:
                if check_func():
                    self.logger.info(f"✅ {check_name}: PASSED")
                else:
                    self.logger.error(f"❌ {check_name}: FAILED")
                    return False
            except Exception as e:
                self.logger.error(f"❌ {check_name}: ERROR - {e}")
                return False
        
        return True
    
    def check_api_connection(self):
        """Check API connection"""
        return self.trading_engine.test_connection()
    
    def check_database_access(self):
        """Check database access"""
        try:
            self.db_manager.test_connection()
            return True
        except:
            return False
    
    def check_risk_limits(self):
        """Check risk management settings"""
        return self.daily_risk_check.check_daily_limits()
    
    def check_market_status(self):
        """Check if market is open or will open soon"""
        return self.trading_engine.is_market_open_or_opening_soon()
    
    def check_account_status(self):
        """Check account status and buying power"""
        account = self.trading_engine.get_account()
        buying_power = float(account.buying_power)
        return buying_power > 1000  # Minimum required buying power
    
    def schedule_trading_operations(self):
        """Schedule all trading operations"""
        # Pre-market sentiment analysis
        schedule.every().day.at("08:00").do(self.pre_market_analysis)
        
        # Market open preparation
        schedule.every().day.at("09:25").do(self.market_open_preparation)
        
        # Main trading loop (every 5 minutes during market hours)
        schedule.every(5).minutes.do(self.trading_cycle)
        
        # End of day processing
        schedule.every().day.at("16:30").do(self.end_of_day_processing)
        
        self.logger.info("📅 Trading operations scheduled")
    
    def schedule_monitoring_operations(self):
        """Schedule monitoring and maintenance operations"""
        # Risk checks every 15 minutes
        schedule.every(15).minutes.do(self.risk_monitoring_cycle)
        
        # Database backup daily
        schedule.every().day.at("02:00").do(self.db_manager.backup_database)
        
        # Weekly performance review
        schedule.every().monday.at("06:00").do(self.weekly_performance_review)
        
        # Monthly KPI calculation
        schedule.every().month.do(self.monthly_kpi_review)
        
        self.logger.info("📊 Monitoring operations scheduled")
    
    def run_main_loop(self):
        """Main execution loop"""
        self.logger.info("🔄 Starting main execution loop")
        
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
        
        self.logger.info("🛑 Main execution loop stopped")
    
    def trading_cycle(self):
        """Main trading cycle - runs every 5 minutes during market hours"""
        if not self.trading_engine.is_market_open():
            return
        
        try:
            self.logger.info("🔄 Starting trading cycle")
            
            # 1. Check daily risk limits
            if not self.daily_risk_check.check_daily_limits():
                self.logger.warning("Daily risk limits exceeded - skipping trading cycle")
                return
            
            # 2. Analyze sentiment for watchlist stocks
            watchlist = self.get_watchlist()
            sentiment_results = {}
            
            for symbol in watchlist:
                sentiment_score = self.sentiment_analyzer.analyze_symbol_sentiment(symbol)
                sentiment_results[symbol] = sentiment_score
                
                # Store sentiment data
                self.db_manager.store_sentiment_data(symbol, sentiment_score, 'combined')
            
            # 3. Generate trading signals
            trading_signals = self.generate_trading_signals(sentiment_results)
            
            # 4. Execute trades based on signals
            for signal in trading_signals:
                self.execute_trade_signal(signal)
            
            self.logger.info(f"✅ Trading cycle completed - {len(trading_signals)} signals processed")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
    
    def generate_trading_signals(self, sentiment_results):
        """Generate trading signals based on sentiment analysis"""
        signals = []
        
        for symbol, sentiment_score in sentiment_results.items():
            # Get current position
            current_position = self.trading_engine.get_position(symbol)
            
            # Signal generation logic
            if sentiment_score > 0.7 and not current_position:
                # Strong positive sentiment - buy signal
                signals.append({
                    'symbol': symbol,
                    'action': 'BUY',
                    'sentiment_score': sentiment_score,
                    'confidence': 'HIGH' if sentiment_score > 0.8 else 'MEDIUM'
                })
            
            elif sentiment_score < -0.7 and current_position:
                # Strong negative sentiment - sell signal
                signals.append({
                    'symbol': symbol,
                    'action': 'SELL',
                    'sentiment_score': sentiment_score,
                    'confidence': 'HIGH' if sentiment_score < -0.8 else 'MEDIUM'
                })
        
        return signals
    
    def execute_trade_signal(self, signal):
        """Execute a trading signal with full risk management"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            sentiment_score = signal['sentiment_score']
            
            # Risk assessment
            risk_assessment = self.risk_manager.assess_trade_risk(
                symbol, action, sentiment_score
            )
            
            if not risk_assessment['approved']:
                self.logger.warning(f"Trade rejected by risk manager: {risk_assessment['reason']}")
                return False
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol, risk_assessment['risk_level']
            )
            
            # Execute trade
            if action == 'BUY':
                order = self.trading_engine.place_buy_order(symbol, position_size)
            else:
                order = self.trading_engine.place_sell_order(symbol, position_size)
            
            if order:
                # Log successful trade
                trade_info = {
                    'symbol': symbol,
                    'side': action,
                    'quantity': position_size,
                    'price': order.filled_avg_price or order.limit_price,
                    'sentiment_score': sentiment_score
                }
                
                self.logger.info(f"TRADE EXECUTED: {trade_info}")
                
                # Store trade in database
                self.db_manager.store_trade(
                    symbol, action, position_size, 
                    float(order.filled_avg_price or order.limit_price),
                    sentiment_score
                )
                
                # Send notification
                self.notification_system.send_trade_alert(trade_info)
                
                return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade signal: {e}")
            return False
    
    def pre_market_analysis(self):
        """Pre-market sentiment analysis and preparation"""
        self.logger.info("🌅 Starting pre-market analysis")
        
        try:
            # Get watchlist
            watchlist = self.get_watchlist()
            
            # Analyze pre-market sentiment
            pre_market_sentiment = {}
            for symbol in watchlist:
                sentiment = self.sentiment_analyzer.analyze_symbol_sentiment(symbol)
                pre_market_sentiment[symbol] = sentiment
            
            # Store pre-market data
            for symbol, sentiment in pre_market_sentiment.items():
                self.db_manager.store_sentiment_data(symbol, sentiment, 'pre_market')
            
            self.logger.info(f"✅ Pre-market analysis completed for {len(watchlist)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error in pre-market analysis: {e}")
    
    def market_open_preparation(self):
        """Prepare for market open"""
        self.logger.info("🔔 Market open preparation")
        
        try:
            # Check account status
            account = self.trading_engine.get_account()
            self.logger.info(f"Account equity: ${float(account.equity):,.2f}")
            self.logger.info(f"Buying power: ${float(account.buying_power):,.2f}")
            
            # Reset daily risk counters
            self.daily_risk_check.reset_daily_counters()
            
            # Check for any overnight news/events
            self.check_overnight_events()
            
            self.logger.info("✅ Market open preparation completed")
            
        except Exception as e:
            self.logger.error(f"Error in market open preparation: {e}")
    
    def end_of_day_processing(self):
        """End of day processing and reporting"""
        self.logger.info("🌆 Starting end-of-day processing")
        
        try:
            # Calculate daily performance
            daily_performance = self.calculate_daily_performance()
            
            # Store daily performance
            self.db_manager.store_daily_performance(daily_performance)
            
            # Generate daily report
            self.generate_daily_report(daily_performance)
            
            # Close any open positions if configured
            if self.config.get('CLOSE_POSITIONS_EOD', False):
                self.close_all_positions()
            
            self.logger.info("✅ End-of-day processing completed")
            
        except Exception as e:
            self.logger.error(f"Error in end-of-day processing: {e}")
    
    def risk_monitoring_cycle(self):
        """Risk monitoring cycle - runs every 15 minutes"""
        try:
            # Check for risk violations
            risk_status = self.daily_risk_check.check_daily_limits()
            
            if not risk_status:
                self.logger.warning("🚨 Risk limits violated")
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
                'api_connection': self.check_api_connection(),
                'database_status': self.check_database_access(),
                'memory_usage': self.get_memory_usage(),
                'active_positions': len(self.trading_engine.get_all_positions())
            }
            
            # Log health status
            self.logger.info(f"System Health: {health_status}")
            
            # Store health data
            self.db_manager.store_system_health(health_status)
            
        except Exception as e:
            self.logger.error(f"Error in system health check: {e}")
    
    def weekly_performance_review(self):
        """Generate weekly performance review"""
        try:
            self.logger.info("📊 Generating weekly performance review")
            
            weekly_report = self.weekly_review.generate_weekly_report()
            
            # Send weekly report
            self.send_weekly_report(weekly_report)
            
        except Exception as e:
            self.logger.error(f"Error in weekly performance review: {e}")
    
    def monthly_kpi_review(self):
        """Generate monthly KPI review"""
        try:
            self.logger.info("📈 Generating monthly KPI review")
            
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
            account = self.trading_engine.get_account()
            positions = self.trading_engine.get_all_positions()
            
            total_equity = float(account.equity)
            total_pnl = float(account.unrealized_pl) + float(account.realized_pl)
            
            return {
                'date': datetime.now().date(),
                'total_equity': total_equity,
                'total_pnl': total_pnl,
                'unrealized_pnl': float(account.unrealized_pl),
                'realized_pnl': float(account.realized_pl),
                'position_count': len(positions),
                'buying_power': float(account.buying_power)
            }
        except Exception as e:
            self.logger.error(f"Error calculating daily performance: {e}")
            return {}
    
    def monitor_position_risks(self):
        """Monitor individual position risks"""
        try:
            positions = self.trading_engine.get_all_positions()
            
            for position in positions:
                unrealized_pnl_pct = float(position.unrealized_plpc)
                
                # Check for significant losses
                if unrealized_pnl_pct < -0.10:  # 10% loss
                    self.logger.warning(f"Large loss detected: {position.symbol} down {unrealized_pnl_pct:.2%}")
                    
                    # Consider automated stop-loss
                    if unrealized_pnl_pct < -0.15:  # 15% loss
                        self.consider_stop_loss(position)
        
        except Exception as e:
            self.logger.error(f"Error monitoring position risks: {e}")
    
    def consider_stop_loss(self, position):
        """Consider executing stop-loss for a position"""
        try:
            symbol = position.symbol
            quantity = abs(int(position.qty))
            
            self.logger.warning(f"Considering stop-loss for {symbol}")
            
            # Execute stop-loss
            order = self.trading_engine.place_sell_order(symbol, quantity)
            
            if order:
                self.logger.info(f"Stop-loss executed for {symbol}")
                self.notification_system.send_risk_alert(
                    "Stop-Loss Executed",
                    f"Stop-loss order placed for {symbol} due to excessive losses"
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
    
    def check_overnight_events(self):
        """Check for overnight news and events"""
        # Placeholder for checking news APIs, earnings calendars, etc.
        pass
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = self.trading_engine.get_all_positions()
            
            for position in positions:
                symbol = position.symbol
                quantity = abs(int(position.qty))
                
                self.trading_engine.place_sell_order(symbol, quantity)
                
            self.logger.info(f"Closed {len(positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
    
    def send_weekly_report(self, report):
        """Send weekly performance report"""
        message = f"""
📊 WEEKLY TRADING REPORT
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
        message = f"""
📈 MONTHLY KPI REPORT
Total Return: {kpis.get('total_return_pct', 0):.2f}%
Win Rate: {kpis.get('win_rate', 0):.2%}
Sharpe Ratio: {kpis.get('sharpe_ratio', 0):.3f}
Max Drawdown: {kpis.get('max_drawdown_pct', 0):.2f}%

Top Recommendations:
{chr(10).join([f"• {rec['recommendation']}" for rec in recommendations[:3]])}
        """
        
        self.notification_system.send_notification("Monthly KPI Report", message)
    
    def generate_daily_report(self, performance):
        """Generate and send daily report"""
        message = f"""
📅 DAILY TRADING REPORT
Date: {performance.get('date', 'N/A')}
Total Equity: ${performance.get('total_equity', 0):,.2f}
Daily P&L: ${performance.get('total_pnl', 0):,.2f}
Active Positions: {performance.get('position_count', 0)}
Buying Power: ${performance.get('buying_power', 0):,.2f}
        """
        
        self.notification_system.send_notification("Daily Report", message)
    
    def send_critical_alert(self, message):
        """Send critical system alert"""
        self.notification_system.send_risk_alert("CRITICAL SYSTEM ERROR", message)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum} - initiating graceful shutdown")
        self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of the trading bot"""
        self.logger.info("🛑 Initiating graceful shutdown...")
        
        try:
            # Set shutdown flag
            self.running = False
            self.shutdown_event.set()
            
            # Close any open orders
            self.trading_engine.cancel_all_orders()
            
            # Generate final report
            final_performance = self.calculate_daily_performance()
            self.generate_daily_report(final_performance)
            
            # Send shutdown notification
            self.notification_system.send_notification(
                "Trading Bot Shutdown",
                "Production trading bot has been shut down gracefully"
            )
            
            self.logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


def main():
    """Main entry point for the production trading bot"""
    try:
        # Create and start the trading bot
        bot = ProductionTradingBot()
        
        print("🚀 Starting Production Trading Bot...")
        print("Press Ctrl+C to stop the bot gracefully")
        
        # Start the bot
        success = bot.start_production_bot()
        
        if not success:
            print("❌ Failed to start trading bot")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
