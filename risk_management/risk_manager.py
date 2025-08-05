import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class RiskManager:
    def __init__(self, max_daily_loss=25.0, max_position_size=50.0, max_positions=3):
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.logger = logging.getLogger(__name__)
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.position_count = 0
        self.risk_events = []
        
        self.logger.info(f"Risk Manager initialized: Max daily loss: ${max_daily_loss}, Max position: ${max_position_size}")
    
    def check_daily_loss_limit(self, current_pnl):
        """Check if daily loss limit is exceeded"""
        if abs(current_pnl) >= self.max_daily_loss:
            self.log_risk_event(
                "DAILY_LOSS_LIMIT_EXCEEDED",
                f"Daily P&L: ${current_pnl:.2f} exceeds limit of ${self.max_daily_loss}",
                "CRITICAL"
            )
            return False
        return True
    
    def check_position_size_limit(self, trade_size):
        """Check if trade size exceeds limits"""
        if trade_size > self.max_position_size:
            self.log_risk_event(
                "POSITION_SIZE_EXCEEDED",
                f"Trade size ${trade_size:.2f} exceeds limit of ${self.max_position_size}",
                "ERROR"
            )
            return False
        return True
    
    def check_max_positions(self, current_positions):
        """Check if maximum position count is exceeded"""
        if len(current_positions) >= self.max_positions:
            self.log_risk_event(
                "MAX_POSITIONS_EXCEEDED",
                f"Current positions: {len(current_positions)} exceeds limit of {self.max_positions}",
                "WARNING"
            )
            return False
        return True
    
    def evaluate_trade_risk(self, symbol, trade_size, sentiment_confidence, current_positions):
        """Comprehensive trade risk evaluation"""
        risk_factors = []
        
        # Position size check
        if not self.check_position_size_limit(trade_size):
            risk_factors.append("Position size too large")
        
        # Position count check
        if not self.check_max_positions(current_positions):
            risk_factors.append("Too many open positions")
        
        # Confidence check
        if sentiment_confidence < 0.6:
            risk_factors.append(f"Low sentiment confidence: {sentiment_confidence:.3f}")
        
        # Concentration risk check
        total_exposure = sum(pos.get('market_value', 0) for pos in current_positions)
        if trade_size / (total_exposure + trade_size) > 0.5:
            risk_factors.append("High concentration risk")
        
        # Symbol-specific checks
        existing_position = next((pos for pos in current_positions if pos['symbol'] == symbol), None)
        if existing_position:
            risk_factors.append(f"Already have position in {symbol}")
        
        return {
            'approved': len(risk_factors) == 0,
            'risk_factors': risk_factors,
            'risk_score': len(risk_factors) / 5.0  # Normalize to 0-1
        }
    
    def calculate_position_risk(self, symbol, entry_price, quantity, stop_loss_price):
        """Calculate position risk metrics"""
        position_value = entry_price * quantity
        max_loss = (entry_price - stop_loss_price) * quantity
        risk_percent = (max_loss / position_value) * 100
        
        return {
            'position_value': position_value,
            'max_loss': max_loss,
            'risk_percent': risk_percent,
            'risk_reward_ratio': 0  # Would need take profit to calculate
        }
    
    def monitor_portfolio_risk(self, account_info, positions):
        """Monitor overall portfolio risk"""
        try:
            total_value = account_info.get('portfolio_value', 0)
            total_unrealized_pnl = sum(pos.get('unrealized_pl', 0) for pos in positions)
            
            # Portfolio concentration
            largest_position = max([abs(pos.get('market_value', 0)) for pos in positions] + [0])
            concentration_risk = largest_position / total_value if total_value > 0 else 0
            
            # Drawdown calculation
            equity = account_info.get('equity', 0)
            last_equity = account_info.get('last_equity', equity)
            drawdown = (last_equity - equity) / last_equity if last_equity > 0 else 0
            
            risk_metrics = {
                'portfolio_value': total_value,
                'unrealized_pnl': total_unrealized_pnl,
                'concentration_risk': concentration_risk,
                'drawdown': drawdown,
                'position_count': len(positions),
                'cash_ratio': account_info.get('cash', 0) / total_value if total_value > 0 else 0
            }
            
            # Risk alerts
            if concentration_risk > 0.4:
                self.log_risk_event(
                    "HIGH_CONCENTRATION",
                    f"Largest position represents {concentration_risk*100:.1f}% of portfolio",
                    "WARNING"
                )
            
            if drawdown > 0.1:
                self.log_risk_event(
                    "HIGH_DRAWDOWN",
                    f"Portfolio drawdown: {drawdown*100:.1f}%",
                    "WARNING"
                )
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring portfolio risk: {e}")
            return {}
    
    def log_risk_event(self, event_type, description, severity="INFO"):
        """Log risk management events"""
        risk_event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'description': description,
            'severity': severity
        }
        
        self.risk_events.append(risk_event)
        
        # Log to system logger
        if severity == "CRITICAL":
            self.logger.critical(f"RISK EVENT: {event_type} - {description}")
        elif severity == "ERROR":
            self.logger.error(f"RISK EVENT: {event_type} - {description}")
        elif severity == "WARNING":
            self.logger.warning(f"RISK EVENT: {event_type} - {description}")
        else:
            self.logger.info(f"RISK EVENT: {event_type} - {description}")
    
    def get_recent_risk_events(self, hours=24):
        """Get recent risk events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [event for event in self.risk_events if event['timestamp'] > cutoff_time]
    
    def reset_daily_tracking(self):
        """Reset daily risk tracking"""
        self.daily_pnl = 0.0
        self.position_count = 0
        # Keep risk events for historical analysis
        self.logger.info("Daily risk tracking reset")


class DailyRiskCheck:
    def __init__(self, trading_engine, db_manager):
        self.engine = trading_engine
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
    
    def pre_market_check(self):
        """Complete pre-market safety verification"""
        checks = {
            'account_status': self.verify_account_health(),
            'position_limits': self.check_position_exposure(),
            'cash_availability': self.verify_buying_power(),
            'api_connections': self.test_all_connections(),
            'risk_parameters': self.validate_risk_settings(),
            'market_conditions': self.assess_market_volatility()
        }
        
        # All checks must pass
        if all(checks.values()):
            self.db.log_risk_event("DAILY_CHECK", "All pre-market checks passed", severity="INFO")
            return True
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            self.db.log_risk_event("DAILY_CHECK", f"Failed checks: {failed_checks}", severity="ERROR")
            return False
    
    def verify_account_health(self):
        """Check account is in good standing"""
        try:
            account = self.engine.api.get_account()
            
            # Critical checks
            if account.trading_blocked or account.account_blocked:
                return False
            
            # Equity checks
            equity = float(account.equity)
            if equity < 500:  # Emergency threshold
                self.db.log_risk_event("LOW_EQUITY", f"Account equity below $500: ${equity}", severity="CRITICAL")
                return False
            
            # Day trading buying power (if applicable)
            if hasattr(account, 'daytrading_buying_power'):
                dt_bp = float(account.daytrading_buying_power)
                if dt_bp < 100:
                    self.db.log_risk_event("LOW_DAYTRADING_BP", f"Day trading BP low: ${dt_bp}", severity="WARNING")
            
            return True
            
        except Exception as e:
            self.db.log_risk_event("ACCOUNT_CHECK_ERROR", str(e), severity="ERROR")
            return False
    
    def check_position_exposure(self):
        """Check position exposure limits"""
        try:
            positions = self.engine.get_positions()
            
            if len(positions) > self.engine.config['MAX_OPEN_POSITIONS']:
                return False
            
            # Check individual position sizes
            for position in positions:
                if abs(position['market_value']) > self.engine.config['MAX_SINGLE_TRADE']:
                    self.db.log_risk_event(
                        "OVERSIZED_POSITION", 
                        f"{position['symbol']}: ${abs(position['market_value']):.2f}",
                        severity="WARNING"
                    )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Position exposure check failed: {e}")
            return False
    
    def verify_buying_power(self):
        """Verify adequate buying power"""
        try:
            account_info = self.engine.get_account_info()
            if not account_info:
                return False
            
            buying_power = account_info['buying_power']
            min_required = self.engine.config['MAX_SINGLE_TRADE']
            
            if buying_power < min_required:
                self.db.log_risk_event(
                    "INSUFFICIENT_BUYING_POWER",
                    f"Buying power: ${buying_power:.2f}, Required: ${min_required:.2f}",
                    severity="WARNING"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Buying power check failed: {e}")
            return False
    
    def test_all_connections(self):
        """Test all API connections"""
        try:
            # Test Alpaca connection
            account = self.engine.api.get_account()
            
            # Test market data
            clock = self.engine.api.get_clock()
            
            return True
            
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            return False
    
    def validate_risk_settings(self):
        """Validate risk management settings"""
        config = self.engine.config
        
        # Validate key parameters are within safe ranges
        if config['STOP_LOSS_PERCENT'] > 0.1:  # 10% max stop loss
            return False
        
        if config['MAX_SINGLE_TRADE'] > config['MAX_TOTAL_RISK']:
            return False
        
        if config['MAX_DAILY_TRADES'] > 20:  # Prevent overtrading
            return False
        
        return True
    
    def assess_market_volatility(self):
        """Assess current market conditions"""
        try:
            # Check VIX or other volatility measures
            # This is a placeholder - in production you'd check actual volatility indicators
            return True
            
        except Exception as e:
            self.logger.error(f"Market condition assessment failed: {e}")
            return True  # Default to allowing trading
    
    def emergency_protocols(self):
        """Emergency response procedures"""
        # 1. Immediate position closure if loss > 50% of max daily limit
        if abs(self.engine.daily_pnl) > (self.engine.config['DAILY_LOSS_LIMIT'] * 0.5):
            self.engine.emergency_close_all_positions()
            return "EMERGENCY_CLOSE_EXECUTED"
        
        # 2. Trading halt if 3 consecutive losses
        recent_trades = self.db.get_recent_trades(count=3)
        if len(recent_trades) == 3 and all(trade.get('pnl', 0) < 0 for trade in recent_trades):
            self.engine.circuit_breaker_triggered = True
            return "TRADING_HALT_ACTIVATED"
        
        # 3. API failure fallback
        if not self.test_all_connections():
            # Send email alert, stop trading
            return "API_FAILURE_PROTOCOL"
        
        return "NO_EMERGENCY_ACTION_NEEDED"
