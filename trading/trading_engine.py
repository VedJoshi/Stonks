import alpaca_trade_api as tradeapi
import logging
from datetime import datetime, timedelta
import time
import requests
from typing import Dict, List, Optional
import yfinance as yf

class ProductionTradingEngine:
    def __init__(self, api_key, secret_key, base_url, config):
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Real money safety checks
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        self.last_trade_time = {}
        self.circuit_breaker_triggered = False
        
        # Verify account connection
        self.verify_account_connection()
    
    def verify_account_connection(self):
        """Verify real account connection and safety"""
        try:
            account = self.api.get_account()
            
            # Safety checks for real account
            if account.trading_blocked:
                raise Exception("Account is blocked from trading!")
            
            if account.account_blocked:
                raise Exception("Account is completely blocked!")
            
            if float(account.cash) < self.config['MAX_SINGLE_TRADE']:
                self.logger.warning(f"Low cash balance: ${account.cash}")
            
            self.logger.info(f"Real account connected successfully")
            self.logger.info(f"Account Value: ${float(account.portfolio_value):,.2f}")
            self.logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
            self.logger.info(f"Cash: ${float(account.cash):,.2f}")
            
        except Exception as e:
            self.logger.critical(f"Account connection failed: {e}")
            raise
    
    def check_circuit_breaker(self):
        """Enhanced circuit breaker for real money protection"""
        try:
            # Check daily loss limit
            if abs(self.daily_pnl) >= self.config['DAILY_LOSS_LIMIT']:
                self.circuit_breaker_triggered = True
                self.logger.critical(f"CIRCUIT BREAKER: Daily loss limit reached: ${self.daily_pnl:.2f}")
                return False
            
            # Check daily trade limit
            if self.daily_trades_count >= self.config['MAX_DAILY_TRADES']:
                self.logger.warning(f"Daily trade limit reached: {self.daily_trades_count}")
                return False
            
            # Check account status
            account = self.api.get_account()
            if account.trading_blocked or account.account_blocked:
                self.logger.critical("Account is blocked!")
                return False
            
            # Check if market is open
            clock = self.api.get_clock()
            if not clock.is_open:
                self.logger.info("Market is closed")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Circuit breaker check failed: {e}")
            return False
    
    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        try:
            # Try Alpaca first
            try:
                latest_trade = self.api.get_latest_trade(symbol)
                return float(latest_trade.price)
            except:
                # Fallback to yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    return float(data['Close'].iloc[-1])
                else:
                    self.logger.error(f"No price data available for {symbol}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None
    
    def calculate_precise_position_size(self, symbol, price, sentiment_strength):
        """Calculate position size with real money precision"""
        try:
            account = self.api.get_account()
            available_cash = float(account.buying_power)
            
            # Base position size from config
            base_size = min(
                self.config['MAX_SINGLE_TRADE'],
                available_cash * self.config['POSITION_SIZE_PERCENT']
            )
            
            # Adjust based on sentiment strength (0.5 to 1.5 multiplier)
            sentiment_multiplier = 0.5 + (sentiment_strength * 1.0)
            adjusted_size = base_size * sentiment_multiplier
            
            # Calculate shares
            shares = int(adjusted_size / price)
            
            # Ensure minimum viable position
            if shares < 1 and available_cash > price:
                shares = 1
            
            # Final safety check
            total_cost = shares * price
            if total_cost > self.config['MAX_SINGLE_TRADE']:
                shares = int(self.config['MAX_SINGLE_TRADE'] / price)
            
            self.logger.info(f"Position size calculation for {symbol}:")
            self.logger.info(f"  Price: ${price:.2f}")
            self.logger.info(f"  Sentiment strength: {sentiment_strength:.2f}")
            self.logger.info(f"  Calculated shares: {shares}")
            self.logger.info(f"  Total cost: ${shares * price:.2f}")
            
            return shares
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return 0
    
    def place_buy_order_with_stops(self, symbol, sentiment_score, confidence):
        """Place buy order with automatic stop-loss and take-profit"""
        try:
            if not self.check_circuit_breaker():
                return None
            
            # Check cooldown period
            if symbol in self.last_trade_time:
                time_since_last = time.time() - self.last_trade_time[symbol]
                if time_since_last < self.config['TRADING_COOLDOWN']:
                    self.logger.info(f"Cooldown active for {symbol}: {int(self.config['TRADING_COOLDOWN'] - time_since_last)}s remaining")
                    return None
            
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
            
            # Calculate position size
            quantity = self.calculate_precise_position_size(symbol, current_price, confidence)
            if quantity <= 0:
                self.logger.warning(f"Invalid quantity calculated for {symbol}")
                return None
            
            # Calculate stop-loss and take-profit levels
            stop_loss_price = current_price * (1 - self.config['STOP_LOSS_PERCENT'])
            take_profit_price = current_price * (1 + self.config['TAKE_PROFIT_PERCENT'])
            
            self.logger.info(f"Placing BUY order for {symbol}:")
            self.logger.info(f"  Quantity: {quantity} shares")
            self.logger.info(f"  Entry Price: ${current_price:.2f}")
            self.logger.info(f"  Stop Loss: ${stop_loss_price:.2f} (-{self.config['STOP_LOSS_PERCENT']*100:.1f}%)")
            self.logger.info(f"  Take Profit: ${take_profit_price:.2f} (+{self.config['TAKE_PROFIT_PERCENT']*100:.1f}%)")
            self.logger.info(f"  Sentiment: {sentiment_score:.3f}, Confidence: {confidence:.3f}")
            
            # Place the buy order
            buy_order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            
            # Wait for fill confirmation
            time.sleep(2)
            buy_order = self.api.get_order(buy_order.id)
            
            if buy_order.status == 'filled':
                filled_price = float(buy_order.filled_avg_price)
                
                # Recalculate stops based on actual fill price
                stop_loss_price = filled_price * (1 - self.config['STOP_LOSS_PERCENT'])
                take_profit_price = filled_price * (1 + self.config['TAKE_PROFIT_PERCENT'])
                
                # Place bracket orders (stop-loss and take-profit)
                try:
                    # Stop-loss order
                    stop_order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='sell',
                        type='stop',
                        stop_price=round(stop_loss_price, 2),
                        time_in_force='gtc'
                    )
                    
                    # Take-profit order
                    limit_order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='sell',
                        type='limit',
                        limit_price=round(take_profit_price, 2),
                        time_in_force='gtc'
                    )
                    
                    self.logger.info(f"Buy order filled at ${filled_price:.2f}")
                    self.logger.info(f"Stop-loss placed: {stop_order.id}")
                    self.logger.info(f"Take-profit placed: {limit_order.id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to place stop orders: {e}")
                
                # Update tracking
                self.daily_trades_count += 1
                self.last_trade_time[symbol] = time.time()
                
                return {
                    'order_id': buy_order.id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': filled_price,
                    'side': 'buy',
                    'sentiment_score': sentiment_score,
                    'confidence': confidence,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price
                }
            else:
                self.logger.warning(f"Buy order not filled: {buy_order.status}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing buy order for {symbol}: {e}")
            return None
    
    def place_sell_order(self, symbol, sentiment_score, confidence):
        """Place sell order for existing position"""
        try:
            # Get current position
            try:
                position = self.api.get_position(symbol)
                if not position or int(position.qty) <= 0:
                    self.logger.warning(f"No position found for {symbol}")
                    return None
                    
                quantity = int(position.qty)
            except:
                self.logger.warning(f"No position found for {symbol}")
                return None
            
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
            
            self.logger.info(f"Placing SELL order for {symbol}:")
            self.logger.info(f"  Quantity: {quantity} shares")
            self.logger.info(f"  Current Price: ${current_price:.2f}")
            self.logger.info(f"  Sentiment: {sentiment_score:.3f}, Confidence: {confidence:.3f}")
            
            # Place the sell order
            sell_order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            
            # Wait for fill confirmation
            time.sleep(2)
            sell_order = self.api.get_order(sell_order.id)
            
            if sell_order.status == 'filled':
                filled_price = float(sell_order.filled_avg_price)
                
                self.logger.info(f"Sell order filled at ${filled_price:.2f}")
                
                # Update tracking
                self.daily_trades_count += 1
                self.last_trade_time[symbol] = time.time()
                
                return {
                    'order_id': sell_order.id,
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': filled_price,
                    'side': 'sell',
                    'sentiment_score': sentiment_score,
                    'confidence': confidence
                }
            else:
                self.logger.warning(f"Sell order not filled: {sell_order.status}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing sell order for {symbol}: {e}")
            return None
    
    def generate_trading_signal(self, symbol, sentiment_data):
        """Generate trading signal based on sentiment"""
        try:
            sentiment_score = sentiment_data['combined_sentiment']
            confidence = sentiment_data['combined_confidence']
            
            # Check confidence threshold
            if confidence < self.config['CONFIDENCE_THRESHOLD']:
                return {
                    'signal': 'hold',
                    'reason': f'Low confidence: {confidence:.3f}',
                    'strength': 0
                }
            
            # Generate signal based on sentiment thresholds
            if sentiment_score >= self.config['SENTIMENT_BUY_THRESHOLD']:
                signal_strength = min(1.0, sentiment_score / self.config['SENTIMENT_BUY_THRESHOLD'])
                return {
                    'signal': 'buy',
                    'reason': f'Bullish sentiment: {sentiment_score:.3f}',
                    'strength': signal_strength
                }
            elif sentiment_score <= self.config['SENTIMENT_SELL_THRESHOLD']:
                signal_strength = min(1.0, abs(sentiment_score) / abs(self.config['SENTIMENT_SELL_THRESHOLD']))
                return {
                    'signal': 'sell',
                    'reason': f'Bearish sentiment: {sentiment_score:.3f}',
                    'strength': signal_strength
                }
            else:
                return {
                    'signal': 'hold',
                    'reason': f'Neutral sentiment: {sentiment_score:.3f}',
                    'strength': 0
                }
                
        except Exception as e:
            self.logger.error(f"Error generating trading signal for {symbol}: {e}")
            return {
                'signal': 'hold',
                'reason': f'Error: {str(e)}',
                'strength': 0
            }
    
    def get_real_time_pnl(self):
        """Get real-time P&L for circuit breaker"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()
            
            unrealized_pnl = sum(float(pos.unrealized_pl) for pos in positions)
            
            # Get today's realized P&L from closed trades
            # This is simplified - in production you'd track this more precisely
            today_start = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
            
            self.daily_pnl = unrealized_pnl  # Simplified for this example
            
            return {
                'total_pnl': float(account.equity) - float(account.last_equity),
                'unrealized_pnl': unrealized_pnl,
                'account_value': float(account.equity)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting P&L: {e}")
            return {'total_pnl': 0, 'unrealized_pnl': 0, 'account_value': 0}
    
    def get_account_info(self):
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'equity': float(account.equity),
                'last_equity': float(account.last_equity),
                'daytrading_buying_power': float(account.daytrading_buying_power) if hasattr(account, 'daytrading_buying_power') else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None
    
    def get_positions(self):
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': pos.symbol,
                'qty': int(pos.qty),
                'side': 'long' if int(pos.qty) > 0 else 'short',
                'market_value': float(pos.market_value),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_plpc': float(pos.unrealized_plpc)
            } for pos in positions]
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def emergency_close_all_positions(self):
        """Emergency function to close all positions"""
        try:
            self.logger.critical("EMERGENCY: Closing all positions!")
            
            positions = self.api.list_positions()
            
            for position in positions:
                try:
                    self.api.submit_order(
                        symbol=position.symbol,
                        qty=abs(int(float(position.qty))),
                        side='sell' if float(position.qty) > 0 else 'buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    self.logger.info(f"Emergency close order placed for {position.symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to close {position.symbol}: {e}")
            
            # Cancel all open orders
            open_orders = self.api.list_orders(status='open')
            for order in open_orders:
                try:
                    self.api.cancel_order(order.id)
                except Exception as e:
                    self.logger.error(f"Failed to cancel order {order.id}: {e}")
                    
        except Exception as e:
            self.logger.critical(f"Emergency close failed: {e}")
