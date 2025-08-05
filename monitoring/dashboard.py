import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
import logging
from datetime import datetime

class NotificationSystem:
    def __init__(self, config):
        self.config = config
        self.email_enabled = config.get('EMAIL_NOTIFICATIONS', False)
        self.telegram_enabled = config.get('TELEGRAM_NOTIFICATIONS', False)
        self.logger = logging.getLogger(__name__)
        
    def send_trade_alert(self, trade_info):
        """Send immediate trade execution alerts"""
        message = f"""
🔔 TRADE EXECUTED
Symbol: {trade_info['symbol']}
Side: {trade_info['side'].upper()}
Quantity: {trade_info['quantity']} shares
Price: ${trade_info['price']:.2f}
Total: ${trade_info['quantity'] * trade_info['price']:.2f}
Sentiment: {trade_info['sentiment_score']:.3f}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        self.send_notification("Trade Alert", message, priority="HIGH")
    
    def send_risk_alert(self, alert_type, message):
        """Send risk management alerts"""
        risk_message = f"""
🚨 RISK ALERT: {alert_type}
{message}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Account Status: Check dashboard immediately
        """
        
        self.send_notification("RISK ALERT", risk_message, priority="CRITICAL")
    
    def send_notification(self, subject, message, priority="NORMAL"):
        """Send notification via all enabled channels"""
        try:
            if self.email_enabled:
                self.send_email(subject, message)
            
            if self.telegram_enabled:
                self.send_telegram(message)
                
        except Exception as e:
            self.logger.error(f"Notification failed: {e}")
    
    def send_email(self, subject, body):
        """Send email notification"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.config['EMAIL_FROM']
            msg['To'] = self.config['EMAIL_TO']
            msg['Subject'] = f"Trading Bot: {subject}"
            
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(self.config['SMTP_SERVER'], self.config['SMTP_PORT'])
            server.starttls()
            server.login(self.config['EMAIL_FROM'], self.config['EMAIL_PASSWORD'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent: {subject}")
            
        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")
    
    def send_telegram(self, message):
        """Send Telegram notification"""
        try:
            url = f"https://api.telegram.org/bot{self.config['TELEGRAM_BOT_TOKEN']}/sendMessage"
            data = {
                'chat_id': self.config['TELEGRAM_CHAT_ID'],
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            self.logger.info("Telegram notification sent")
            
        except Exception as e:
            self.logger.error(f"Telegram notification failed: {e}")


class KPITracker:
    def __init__(self, db_manager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
    
    def calculate_monthly_kpis(self):
        """Calculate comprehensive KPIs for performance review"""
        performance = self.db.get_performance_summary(days=30)
        
        kpis = {
            # Profitability Metrics
            'total_return_pct': self.calculate_total_return(performance),
            'monthly_roi': self.calculate_monthly_roi(performance),
            'profit_factor': self.calculate_profit_factor(performance),
            'average_trade_pnl': self.calculate_avg_trade_pnl(performance),
            
            # Risk Metrics
            'max_drawdown_pct': self.calculate_max_drawdown(performance),
            'sharpe_ratio': self.calculate_sharpe_ratio(performance),
            'win_rate': self.calculate_win_rate(performance),
            'risk_reward_ratio': self.calculate_risk_reward_ratio(performance),
            
            # Efficiency Metrics
            'trades_per_week': self.calculate_trading_frequency(performance),
            'avg_hold_time': self.calculate_avg_hold_time(performance),
            'sentiment_accuracy': self.calculate_sentiment_accuracy(performance),
            
            # System Metrics
            'uptime_percentage': self.calculate_uptime(),
            'api_error_rate': self.calculate_api_error_rate(),
            'execution_latency': self.calculate_execution_latency()
        }
        
        return kpis
    
    def calculate_total_return(self, performance):
        """Calculate total return percentage"""
        try:
            daily_perf = performance['daily_performance']
            if daily_perf.empty:
                return 0
            
            starting_balance = daily_perf['starting_balance'].iloc[-1]
            ending_balance = daily_perf['ending_balance'].iloc[0]
            
            return (ending_balance - starting_balance) / starting_balance * 100
        except:
            return 0
    
    def calculate_monthly_roi(self, performance):
        """Calculate monthly ROI"""
        return self.calculate_total_return(performance)
    
    def calculate_profit_factor(self, performance):
        """Calculate profit factor (gross profit / gross loss)"""
        try:
            daily_perf = performance['daily_performance']
            if daily_perf.empty:
                return 0
            
            total_wins = daily_perf['largest_win'].sum()
            total_losses = abs(daily_perf['largest_loss'].sum())
            
            return total_wins / total_losses if total_losses > 0 else 0
        except:
            return 0
    
    def calculate_avg_trade_pnl(self, performance):
        """Calculate average trade P&L"""
        try:
            trade_stats = performance['trade_statistics']
            if trade_stats.empty:
                return 0
            
            return trade_stats['avg_pnl'].iloc[0] if not trade_stats.empty else 0
        except:
            return 0
    
    def calculate_max_drawdown(self, performance):
        """Calculate maximum drawdown"""
        try:
            daily_perf = performance['daily_performance']
            if daily_perf.empty:
                return 0
            
            return daily_perf['max_drawdown'].min()
        except:
            return 0
    
    def calculate_sharpe_ratio(self, performance):
        """Calculate Sharpe ratio"""
        try:
            daily_perf = performance['daily_performance']
            if daily_perf.empty:
                return 0
            
            # Simplified Sharpe ratio calculation
            returns = daily_perf['total_pnl'].pct_change().dropna()
            if returns.std() == 0:
                return 0
            
            return returns.mean() / returns.std() * (252 ** 0.5)  # Annualized
        except:
            return 0
    
    def calculate_win_rate(self, performance):
        """Calculate win rate"""
        try:
            daily_perf = performance['daily_performance']
            if daily_perf.empty:
                return 0
            
            total_trades = daily_perf['total_trades'].sum()
            winning_trades = daily_perf['winning_trades'].sum()
            
            return winning_trades / total_trades if total_trades > 0 else 0
        except:
            return 0
    
    def calculate_risk_reward_ratio(self, performance):
        """Calculate risk-reward ratio"""
        try:
            daily_perf = performance['daily_performance']
            if daily_perf.empty:
                return 0
            
            avg_win = daily_perf['avg_win'].mean()
            avg_loss = abs(daily_perf['avg_loss'].mean())
            
            return avg_win / avg_loss if avg_loss > 0 else 0
        except:
            return 0
    
    def calculate_trading_frequency(self, performance):
        """Calculate trades per week"""
        try:
            daily_perf = performance['daily_performance']
            if daily_perf.empty:
                return 0
            
            total_trades = daily_perf['total_trades'].sum()
            days = len(daily_perf)
            
            return (total_trades / days) * 7 if days > 0 else 0
        except:
            return 0
    
    def calculate_avg_hold_time(self, performance):
        """Calculate average holding time"""
        try:
            trade_stats = performance['trade_statistics']
            if trade_stats.empty:
                return 0
            
            return trade_stats['avg_hold_time'].iloc[0] if not trade_stats.empty else 0
        except:
            return 0
    
    def calculate_sentiment_accuracy(self, performance):
        """Calculate sentiment prediction accuracy"""
        # This would require more detailed trade analysis
        # Placeholder implementation
        return 0.65
    
    def calculate_uptime(self):
        """Calculate system uptime percentage"""
        # Placeholder implementation
        return 0.95
    
    def calculate_api_error_rate(self):
        """Calculate API error rate"""
        # Placeholder implementation
        return 0.02
    
    def calculate_execution_latency(self):
        """Calculate average execution latency"""
        # Placeholder implementation
        return 150  # milliseconds
    
    def generate_optimization_recommendations(self, kpis):
        """Generate data-driven optimization recommendations"""
        recommendations = []
        
        if kpis['win_rate'] < 0.45:
            recommendations.append({
                'area': 'Signal Quality',
                'issue': 'Low win rate',
                'recommendation': 'Increase sentiment confidence threshold',
                'priority': 'HIGH'
            })
        
        if kpis['risk_reward_ratio'] < 1.5:
            recommendations.append({
                'area': 'Risk Management',  
                'issue': 'Poor risk-reward ratio',
                'recommendation': 'Widen take-profit or tighten stop-loss',
                'priority': 'HIGH'
            })
        
        if kpis['sentiment_accuracy'] < 0.6:
            recommendations.append({
                'area': 'Data Quality',
                'issue': 'Low sentiment accuracy',
                'recommendation': 'Review sentiment sources and filtering',
                'priority': 'MEDIUM'
            })
        
        if kpis['max_drawdown_pct'] < -0.15:
            recommendations.append({
                'area': 'Risk Management',
                'issue': 'High drawdown',
                'recommendation': 'Reduce position sizes or improve stop-loss strategy',
                'priority': 'HIGH'
            })
        
        return recommendations


class WeeklyReview:
    def __init__(self, db_manager):
        self.db = db_manager
        self.kpi_tracker = KPITracker(db_manager)
        self.logger = logging.getLogger(__name__)
    
    def generate_weekly_report(self):
        """Generate comprehensive weekly performance report"""
        performance = self.db.get_performance_summary(days=7)
        kpis = self.kpi_tracker.calculate_monthly_kpis()
        
        report = {
            'period': '7 days',
            'total_trades': self.calculate_total_trades(performance),
            'win_rate': kpis['win_rate'],
            'avg_return_per_trade': kpis['average_trade_pnl'],
            'max_drawdown': kpis['max_drawdown_pct'],
            'risk_adjusted_return': kpis['sharpe_ratio'],
            'sentiment_accuracy': kpis['sentiment_accuracy'],
            'recommendations': self.kpi_tracker.generate_optimization_recommendations(kpis),
            'risk_events': self.get_weekly_risk_events(performance)
        }
        
        return report
    
    def calculate_total_trades(self, performance):
        """Calculate total trades in period"""
        try:
            trade_stats = performance['trade_statistics']
            return trade_stats['total_trades'].iloc[0] if not trade_stats.empty else 0
        except:
            return 0
    
    def get_weekly_risk_events(self, performance):
        """Get risk events for the week"""
        try:
            risk_events = performance['risk_events']
            return risk_events.to_dict('records') if not risk_events.empty else []
        except:
            return []
