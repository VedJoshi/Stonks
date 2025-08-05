#!/usr/bin/env python3
"""
Quick Setup Script for Trading Bot
This script will help you get the trading bot running quickly
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully")
        return True
    except Exception as e:
        print(f"Failed to install requirements: {e}")
        return False

def create_config_template():
    """Create a basic config template"""
    config_dir = "config"
    os.makedirs(config_dir, exist_ok=True)
    
    config_content = '''"""
Configuration file for Trading Bot
Replace the placeholder values with your actual API keys
"""

def get_config():
    return {
        # Trading Configuration
        'MAX_DAILY_TRADES': 10,
        'DAILY_LOSS_LIMIT': 100.0,
        'MAX_SINGLE_TRADE': 500.0,
        'POSITION_SIZE_PERCENT': 0.1,
        'STOP_LOSS_PERCENT': 0.05,
        'TAKE_PROFIT_PERCENT': 0.10,
        'CONFIDENCE_THRESHOLD': 0.6,
        'SENTIMENT_BUY_THRESHOLD': 0.7,
        'SENTIMENT_SELL_THRESHOLD': -0.7,
        'TRADING_COOLDOWN': 300,
        'MAX_POSITIONS': 3,
        
        # API Keys (REPLACE WITH YOUR ACTUAL KEYS)
        'ALPACA_API_KEY': 'YOUR_ALPACA_API_KEY_HERE',
        'ALPACA_SECRET_KEY': 'YOUR_ALPACA_SECRET_KEY_HERE',
        'ALPACA_BASE_URL': 'https://paper-api.alpaca.markets',  # Use paper trading URL
        
        # Reddit API (REPLACE WITH YOUR ACTUAL KEYS)
        'REDDIT_CLIENT_ID': 'YOUR_REDDIT_CLIENT_ID_HERE',
        'REDDIT_CLIENT_SECRET': 'YOUR_REDDIT_CLIENT_SECRET_HERE',
        'REDDIT_USER_AGENT': 'TradingBot/1.0 by YourUsername',
        
        # Notification Settings
        'EMAIL_NOTIFICATIONS': False,
        'TELEGRAM_NOTIFICATIONS': False,
        
        # Email Settings (if enabled)
        'EMAIL_FROM': 'your_email@example.com',
        'EMAIL_TO': 'your_email@example.com',
        'EMAIL_PASSWORD': 'your_app_password',
        'SMTP_SERVER': 'smtp.gmail.com',
        'SMTP_PORT': 587,
        
        # Telegram Settings (if enabled)
        'TELEGRAM_BOT_TOKEN': 'YOUR_BOT_TOKEN',
        'TELEGRAM_CHAT_ID': 'YOUR_CHAT_ID'
    }
'''
    
    config_path = os.path.join(config_dir, "credentials.py")
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"✅ Created config template at {config_path}")
        print("⚠️  IMPORTANT: Edit config/credentials.py with your actual API keys!")
    else:
        print("✅ Config file already exists")

def create_directories():
    """Create necessary directories"""
    dirs = ["logs", "database_backups", "config"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("✅ Created necessary directories")

def add_database_method():
    """Add missing test_connection method to database manager"""
    db_file = "database/db_manager.py"
    
    if not os.path.exists(db_file):
        print("❌ database/db_manager.py not found")
        return False
    
    # Check if method already exists
    with open(db_file, 'r') as f:
        content = f.read()
    
    if "def test_connection(self):" in content:
        print("✅ Database test_connection method already exists")
        return True
    
    # Add the method
    method_code = '''
    def test_connection(self):
        """Test database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            return result is not None
        except Exception as e:
            logging.error(f"Database connection test failed: {e}")
            return False
'''
    
    # Insert before the last class (find a good insertion point)
    lines = content.split('\n')
    
    # Find the end of the ProductionDatabaseManager class
    insert_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("def get_recent_trades(self"):
            insert_index = i + 10  # Add after this method
            break
    
    if insert_index > 0:
        lines.insert(insert_index, method_code)
        
        with open(db_file, 'w') as f:
            f.write('\n'.join(lines))
        print("✅ Added test_connection method to database manager")
        return True
    else:
        print("⚠️  Could not automatically add test_connection method")
        print("Please add it manually to the ProductionDatabaseManager class")
        return False

def test_imports():
    """Test if all imports work"""
    print("🔍 Testing imports...")
    try:
        # Test basic imports
        import sqlite3
        import pandas as pd
        import logging
        import schedule
        print("✅ Basic imports successful")
        
        # Test if vaderSentiment is available
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            print("✅ Sentiment analysis imports successful")
        except ImportError:
            print("⚠️  vaderSentiment not found - install with: pip install vaderSentiment")
        
        # Test if PRAW is available
        try:
            import praw
            print("✅ Reddit API imports successful")
        except ImportError:
            print("⚠️  praw not found - install with: pip install praw")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def run_basic_test():
    """Run a basic test of the trading bot"""
    print("🧪 Running basic test...")
    try:
        # Import the main module
        from main import ProductionTradingBot
        
        # Try to create an instance
        bot = ProductionTradingBot()
        print("✅ Trading bot initialized successfully!")
        
        # Test database
        if bot.db_manager:
            print("✅ Database manager initialized")
        
        # Test sentiment analyzer
        if bot.sentiment_analyzer:
            print("✅ Sentiment analyzer initialized")
        else:
            print("⚠️  Sentiment analyzer not initialized (API keys needed)")
        
        # Test trading engine
        if bot.trading_engine:
            print("✅ Trading engine initialized")
        else:
            print("⚠️  Trading engine not initialized (API keys needed)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("Trading Bot Quick Setup")
    print("=" * 50)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return
    
    # Step 3: Create config template
    create_config_template()
    
    # Step 4: Add missing database method
    add_database_method()
    
    # Step 5: Test imports
    if not test_imports():
        print("❌ Some imports failed - please install missing packages")
        return
    
    # Step 6: Run basic test
    if run_basic_test():
        print("\nSETUP COMPLETE!")
        print("\nNext steps:")
        print("1. Edit config/credentials.py with your API keys")
        print("2. Run the bot with: python main.py")
        print("3. Access web dashboard at: http://localhost:5000 (if enabled)")
    else:
        print("\n⚠️  Setup completed but bot test failed")
        print("This is normal if you haven't configured API keys yet")
        print("\nNext steps:")
        print("1. Edit config/credentials.py with your API keys")
        print("2. Run the bot with: python main.py")

if __name__ == "__main__":
    main()