# Windows Service Installation Script for Production Trading Bot

import sys
import os
import win32serviceutil
import win32service
import win32event
import servicemanager
import logging
import time
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from main import ProductionTradingBot

class TradingBotService(win32serviceutil.ServiceFramework):
    """Windows service wrapper for the Production Trading Bot"""
    
    _svc_name_ = "ProductionTradingBot"
    _svc_display_name_ = "Production Trading Bot Service"
    _svc_description_ = "Automated stock trading bot with sentiment analysis and risk management"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.running = True
        self.bot = None
        
        # Setup service logging
        self.setup_service_logging()
        
    def setup_service_logging(self):
        """Setup logging for the Windows service"""
        log_dir = Path(current_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "service.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def SvcStop(self):
        """Stop the service"""
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        self.running = False
        
        if self.bot:
            self.logger.info("Shutting down trading bot...")
            self.bot.shutdown()
        
        self.logger.info("Trading bot service stopped")
    
    def SvcDoRun(self):
        """Run the service"""
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        
        self.logger.info("Starting Production Trading Bot Service...")
        
        try:
            # Initialize and start the trading bot
            self.bot = ProductionTradingBot()
            
            # Start the bot in a separate thread to avoid blocking
            import threading
            bot_thread = threading.Thread(target=self.bot.start_production_bot)
            bot_thread.daemon = True
            bot_thread.start()
            
            # Wait for stop signal
            while self.running:
                rc = win32event.WaitForSingleObject(self.hWaitStop, 5000)
                if rc == win32event.WAIT_OBJECT_0:
                    break
                    
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            servicemanager.LogErrorMsg(f"Trading Bot Service Error: {e}")

def install_service():
    """Install the trading bot as a Windows service"""
    try:
        win32serviceutil.InstallService(
            TradingBotService,
            TradingBotService._svc_name_,
            TradingBotService._svc_display_name_,
            description=TradingBotService._svc_description_
        )
        print("✅ Trading Bot service installed successfully!")
        print("Use 'sc start ProductionTradingBot' to start the service")
        
    except Exception as e:
        print(f"❌ Error installing service: {e}")

def uninstall_service():
    """Uninstall the trading bot service"""
    try:
        win32serviceutil.RemoveService(TradingBotService._svc_name_)
        print("✅ Trading Bot service uninstalled successfully!")
        
    except Exception as e:
        print(f"❌ Error uninstalling service: {e}")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(TradingBotService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        if sys.argv[1] == 'install':
            install_service()
        elif sys.argv[1] == 'remove':
            uninstall_service()
        else:
            win32serviceutil.HandleCommandLine(TradingBotService)
