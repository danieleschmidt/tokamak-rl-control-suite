"""
Notification services for tokamak RL control suite.

This module provides notification capabilities for training completion,
alerts, and system status updates via multiple channels.
"""

import os
import json
import smtplib
from typing import Dict, Any, Optional, List
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime
import warnings
from abc import ABC, abstractmethod


class NotificationChannel(ABC):
    """Base class for notification channels."""
    
    @abstractmethod
    def send(self, message: str, subject: str = "", **kwargs) -> bool:
        """Send notification message."""
        pass


class EmailNotification(NotificationChannel):
    """Email notification channel."""
    
    def __init__(self, smtp_server: str, smtp_port: int = 587,
                 username: str = "", password: str = "",
                 use_tls: bool = True):
        """
        Initialize email notification.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            use_tls: Whether to use TLS encryption
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        
    def send(self, message: str, subject: str = "Tokamak RL Notification",
             to_addresses: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Send email notification.
        
        Args:
            message: Email message content
            subject: Email subject
            to_addresses: List of recipient email addresses
            
        Returns:
            True if successful, False otherwise
        """
        if not to_addresses:
            warnings.warn("No recipient addresses provided for email notification")
            return False
            
        try:
            # Create message
            msg = MimeMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(to_addresses)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MimeText(message, 'plain'))
            
            # Connect to server and send
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            if self.use_tls:
                server.starttls()
                
            if self.username and self.password:
                server.login(self.username, self.password)
                
            server.sendmail(self.username, to_addresses, msg.as_string())
            server.quit()
            
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to send email notification: {e}")
            return False


class SlackNotification(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, webhook_url: str):
        """
        Initialize Slack notification.
        
        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url
        
    def send(self, message: str, subject: str = "", 
             channel: Optional[str] = None, **kwargs) -> bool:
        """
        Send Slack notification.
        
        Args:
            message: Slack message content
            subject: Message title (used as Slack attachment title)
            channel: Slack channel to send to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import requests
            
            # Format message for Slack
            payload = {
                'text': subject if subject else 'Tokamak RL Notification',
                'attachments': [
                    {
                        'color': 'good',
                        'text': message,
                        'footer': 'Tokamak RL Control Suite',
                        'ts': int(datetime.now().timestamp())
                    }
                ]
            }
            
            if channel:
                payload['channel'] = channel
                
            response = requests.post(self.webhook_url, json=payload)
            
            return response.status_code == 200
            
        except ImportError:
            warnings.warn("requests library required for Slack notifications")
            return False
        except Exception as e:
            warnings.warn(f"Failed to send Slack notification: {e}")
            return False


class DiscordNotification(NotificationChannel):
    """Discord notification channel."""
    
    def __init__(self, webhook_url: str):
        """
        Initialize Discord notification.
        
        Args:
            webhook_url: Discord webhook URL
        """
        self.webhook_url = webhook_url
        
    def send(self, message: str, subject: str = "", **kwargs) -> bool:
        """
        Send Discord notification.
        
        Args:
            message: Discord message content
            subject: Message title
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import requests
            
            # Format message for Discord
            embed = {
                'title': subject or 'Tokamak RL Notification',
                'description': message,
                'color': 0x00ff00,  # Green color
                'footer': {
                    'text': 'Tokamak RL Control Suite'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            payload = {
                'embeds': [embed]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            
            return response.status_code == 204  # Discord returns 204 for success
            
        except ImportError:
            warnings.warn("requests library required for Discord notifications")
            return False
        except Exception as e:
            warnings.warn(f"Failed to send Discord notification: {e}")
            return False


class ConsoleNotification(NotificationChannel):
    """Console notification channel for development."""
    
    def send(self, message: str, subject: str = "", **kwargs) -> bool:
        """
        Send console notification.
        
        Args:
            message: Console message content
            subject: Message title
            
        Returns:
            Always True
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n{'='*60}")
        print(f"NOTIFICATION [{timestamp}]")
        if subject:
            print(f"Subject: {subject}")
        print(f"{'='*60}")
        print(message)
        print(f"{'='*60}\n")
        
        return True


class NotificationService:
    """Central notification service supporting multiple channels."""
    
    def __init__(self):
        """Initialize notification service."""
        self.channels: Dict[str, NotificationChannel] = {}
        self.default_channels: List[str] = []
        
        # Add console channel by default
        self.add_channel('console', ConsoleNotification())
        self.default_channels.append('console')
        
    def add_channel(self, name: str, channel: NotificationChannel) -> None:
        """
        Add notification channel.
        
        Args:
            name: Channel name
            channel: Notification channel instance
        """
        self.channels[name] = channel
        
    def set_default_channels(self, channel_names: List[str]) -> None:
        """
        Set default channels for notifications.
        
        Args:
            channel_names: List of channel names to use by default
        """
        # Validate channels exist
        for name in channel_names:
            if name not in self.channels:
                raise ValueError(f"Channel '{name}' not found")
                
        self.default_channels = channel_names
        
    def send(self, message: str, subject: str = "",
             channels: Optional[List[str]] = None, **kwargs) -> Dict[str, bool]:
        """
        Send notification to specified channels.
        
        Args:
            message: Notification message
            subject: Notification subject
            channels: List of channel names (uses default if None)
            **kwargs: Additional channel-specific parameters
            
        Returns:
            Dictionary of channel names and success status
        """
        if channels is None:
            channels = self.default_channels
            
        results = {}
        
        for channel_name in channels:
            if channel_name in self.channels:
                try:
                    success = self.channels[channel_name].send(
                        message, subject, **kwargs
                    )
                    results[channel_name] = success
                except Exception as e:
                    warnings.warn(f"Error sending to channel {channel_name}: {e}")
                    results[channel_name] = False
            else:
                warnings.warn(f"Channel '{channel_name}' not found")
                results[channel_name] = False
                
        return results
        
    def send_training_complete(self, experiment_name: str, 
                             metrics: Dict[str, Any],
                             duration: float,
                             channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Send training completion notification.
        
        Args:
            experiment_name: Name of completed experiment
            metrics: Training metrics
            duration: Training duration in seconds
            channels: Channels to send to
            
        Returns:
            Dictionary of channel names and success status
        """
        duration_str = self._format_duration(duration)
        
        subject = f"Training Complete: {experiment_name}"
        
        message = f"""
Training experiment '{experiment_name}' has completed successfully!

Duration: {duration_str}

Final Metrics:
"""
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                message += f"  {metric}: {value:.4f}\n"
            else:
                message += f"  {metric}: {value}\n"
                
        message += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.send(message, subject, channels)
        
    def send_alert(self, alert_type: str, message: str,
                  severity: str = "warning",
                  channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Send system alert notification.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (info, warning, error, critical)
            channels: Channels to send to
            
        Returns:
            Dictionary of channel names and success status
        """
        severity_emoji = {
            'info': 'ℹ️',
            'warning': '⚠️', 
            'error': '❌',
            'critical': '🚨'
        }
        
        emoji = severity_emoji.get(severity, '❓')
        subject = f"{emoji} Tokamak RL Alert: {alert_type}"
        
        alert_message = f"""
Alert Type: {alert_type}
Severity: {severity.upper()}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Message:
{message}
"""
        
        return self.send(alert_message, subject, channels)
        
    def send_model_upload(self, model_name: str, metrics: Dict[str, Any],
                         upload_location: str,
                         channels: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Send model upload notification.
        
        Args:
            model_name: Name of uploaded model
            metrics: Model performance metrics
            upload_location: Where model was uploaded
            channels: Channels to send to
            
        Returns:
            Dictionary of channel names and success status
        """
        subject = f"Model Uploaded: {model_name}"
        
        message = f"""
A new trained model has been uploaded!

Model: {model_name}
Location: {upload_location}
Upload Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
"""
        
        for metric, value in metrics.items():
            if isinstance(value, float):
                message += f"  {metric}: {value:.4f}\n"
            else:
                message += f"  {metric}: {value}\n"
                
        return self.send(message, subject, channels)
        
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}m {int(secs)}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"


def create_notification_service() -> NotificationService:
    """Create notification service with environment-based configuration."""
    service = NotificationService()
    
    # Configure email if environment variables are set
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_user = os.getenv('SMTP_USERNAME')
    smtp_pass = os.getenv('SMTP_PASSWORD')
    
    if smtp_server and smtp_user:
        email_channel = EmailNotification(
            smtp_server=smtp_server,
            username=smtp_user,
            password=smtp_pass or ""
        )
        service.add_channel('email', email_channel)
        
    # Configure Slack if webhook URL is set
    slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
    if slack_webhook:
        slack_channel = SlackNotification(slack_webhook)
        service.add_channel('slack', slack_channel)
        
    # Configure Discord if webhook URL is set
    discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
    if discord_webhook:
        discord_channel = DiscordNotification(discord_webhook)
        service.add_channel('discord', discord_channel)
        
    return service