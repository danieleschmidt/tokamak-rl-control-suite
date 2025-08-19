"""
Internationalization and localization support for tokamak RL control system.

This module provides comprehensive I18n support including message translation,
locale-aware formatting, timezone handling, and accessibility features.
"""

import os
import json
import locale
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    import babel
    from babel import Locale, dates, numbers, units
    from babel.messages import Catalog
    BABEL_AVAILABLE = True
except ImportError:
    BABEL_AVAILABLE = False
    # Using fallback implementations


class SupportedLanguage(Enum):
    """Supported languages for the tokamak control system."""
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    SPANISH = "es"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    RUSSIAN = "ru"


class SupportedRegion(Enum):
    """Supported regions/locales."""
    US = "US"
    GB = "GB"
    FR = "FR"
    DE = "DE"
    IT = "IT"
    ES = "ES"
    JP = "JP"
    KR = "KR"
    CN = "CN"
    TW = "TW"
    RU = "RU"


@dataclass
class LocaleConfig:
    """Configuration for locale-specific settings."""
    language: SupportedLanguage
    region: SupportedRegion
    timezone: str = "UTC"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "standard"
    currency: str = "USD"
    measurement_system: str = "metric"  # metric, imperial, mixed


class MessageCatalog:
    """Message catalog for internationalized strings."""
    
    def __init__(self):
        self.messages: Dict[str, Dict[str, str]] = {}
        self._load_default_messages()
    
    def _load_default_messages(self):
        """Load default message translations."""
        # English (base language)
        self.messages["en"] = {
            # System messages
            "system.startup": "Tokamak RL Control System Starting",
            "system.shutdown": "System Shutdown Initiated",
            "system.emergency": "EMERGENCY: System in Safe Mode",
            "system.ready": "System Ready for Operation",
            
            # Safety messages
            "safety.violation": "Safety Violation Detected",
            "safety.disruption_warning": "Disruption Risk Warning",
            "safety.constraint_applied": "Safety Constraint Applied",
            "safety.emergency_mode": "Emergency Safety Mode Active",
            
            # Control messages
            "control.action_filtered": "Control Action Safety Filtered",
            "control.optimization_active": "Performance Optimization Active",
            "control.learning_progress": "Agent Learning in Progress",
            
            # Error messages
            "error.invalid_input": "Invalid Input Parameters",
            "error.system_fault": "System Fault Detected",
            "error.communication_lost": "Communication Lost",
            "error.sensor_malfunction": "Sensor Malfunction",
            
            # Units and measurements
            "units.tesla": "T",
            "units.ampere": "A",
            "units.meter": "m",
            "units.second": "s",
            "units.kelvin": "K",
            "units.pascal": "Pa",
            "units.watt": "W",
            "units.mega": "M",
            "units.kilo": "k",
            
            # Status indicators
            "status.online": "Online",
            "status.offline": "Offline",
            "status.warning": "Warning",
            "status.error": "Error",
            "status.normal": "Normal",
            "status.critical": "Critical"
        }
        
        # French translations
        self.messages["fr"] = {
            "system.startup": "Démarrage du Système de Contrôle RL Tokamak",
            "system.shutdown": "Arrêt du Système Initié",
            "system.emergency": "URGENCE: Système en Mode Sécurisé",
            "system.ready": "Système Prêt pour Fonctionnement",
            "safety.violation": "Violation de Sécurité Détectée",
            "safety.disruption_warning": "Avertissement Risque de Disruption",
            "safety.constraint_applied": "Contrainte de Sécurité Appliquée",
            "safety.emergency_mode": "Mode de Sécurité d'Urgence Actif",
            "units.tesla": "T",
            "units.ampere": "A",
            "units.meter": "m",
            "status.online": "En Ligne",
            "status.offline": "Hors Ligne",
            "status.warning": "Avertissement",
            "status.error": "Erreur",
            "status.normal": "Normal",
            "status.critical": "Critique"
        }
        
        # German translations
        self.messages["de"] = {
            "system.startup": "Tokamak RL Kontrollsystem Startet",
            "system.shutdown": "Systemabschaltung Eingeleitet",
            "system.emergency": "NOTFALL: System im Sicherheitsmodus",
            "system.ready": "System Bereit für Betrieb",
            "safety.violation": "Sicherheitsverletzung Erkannt",
            "safety.disruption_warning": "Disruptions-Risiko Warnung",
            "safety.constraint_applied": "Sicherheitsbeschränkung Angewendet",
            "safety.emergency_mode": "Notfall-Sicherheitsmodus Aktiv",
            "status.online": "Online",
            "status.offline": "Offline",
            "status.warning": "Warnung",
            "status.error": "Fehler",
            "status.normal": "Normal",
            "status.critical": "Kritisch"
        }
        
        # Japanese translations
        self.messages["ja"] = {
            "system.startup": "トカマクRL制御システム開始",
            "system.shutdown": "システム停止開始",
            "system.emergency": "緊急事態：システムはセーフモード",
            "system.ready": "システム動作準備完了",
            "safety.violation": "安全違反検出",
            "safety.disruption_warning": "ディスラプションリスク警告",
            "safety.constraint_applied": "安全制約適用",
            "safety.emergency_mode": "緊急安全モード有効",
            "status.online": "オンライン",
            "status.offline": "オフライン",
            "status.warning": "警告",
            "status.error": "エラー",
            "status.normal": "正常",
            "status.critical": "重要"
        }
        
        # Chinese Simplified translations
        self.messages["zh-CN"] = {
            "system.startup": "托卡马克强化学习控制系统启动",
            "system.shutdown": "系统关闭启动",
            "system.emergency": "紧急情况：系统处于安全模式",
            "system.ready": "系统准备运行",
            "safety.violation": "检测到安全违规",
            "safety.disruption_warning": "破裂风险警告",
            "safety.constraint_applied": "已应用安全约束",
            "safety.emergency_mode": "紧急安全模式激活",
            "status.online": "在线",
            "status.offline": "离线",
            "status.warning": "警告",
            "status.error": "错误",
            "status.normal": "正常",
            "status.critical": "严重"
        }
    
    def get_message(self, key: str, language: str = "en", **kwargs) -> str:
        """Get localized message."""
        lang_messages = self.messages.get(language, self.messages.get("en", {}))
        message = lang_messages.get(key, key)
        
        # Simple parameter substitution
        if kwargs:
            try:
                message = message.format(**kwargs)
            except (KeyError, ValueError):
                pass  # Return unformatted message if substitution fails
                
        return message
    
    def add_message(self, key: str, language: str, message: str):
        """Add or update a message for a specific language."""
        if language not in self.messages:
            self.messages[language] = {}
        self.messages[language][key] = message
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.messages.keys())


class NumberFormatter:
    """Locale-aware number formatting."""
    
    def __init__(self, locale_config: LocaleConfig):
        self.locale_config = locale_config
        self.locale_str = f"{locale_config.language.value}_{locale_config.region.value}"
    
    def format_number(self, value: Union[int, float], precision: int = 2) -> str:
        """Format number according to locale."""
        if BABEL_AVAILABLE:
            try:
                babel_locale = Locale.parse(self.locale_str)
                return numbers.format_decimal(value, locale=babel_locale)
            except Exception:
                pass
        
        # Fallback formatting
        if self.locale_config.language in [SupportedLanguage.FRENCH, SupportedLanguage.GERMAN]:
            # Use comma as decimal separator, space as thousands separator
            formatted = f"{value:,.{precision}f}".replace(",", " ").replace(".", ",")
            return formatted.replace(" ", ".")
        else:
            # English-style formatting
            return f"{value:,.{precision}f}"
    
    def format_scientific(self, value: float, precision: int = 2) -> str:
        """Format number in scientific notation."""
        return f"{value:.{precision}e}"
    
    def format_percentage(self, value: float, precision: int = 1) -> str:
        """Format percentage."""
        if BABEL_AVAILABLE:
            try:
                babel_locale = Locale.parse(self.locale_str)
                return numbers.format_percent(value, locale=babel_locale)
            except Exception:
                pass
        
        # Fallback
        return f"{value * 100:.{precision}f}%"


class DateTimeFormatter:
    """Locale-aware date and time formatting."""
    
    def __init__(self, locale_config: LocaleConfig):
        self.locale_config = locale_config
        self.locale_str = f"{locale_config.language.value}_{locale_config.region.value}"
    
    def format_datetime(self, dt: datetime, include_timezone: bool = True) -> str:
        """Format datetime according to locale."""
        if BABEL_AVAILABLE:
            try:
                babel_locale = Locale.parse(self.locale_str)
                return dates.format_datetime(dt, locale=babel_locale)
            except Exception:
                pass
        
        # Fallback formatting
        format_str = f"{self.locale_config.date_format} {self.locale_config.time_format}"
        if include_timezone:
            format_str += " %Z"
        
        return dt.strftime(format_str)
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}min"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}h"
        else:
            days = seconds / 86400
            return f"{days:.1f}d"


class UnitFormatter:
    """Physical unit formatting with locale support."""
    
    def __init__(self, locale_config: LocaleConfig):
        self.locale_config = locale_config
        self.measurement_system = locale_config.measurement_system
    
    def format_magnetic_field(self, value: float, unit: str = "T") -> str:
        """Format magnetic field strength."""
        if abs(value) >= 1:
            return f"{value:.2f} {unit}"
        elif abs(value) >= 0.001:
            return f"{value * 1000:.0f} m{unit}"
        else:
            return f"{value * 1000000:.0f} µ{unit}"
    
    def format_current(self, value: float, unit: str = "A") -> str:
        """Format electric current."""
        if abs(value) >= 1000000:
            return f"{value / 1000000:.1f} M{unit}"
        elif abs(value) >= 1000:
            return f"{value / 1000:.1f} k{unit}"
        else:
            return f"{value:.1f} {unit}"
    
    def format_distance(self, value: float, unit: str = "m") -> str:
        """Format distance/length."""
        if self.measurement_system == "imperial" and unit == "m":
            # Convert to feet/inches
            feet = value * 3.28084
            if feet >= 1:
                return f"{feet:.1f} ft"
            else:
                inches = feet * 12
                return f"{inches:.1f} in"
        
        # Metric formatting
        if abs(value) >= 1000:
            return f"{value / 1000:.1f} k{unit}"
        elif abs(value) >= 1:
            return f"{value:.2f} {unit}"
        elif abs(value) >= 0.01:
            return f"{value * 100:.0f} c{unit}"
        else:
            return f"{value * 1000:.1f} m{unit}"
    
    def format_temperature(self, value: float, unit: str = "K") -> str:
        """Format temperature."""
        if self.measurement_system == "imperial" and unit in ["K", "C"]:
            # Convert to Fahrenheit
            if unit == "K":
                celsius = value - 273.15
            else:
                celsius = value
            fahrenheit = celsius * 9/5 + 32
            return f"{fahrenheit:.1f} °F"
        
        if unit == "K":
            return f"{value:.1f} K"
        else:
            return f"{value:.1f} °C"


class AccessibilityFormatter:
    """Accessibility-focused formatting for screen readers and assistive technology."""
    
    def __init__(self, locale_config: LocaleConfig):
        self.locale_config = locale_config
        self.catalog = MessageCatalog()
    
    def format_safety_alert(self, alert_type: str, details: str = "") -> str:
        """Format safety alerts for accessibility."""
        lang = self.locale_config.language.value
        
        base_message = self.catalog.get_message(f"safety.{alert_type}", lang)
        
        # Add accessibility markers
        if alert_type in ["emergency", "critical"]:
            prefix = "URGENT ALERT: " if lang == "en" else "ALERTE URGENTE: "
            return f"{prefix}{base_message}. {details}"
        elif alert_type in ["warning", "disruption_warning"]:
            prefix = "WARNING: " if lang == "en" else "AVERTISSEMENT: "
            return f"{prefix}{base_message}. {details}"
        else:
            return f"{base_message}. {details}"
    
    def format_status_update(self, status: str, component: str = "") -> str:
        """Format status updates for screen readers."""
        lang = self.locale_config.language.value
        status_msg = self.catalog.get_message(f"status.{status}", lang)
        
        if component:
            return f"{component}: {status_msg}"
        else:
            return status_msg
    
    def format_measurement_verbose(self, value: float, unit: str, description: str = "") -> str:
        """Format measurements in verbose, accessible way."""
        formatter = UnitFormatter(self.locale_config)
        number_formatter = NumberFormatter(self.locale_config)
        
        formatted_value = number_formatter.format_number(value)
        
        # Add unit description for accessibility
        unit_descriptions = {
            "T": "tesla",
            "A": "amperes", 
            "m": "meters",
            "K": "kelvin",
            "Pa": "pascals",
            "W": "watts"
        }
        
        unit_desc = unit_descriptions.get(unit, unit)
        result = f"{formatted_value} {unit_desc}"
        
        if description:
            result = f"{description}: {result}"
            
        return result


class LocalizationManager:
    """Central manager for all localization functionality."""
    
    def __init__(self, locale_config: Optional[LocaleConfig] = None):
        if locale_config is None:
            locale_config = LocaleConfig(
                language=SupportedLanguage.ENGLISH,
                region=SupportedRegion.US
            )
        
        self.locale_config = locale_config
        self.catalog = MessageCatalog()
        self.number_formatter = NumberFormatter(locale_config)
        self.datetime_formatter = DateTimeFormatter(locale_config)
        self.unit_formatter = UnitFormatter(locale_config)
        self.accessibility_formatter = AccessibilityFormatter(locale_config)
    
    def set_locale(self, language: SupportedLanguage, region: SupportedRegion):
        """Change current locale."""
        self.locale_config.language = language
        self.locale_config.region = region
        
        # Update formatters
        self.number_formatter = NumberFormatter(self.locale_config)
        self.datetime_formatter = DateTimeFormatter(self.locale_config)
        self.unit_formatter = UnitFormatter(self.locale_config)
        self.accessibility_formatter = AccessibilityFormatter(self.locale_config)
    
    def get_message(self, key: str, **kwargs) -> str:
        """Get localized message."""
        return self.catalog.get_message(key, self.locale_config.language.value, **kwargs)
    
    def format_system_status(self, status: Dict[str, Any]) -> Dict[str, str]:
        """Format system status for display."""
        formatted = {}
        
        for key, value in status.items():
            if isinstance(value, (int, float)):
                if key.endswith('_current'):
                    formatted[key] = self.unit_formatter.format_current(value)
                elif key.endswith('_field'):
                    formatted[key] = self.unit_formatter.format_magnetic_field(value)
                elif key.endswith('_temperature'):
                    formatted[key] = self.unit_formatter.format_temperature(value)
                else:
                    formatted[key] = self.number_formatter.format_number(value)
            elif isinstance(value, str):
                # Try to translate status strings
                translated = self.catalog.get_message(f"status.{value.lower()}", 
                                                    self.locale_config.language.value)
                formatted[key] = translated if translated != f"status.{value.lower()}" else value
            else:
                formatted[key] = str(value)
        
        return formatted
    
    def get_supported_locales(self) -> List[Dict[str, str]]:
        """Get list of supported locales."""
        locales = []
        for lang in SupportedLanguage:
            for region in SupportedRegion:
                locales.append({
                    'language': lang.value,
                    'region': region.value,
                    'display_name': f"{lang.value}_{region.value}"
                })
        return locales


# Global localization manager instance
_global_l10n_manager = None


def get_global_l10n_manager() -> LocalizationManager:
    """Get or create global localization manager."""
    global _global_l10n_manager
    if _global_l10n_manager is None:
        _global_l10n_manager = LocalizationManager()
    return _global_l10n_manager


def set_global_locale(language: SupportedLanguage, region: SupportedRegion):
    """Set global locale for the application."""
    manager = get_global_l10n_manager()
    manager.set_locale(language, region)


# Convenience functions
def _(message_key: str, **kwargs) -> str:
    """Shorthand function for getting localized messages."""
    return get_global_l10n_manager().get_message(message_key, **kwargs)


def format_for_locale(value: Any, format_type: str = "auto") -> str:
    """Format value according to current locale."""
    manager = get_global_l10n_manager()
    
    if format_type == "number" or (format_type == "auto" and isinstance(value, (int, float))):
        return manager.number_formatter.format_number(value)
    elif format_type == "datetime" or (format_type == "auto" and isinstance(value, datetime)):
        return manager.datetime_formatter.format_datetime(value)
    else:
        return str(value)