"""
Composite Indicators Module

Ce module contient des indicateurs composites qui combinent plusieurs indicateurs
pour fournir des signaux plus robustes.
"""

from .confluence import calculate_confluence_score, ConfluenceType
from .signal_strength import calculate_signal_strength

__all__ = ["calculate_confluence_score", "ConfluenceType", "calculate_signal_strength"]
