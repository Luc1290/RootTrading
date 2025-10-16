"""
Composite Indicators Module

Ce module contient des indicateurs composites qui combinent plusieurs indicateurs
pour fournir des signaux plus robustes.
"""

from .confluence import ConfluenceType, calculate_confluence_score
from .signal_strength import calculate_signal_strength

__all__ = [
    "ConfluenceType",
    "calculate_confluence_score",
    "calculate_signal_strength"]
