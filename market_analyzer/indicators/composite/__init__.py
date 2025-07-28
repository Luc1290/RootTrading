"""
Composite Indicators Module

Ce module contient des indicateurs composites qui combinent plusieurs indicateurs
pour fournir des signaux plus robustes.
"""

from .confluence import calculate_confluence_score, ConfluenceType

__all__ = ['calculate_confluence_score', 'ConfluenceType']