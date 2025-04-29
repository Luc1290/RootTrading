import pytest
from portfolio.src.pockets import PocketManager

def test_pocket_manager_init():
    manager = PocketManager()
    assert manager.db is not None
    assert hasattr(manager, 'pocket_config')
