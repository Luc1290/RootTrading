import pytest
from trader.src.cycle_manager import CycleManager

def test_cycle_manager_initialization():
    manager = CycleManager()
    assert manager.active_cycles is not None
    assert isinstance(manager.demo_mode, bool)
