import pytest
from unittest.mock import MagicMock, patch
from analyzer.src.redis_subscriber import RedisSubscriber

def test_publish_signal_formatting():
    subscriber = RedisSubscriber(symbols=["BTCUSDC"])
    subscriber.redis_client = MagicMock()
    signal = MagicMock()
    signal.dict.return_value = {
        "symbol": "BTCUSDC",
        "strategy": "MockStrategy",
        "side": "BUY",
        "timestamp": "2024-01-01T00:00:00",
        "price": 45000
    }
    subscriber.publish_signal(signal)
    subscriber.redis_client.publish.assert_called()
