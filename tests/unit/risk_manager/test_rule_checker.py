import pytest
from risk_manager.src.checker import RuleChecker

def test_rule_checker_init():
    checker = RuleChecker()
    assert isinstance(checker.rules, list)
    assert checker.portfolio_api_url.startswith("http")

def test_rule_evaluation_simple_condition():
    checker = RuleChecker()
    condition = "active_trades > {max_trades}"
    params = {"max_trades": 5}
    state = {"active_trades": 10}
    result = checker._evaluate_condition(condition, params, state)
    assert result is True
