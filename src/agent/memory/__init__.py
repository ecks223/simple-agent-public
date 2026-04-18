from agent.memory.base import BaselineStrategy, MemoryStrategy
from agent.memory.per_user import PerUserStrategy, UserFact
from agent.memory.per_user_plus_patterns import (
    PerUserPlusPatternsStrategy,
    TeamPattern,
)

STRATEGIES = {
    BaselineStrategy.name: BaselineStrategy,
    PerUserStrategy.name: PerUserStrategy,
    PerUserPlusPatternsStrategy.name: PerUserPlusPatternsStrategy,
}

__all__ = [
    "BaselineStrategy",
    "MemoryStrategy",
    "PerUserStrategy",
    "PerUserPlusPatternsStrategy",
    "STRATEGIES",
    "TeamPattern",
    "UserFact",
]
