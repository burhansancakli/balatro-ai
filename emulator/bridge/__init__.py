"""Balatrobot protocol bridge.

Converts between jackdaw engine Actions and balatrobot JSON-RPC calls,
and between balatrobot game state responses and engine game_state dicts.
"""

from emulator.bridge.backend import Backend, LiveBackend, RPCError, SimBackend
from emulator.bridge.balatrobot_adapter import (
    action_to_rpc,
    bot_state_to_game_state,
    extract_comparison_keys,
    game_state_to_bot_response,
)
from emulator.bridge.deserializer import rpc_to_action
from emulator.bridge.serializer import (
    serialize_area,
    serialize_card,
    serialize_hands,
)

__all__ = [
    # adapter (original)
    "action_to_rpc",
    "bot_state_to_game_state",
    "extract_comparison_keys",
    "game_state_to_bot_response",
    # deserializer
    "rpc_to_action",
    # serializer
    "serialize_area",
    "serialize_card",
    "serialize_hands",
    # backend
    "Backend",
    "LiveBackend",
    "RPCError",
    "SimBackend",
]
