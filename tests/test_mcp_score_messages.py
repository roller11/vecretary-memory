# SPDX-License-Identifier: MPL-2.0
from unittest.mock import Mock

import pytest

from instinct import mcp_server


@pytest.mark.parametrize('importance', [93, 94, 95, 100])
def test_save_owner_score_message_matches_guard(importance):
    message = mcp_server._check_write_guards('Synthetic durable note.', 'Synthetic note', importance)
    assert 'capped at score 92' in message
    assert 'Scores 93-100 are owner-only' in message


def test_save_allows_agent_score_ceiling():
    assert mcp_server._check_write_guards(
        'Synthetic durable note with sufficient content for the existing high-score guard.',
        'Synthetic note', 92,
    ) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('importance', [93, 94, 95, 100])
async def test_update_owner_score_message_rejects_before_store_access(importance):
    store = Mock()
    result = await mcp_server._handle_update_score(store, {'segment_id': 'synthetic', 'new_score': importance})
    assert 'capped at score 92' in result[0].text
    assert 'Scores 93-100 are owner-only' in result[0].text
    assert store.mock_calls == []
