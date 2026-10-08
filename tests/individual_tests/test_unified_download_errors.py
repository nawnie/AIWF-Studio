"""Unknown model-download IDs: the bridge says download_not_found, not a ReTrain failure.

An agent that polls or cancels a download ID ReTrain does not know made a request
mistake; reporting it as upstream_error made it look like ReTrain was broken.
"""

from __future__ import annotations

import pytest

from aiwf.services.unified_bridge import BridgeError
from test_unified_bridge import env  # noqa: F401  (env is a pytest fixture)


@pytest.mark.parametrize("call", ["download_status", "cancel_download"])
def test_unknown_download_id_is_download_not_found(env, call) -> None:
    with pytest.raises(BridgeError) as raised:
        getattr(env.bridge, call)("000000000000")
    assert raised.value.status_code == 404 and raised.value.code == "download_not_found"


def test_known_download_id_still_reports_its_state(env) -> None:
    assert env.bridge.download_status("0123456789ab")["download"]["status"] == "completed"
    assert env.bridge.cancel_download("0123456789ab")["download"]["status"] == "cancelled"
