from __future__ import annotations

import os

from agastya.api.app import create_app
from agastya.store.sqlite_store import ViolationStore

STORE_PATH_ENV = "AGASTYA_STORE_PATH"
DEFAULT_STORE_PATH = "agastya_violations.db"

store = ViolationStore(os.environ.get(STORE_PATH_ENV, DEFAULT_STORE_PATH))
app = create_app(store)
