import hashlib
from typing import Optional
from .utils import get_hostname

class WebMetadata:
    def __init__(
        self,
        url: str,
    ):
        self.url = url
        self.host = get_hostname(url)
        self._metadata_hash = hashlib.md5(str(self.url).encode('utf-8')).hexdigest()

    def get_metadata_hash(self) -> str:
        return self._metadata_hash