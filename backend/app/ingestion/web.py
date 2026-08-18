"""Stage: Web page fetching.

A URL is not a file format, so it does not belong in `ExtractorRegistry`. This
module turns a URL into bytes on disk that the normal pipeline can ingest: the
fetched HTML is handed to `HtmlTextExtractor` like any uploaded `.html` file.

Fetching a client-supplied URL means the server makes a request on the client's
behalf, so `WebPageFetcher` refuses private, loopback and link-local addresses
unless `WEB_FETCH_ALLOW_PRIVATE_HOSTS` is set. That closes the obvious SSRF
path to `localhost` and cloud metadata endpoints.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import httpx

from app.core.config import settings
from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_HTML_TYPES = ("text/html", "application/xhtml+xml", "application/xml", "text/plain")
# Dots are excluded deliberately: a title of "../../etc/passwd" must not leave
# a filename containing "..", even though the storage layer sanitises again.
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]+")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class FetchedPage:
    """A downloaded page, ready to be written to disk and ingested."""

    url: str
    content: bytes
    title: str

    @property
    def filename(self) -> str:
        """A safe `.html` filename derived from the page title or its URL."""
        stem = _UNSAFE_NAME_CHARS.sub("_", self.title).strip("_-")[:80]
        return f"{stem or 'web-page'}.html"


class WebPageFetcher:
    """Downloads a web page over HTTP, with SSRF and size guards."""

    def __init__(
        self,
        timeout: float | None = None,
        max_bytes: int | None = None,
        user_agent: str | None = None,
        allow_private_hosts: bool | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.timeout = timeout if timeout is not None else settings.web_fetch_timeout_seconds
        self.max_bytes = max_bytes if max_bytes is not None else settings.web_fetch_max_bytes
        self.user_agent = user_agent or settings.web_fetch_user_agent
        self.allow_private_hosts = (
            settings.web_fetch_allow_private_hosts
            if allow_private_hosts is None
            else allow_private_hosts
        )
        self._client = client or httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            max_redirects=5,
        )

    async def fetch(self, url: str) -> FetchedPage:
        """Download `url`.

        Raises:
            DocumentProcessingError: if the URL is rejected, unreachable, too
                large, or does not return a text-like document.
        """
        self._validate(url)

        try:
            response = await self._client.get(
                url, headers={"User-Agent": self.user_agent, "Accept": "text/html,*/*"}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise DocumentProcessingError(
                f"'{url}' returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.HTTPError as exc:
            raise DocumentProcessingError(f"Could not fetch '{url}': {exc}") from exc

        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type and not content_type.startswith(_HTML_TYPES):
            raise DocumentProcessingError(
                f"'{url}' returned '{content_type}', which is not a web page. "
                "Download the file and upload it instead."
            )

        content = response.content
        if len(content) > self.max_bytes:
            raise DocumentProcessingError(
                f"'{url}' is larger than the {self.max_bytes} byte limit."
            )

        logger.info("Fetched %s (%d bytes)", url, len(content))
        return FetchedPage(url=str(response.url), content=content, title=self._title(content, url))

    def _validate(self, url: str) -> None:
        parsed = urlparse(url)

        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise DocumentProcessingError(
                f"Only http and https URLs can be ingested, not '{parsed.scheme or url}'."
            )
        if not parsed.hostname:
            raise DocumentProcessingError(f"'{url}' has no host.")
        if self.allow_private_hosts:
            return

        for address in self._resolve(parsed.hostname):
            if not address.is_global or address.is_reserved:
                raise DocumentProcessingError(
                    f"Refusing to fetch '{parsed.hostname}': it resolves to the "
                    f"non-public address {address}."
                )

    @staticmethod
    def _resolve(hostname: str) -> list[ipaddress._BaseAddress]:
        try:
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror as exc:
            raise DocumentProcessingError(f"Could not resolve '{hostname}': {exc}") from exc
        return [ipaddress.ip_address(info[4][0]) for info in infos]

    @staticmethod
    def _title(content: bytes, url: str) -> str:
        """The page's <title>, falling back to the last path segment."""
        match = _TITLE.search(content[:65536].decode("utf-8", errors="replace"))
        if match and (title := re.sub(r"\s+", " ", match.group(1)).strip()):
            return title[:120]

        path = unquote(urlparse(url).path).rstrip("/")
        return path.rsplit("/", 1)[-1] or urlparse(url).hostname or "web-page"

    async def aclose(self) -> None:
        await self._client.aclose()
