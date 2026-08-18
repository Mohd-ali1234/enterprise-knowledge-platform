"""Tests for web page fetching.

No test reaches the network: the HTTP client and DNS resolution are both
substituted.
"""

from __future__ import annotations

import ipaddress
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from app.core.exceptions import DocumentProcessingError
from app.ingestion.web import WebPageFetcher

PAGE = b"<html><head><title>Security Standard</title></head><body><h1>Hi</h1></body></html>"


def fetcher(
    content: bytes = PAGE,
    content_type: str = "text/html; charset=utf-8",
    resolves_to: str = "93.184.216.34",
    **kwargs,
) -> WebPageFetcher:
    """A fetcher whose transport and DNS are both stubbed."""
    response = Mock()
    response.content = content
    response.headers = {"content-type": content_type}
    response.url = "https://example.test/page"
    response.raise_for_status = Mock()

    transport = Mock()
    transport.get = AsyncMock(return_value=response)

    instance = WebPageFetcher(client=transport, **kwargs)
    instance._resolve = staticmethod(lambda host: [ipaddress.ip_address(resolves_to)])
    return instance


class TestUrlValidation:
    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x.test/a", "javascript:alert(1)"])
    async def test_rejects_non_http_schemes(self, url):
        with pytest.raises(DocumentProcessingError, match="http and https"):
            await fetcher().fetch(url)

    @pytest.mark.parametrize(
        "address",
        ["127.0.0.1", "::1", "10.0.0.5", "192.168.1.1", "169.254.169.254"],
    )
    async def test_refuses_private_and_loopback_addresses(self, address):
        """The server must not be usable as a proxy into its own network."""
        with pytest.raises(DocumentProcessingError, match="non-public address"):
            await fetcher(resolves_to=address).fetch("http://internal.test/")

    async def test_private_addresses_are_allowed_when_opted_in(self):
        page = await fetcher(resolves_to="127.0.0.1", allow_private_hosts=True).fetch(
            "http://localhost/page"
        )

        assert page.content == PAGE

    async def test_rejects_a_url_with_no_host(self):
        with pytest.raises(DocumentProcessingError, match="no host"):
            await fetcher().fetch("http:///nowhere")


class TestFetch:
    async def test_returns_the_page_content(self):
        page = await fetcher().fetch("https://example.test/page")

        assert page.content == PAGE

    async def test_uses_the_html_title_for_the_document_name(self):
        page = await fetcher().fetch("https://example.test/page")

        assert page.title == "Security Standard"
        assert page.filename == "Security_Standard.html"

    async def test_falls_back_to_the_url_path_without_a_title(self):
        page = await fetcher(content=b"<html><body>no title</body></html>").fetch(
            "https://example.test/handbook"
        )

        assert page.title == "handbook"  # the requested URL's last path segment

    async def test_rejects_non_page_content_types(self):
        with pytest.raises(DocumentProcessingError, match="not a web page"):
            await fetcher(content_type="application/zip").fetch("https://example.test/a")

    async def test_rejects_a_page_over_the_size_limit(self):
        with pytest.raises(DocumentProcessingError, match="larger than"):
            await fetcher(content=b"x" * 100, max_bytes=10).fetch("https://example.test/a")

    async def test_reports_http_errors_with_their_status(self):
        instance = fetcher()
        request = httpx.Request("GET", "https://example.test/a")
        response = httpx.Response(404, request=request)
        instance._client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("nope", request=request, response=response)
        )

        with pytest.raises(DocumentProcessingError, match="HTTP 404"):
            await instance.fetch("https://example.test/a")

    async def test_reports_transport_failures(self):
        instance = fetcher()
        instance._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(DocumentProcessingError, match="Could not fetch"):
            await instance.fetch("https://example.test/a")

    async def test_filename_is_always_safe(self):
        page = await fetcher(
            content=b"<html><title>../../etc/passwd</title></html>"
        ).fetch("https://example.test/a")

        assert "/" not in page.filename
        assert ".." not in page.filename
