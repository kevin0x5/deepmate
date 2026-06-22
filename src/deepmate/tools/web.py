"""Lightweight network research tools."""

from __future__ import annotations

import html
import http.client
import re
import socket
import ssl
import base64
import codecs
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from deepmate.tools.registry import NativeTool, NativeToolResult
from deepmate.tools.url_safety import public_url_resolution, validate_public_url

USER_AGENT = "Deepmate/0.1 (+https://github.com/kevin0x5/deepmate)"
MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_MAX_CHARS = 30_000
MAX_CHARS = 100_000
DEFAULT_SEARCH_RESULTS = 5
MAX_SEARCH_RESULTS = 10
def web_research_tools(*, network_enabled: bool) -> tuple[NativeTool, ...]:
    """Return lightweight web tools when explicit network access is enabled."""
    if not network_enabled:
        return ()
    return (
        NativeTool(
            name="web_search",
            description="Search the public web and return titles, URLs, and snippets.",
            input_schema=_web_search_schema(),
            handler=_web_search,
        ),
        NativeTool(
            name="web_fetch",
            description="Fetch one public HTTP page and extract readable text.",
            input_schema=_web_fetch_schema(),
            handler=_web_fetch,
        ),
    )


def _web_search(arguments: Mapping[str, object]) -> NativeToolResult:
    original_query = _text(arguments, "query")
    query = _search_query(original_query)
    max_results = _int(
        arguments,
        "max_results",
        DEFAULT_SEARCH_RESULTS,
        1,
        MAX_SEARCH_RESULTS,
    )
    language = _optional_text(arguments, "language", "")
    params = {"q": query}
    if language:
        params["kl"] = language
    backend = "duckduckgo_html"
    url = f"https://html.duckduckgo.com/html/?{urlencode(params)}"
    body, final_url, content_type = _request_public_url(url, max_bytes=MAX_RESPONSE_BYTES)
    decoded = _decode_body(body, content_type)
    results = _parse_duckduckgo_results(decoded)
    parse_warnings: list[str] = []
    if not results:
        results = _parse_duckduckgo_fallback_results(decoded)
        if results:
            parse_warnings.append(
                "DuckDuckGo result markup changed; using fallback link extraction."
            )
    if not results:
        parse_warnings.append(
            "DuckDuckGo returned no safe parseable results; trying Bing HTML fallback."
        )
        bing_url = f"https://www.bing.com/search?{urlencode({'q': query})}"
        body, final_url, content_type = _request_public_url(
            bing_url,
            max_bytes=MAX_RESPONSE_BYTES,
        )
        decoded = _decode_body(body, content_type)
        results = _parse_bing_results(decoded)
        backend = "bing_html"
        if not results:
            parse_warnings.append(
                "Bing returned no safe parseable results; the HTML format may have changed."
            )
    raw_result_count = len(results)
    results, unsafe_count = _safe_search_results(results[: max_results * 3], max_results)
    if unsafe_count:
        parse_warnings.append(
            f"Filtered {unsafe_count} unsafe search result URL(s)."
        )
    content = "\n\n".join(
        f"{index}. {item['title']}\n{item['url']}\n{item['snippet']}".rstrip()
        for index, item in enumerate(results, start=1)
    )
    if not results and raw_result_count == 0 and len(body) > 500:
        parse_warnings.append(
            "No safe parseable results were found."
        )
    parse_warning = " ".join(parse_warnings)
    if parse_warning and content:
        content = f"Warning: {parse_warning}\n\n{content}"
    return NativeToolResult(
        content=content or parse_warning or "No search results found.",
        data={
            "query": query,
            "original_query": original_query,
            "result_count": len(results),
            "backend": backend,
            "search_url": final_url,
            "parse_warning": parse_warning,
        },
        refs=tuple(str(item["url"]) for item in results),
    )


def _web_fetch(arguments: Mapping[str, object]) -> NativeToolResult:
    url = _text(arguments, "url")
    max_chars = _int(arguments, "max_chars", DEFAULT_MAX_CHARS, 1, MAX_CHARS)
    body, final_url, content_type = _request_public_url(url, max_bytes=MAX_RESPONSE_BYTES)
    decoded = _decode_body(body, content_type)
    if _looks_like_html(content_type, body, decoded):
        parser = _ReadableHtmlParser()
        parser.feed(decoded)
        parser.close()
        title = parser.title
        text = parser.render()
    else:
        title = ""
        text = decoded.strip()
    truncated = len(text) > max_chars
    content = text[:max_chars]
    returned_chars = len(content)
    if truncated:
        content = (
            content.rstrip()
            + f"\n\n[truncated - total={len(text)} chars, returned={returned_chars} chars]"
        )
    return NativeToolResult(
        content=content or "(empty response)",
        data={
            "url": final_url,
            "title": title,
            "content_type": content_type,
            "chars": returned_chars,
            "total_chars": len(text),
            "truncated": truncated,
        },
        refs=(final_url,),
    )


def _request_public_url(url: str, *, max_bytes: int) -> tuple[bytes, str, str]:
    validate_public_url(url)
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.5",
        },
    )
    with _open_public_request(request) as response:
        final_url = response.geturl()
        validate_public_url(final_url)
        content_type = response.headers.get("Content-Type", "")
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise ValueError(f"response exceeds {max_bytes} bytes")
    return body, final_url, content_type


def _open_public_request(request: Request):
    return build_opener(
        ProxyHandler({}),
        _PublicHTTPHandler(),
        _PublicHTTPSHandler(),
        _PublicRedirectHandler(),
    ).open(request, timeout=20)


class _PublicHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that connects to a prevalidated public IP."""

    _scheme = "http"

    def connect(self) -> None:
        resolution = public_url_resolution(_connection_url(self._scheme, self.host, self.port))
        self.sock = socket.create_connection(
            (resolution.first_address(), self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self._tunnel()


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that connects to a prevalidated public IP with original SNI."""

    _scheme = "https"

    def __init__(self, *args, context=None, check_hostname=None, **kwargs) -> None:
        if context is None:
            context = ssl._create_default_https_context()
        super().__init__(*args, context=context, check_hostname=check_hostname, **kwargs)

    def connect(self) -> None:
        resolution = public_url_resolution(_connection_url(self._scheme, self.host, self.port))
        sock = socket.create_connection(
            (resolution.first_address(), self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self.sock = sock
            self._tunnel()
            server_hostname = self._tunnel_host
        else:
            server_hostname = self.host
        self.sock = self._context.wrap_socket(
            sock,
            server_hostname=server_hostname if ssl.HAS_SNI else None,
        )


class _PublicHTTPHandler(HTTPHandler):
    def http_open(self, req):
        return self.do_open(_PublicHTTPConnection, req)


class _PublicHTTPSHandler(HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_PublicHTTPSConnection, req)


class _PublicRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        validate_public_url(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _validate_public_url(url: str) -> None:
    validate_public_url(url)


def _charset(content_type: str) -> str:
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset":
            return value.strip("\"'")
    return ""


def _decode_body(body: bytes, content_type: str) -> str:
    charset = _charset(content_type) or _html_meta_charset(body) or "utf-8"
    try:
        codecs.lookup(charset)
    except LookupError:
        charset = "utf-8"
    return body.decode(charset, errors="replace")


def _looks_like_html(content_type: str, body: bytes, decoded: str) -> bool:
    if "html" in content_type.lower():
        return True
    head = body[:16384].decode("ascii", errors="ignore").lower()
    compact_head = decoded[:16384].lower()
    markers = ("<!doctype html", "<html", "<head", "<body", "<title", "<meta")
    return any(marker in head or marker in compact_head for marker in markers)


def _html_meta_charset(body: bytes) -> str:
    head = body[:8192].decode("ascii", errors="ignore")
    for pattern in (
        r"<meta[^>]+charset=[\"']?\s*([A-Za-z0-9._:-]+)",
        r"<meta[^>]+content=[\"'][^\"']*charset=\s*([A-Za-z0-9._:-]+)",
    ):
        match = re.search(pattern, head, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("\"'")
    return ""


def _connection_url(scheme: str, host: str, port: int) -> str:
    hostname = host.strip("[]")
    display_host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    return f"{scheme}://{display_host}:{port}/"


class _ReadableHtmlParser(HTMLParser):
    BLOCK_TAGS = frozenset(
        {"p", "div", "section", "article", "main", "li", "blockquote", "pre", "br"}
    )
    SKIP_TAGS = frozenset({"script", "style", "svg", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._heading_level = 0

    @property
    def title(self) -> str:
        return " ".join(" ".join(self._title_parts).split())

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._heading_level = int(tag[1])
            self._parts.append("\n" + ("#" * self._heading_level) + " ")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in self.BLOCK_TAGS or tag.startswith("h"):
            self._parts.append("\n")
        if tag.startswith("h"):
            self._heading_level = 0

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        clean = " ".join(data.split())
        if not clean:
            return
        if self._in_title:
            self._title_parts.append(clean)
            return
        self._parts.append(clean + " ")

    def render(self) -> str:
        lines = (" ".join(line.split()) for line in "".join(self._parts).splitlines())
        return "\n".join(line for line in lines if line).strip()


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            href = _duckduckgo_result_url(attributes.get("href", ""))
            self._current = {"title": "", "url": href, "snippet": ""}
            self._capture = "title"
            self._parts = []
        elif self._current is not None and "result__snippet" in classes:
            self._capture = "snippet"
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._current is None or not self._capture:
            return
        if tag not in {"a", "div"}:
            return
        self._current[self._capture] = html.unescape(" ".join(self._parts).strip())
        if self._capture == "snippet" or (
            self._capture == "title" and self._current["url"]
        ):
            if self._capture == "snippet":
                self.results.append(self._current)
                self._current = None
        self._capture = ""
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)

    def close(self) -> None:
        super().close()
        if self._current and self._current["title"] and self._current["url"]:
            self.results.append(self._current)
            self._current = None


def _duckduckgo_result_url(value: str) -> str:
    absolute = urljoin("https://duckduckgo.com", value)
    parsed = urlparse(absolute)
    query = parse_qs(parsed.query)
    target = query.get("uddg", [""])[0]
    return target or absolute


def _parse_duckduckgo_results(decoded: str) -> list[dict[str, str]]:
    parser = _DuckDuckGoParser()
    parser.feed(decoded)
    parser.close()
    return parser.results


def _parse_duckduckgo_fallback_results(decoded: str) -> list[dict[str, str]]:
    fallback = _DuckDuckGoLinkFallbackParser()
    fallback.feed(decoded)
    fallback.close()
    return fallback.results


def _parse_bing_results(decoded: str) -> list[dict[str, str]]:
    parser = _BingParser()
    parser.feed(decoded)
    parser.close()
    return parser.results


class _BingParser(HTMLParser):
    """Extract organic results from Bing's ordinary HTML results page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._li_depth = 0
        self._h2_depth = 0
        self._current: dict[str, str] | None = None
        self._capture = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if tag == "li" and "b_algo" in classes:
            self._li_depth = 1
            self._h2_depth = 0
            self._current = {"title": "", "url": "", "snippet": ""}
            self._capture = ""
            self._parts = []
            return
        if self._current is None:
            return
        if tag == "li":
            self._li_depth += 1
        if tag == "h2":
            self._h2_depth += 1
        if tag == "a" and self._h2_depth > 0 and not self._current["url"]:
            href = _bing_result_url(attributes.get("href", ""))
            if href:
                self._current["url"] = href
                self._capture = "title"
                self._parts = []
        elif tag == "p" and self._current["url"] and not self._current["snippet"]:
            self._capture = "snippet"
            self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if self._capture and tag in {"a", "p"}:
            self._current[self._capture] = html.unescape(
                " ".join(" ".join(self._parts).split())
            )
            self._capture = ""
            self._parts = []
        if tag == "li":
            self._li_depth = max(0, self._li_depth - 1)
            if self._li_depth == 0:
                if self._current["title"] and self._current["url"]:
                    self.results.append(self._current)
                self._current = None
                self._capture = ""
                self._parts = []
                self._h2_depth = 0
        elif tag == "h2":
            self._h2_depth = max(0, self._h2_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)


def _bing_result_url(value: str) -> str:
    url = html.unescape(value).strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.hostname == "www.bing.com" and parsed.path.startswith("/ck/"):
        target = parse_qs(parsed.query).get("u", [""])[0]
        decoded = _decode_bing_target(target)
        if decoded:
            return decoded
    return url


def _decode_bing_target(value: str) -> str:
    clean = value.strip()
    if clean.startswith("a1"):
        clean = clean[2:]
    if not clean:
        return ""
    padding = "=" * (-len(clean) % 4)
    try:
        decoded = base64.urlsafe_b64decode((clean + padding).encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return ""
    return decoded.decode("utf-8", errors="replace").strip()


def _safe_search_results(
    results: Sequence[Mapping[str, str]],
    max_results: int,
) -> tuple[list[dict[str, str]], int]:
    safe: list[dict[str, str]] = []
    filtered = 0
    seen: set[str] = set()
    for item in results:
        url = str(item.get("url", "")).strip()
        if url in seen:
            continue
        try:
            validate_public_url(url)
        except ValueError:
            filtered += 1
            continue
        seen.add(url)
        safe.append(
            {
                "title": str(item.get("title", "")).strip() or url,
                "url": url,
                "snippet": str(item.get("snippet", "")).strip(),
            }
        )
        if len(safe) >= max_results:
            break
    return safe, filtered


class _DuckDuckGoLinkFallbackParser(HTMLParser):
    """Extract result-like external links when DuckDuckGo class names change."""

    SKIP_HOSTS = {
        "duckduckgo.com",
        "html.duckduckgo.com",
        "duck.com",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current_url = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a" or self._current_url:
            return
        attributes = dict(attrs)
        url = _duckduckgo_result_url(attributes.get("href", ""))
        if not _looks_like_search_result_url(url):
            return
        self._current_url = url
        self._parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._current_url:
            return
        title = html.unescape(" ".join(" ".join(self._parts).split()))
        if title and all(item["url"] != self._current_url for item in self.results):
            self.results.append(
                {
                    "title": title,
                    "url": self._current_url,
                    "snippet": "",
                }
            )
        self._current_url = ""
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._current_url:
            self._parts.append(data)


def _looks_like_search_result_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False
    host = parsed.hostname or ""
    if host in _DuckDuckGoLinkFallbackParser.SKIP_HOSTS:
        return False
    if host.endswith(".duckduckgo.com"):
        return False
    return True


def _text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value.strip()


def _optional_text(arguments: Mapping[str, object], key: str, default: str) -> str:
    value = arguments.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be text")
    return value.strip()


def _int(
    arguments: Mapping[str, object],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}")
    return value


def _search_query(query: str) -> str:
    parsed = urlparse(query)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        host = parsed.netloc.lower().removeprefix("www.")
        return f"site:{host} {host}"
    return query


def _web_search_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_SEARCH_RESULTS,
            },
            "language": {
                "type": "string",
                "description": "Optional DuckDuckGo region such as cn-zh or us-en.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def _web_fetch_schema() -> Mapping[str, object]:
    return {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": MAX_CHARS},
        },
        "required": ["url"],
        "additionalProperties": False,
    }
