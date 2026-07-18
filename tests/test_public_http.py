from __future__ import annotations

import unittest
from types import SimpleNamespace
from urllib.request import HTTPRedirectHandler, Request
from unittest.mock import patch

from deepmate.public_http import (
    _PublicHTTPConnection,
    _PublicRedirectHandler,
    open_public_request,
)
from deepmate.url_safety import validate_public_url


class PublicHttpTests(unittest.TestCase):
    def test_url_with_embedded_credentials_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedded credentials"):
            validate_public_url("https://user:password@example.com/bundle")

    def test_http_connection_uses_prevalidated_address(self) -> None:
        connection = _PublicHTTPConnection("public.example", 80, timeout=3)
        resolution = SimpleNamespace(first_address=lambda: "203.0.113.10")
        connected_socket = object()

        with (
            patch(
                "deepmate.public_http.public_url_resolution",
                return_value=resolution,
            ) as resolve,
            patch(
                "deepmate.public_http.socket.create_connection",
                return_value=connected_socket,
            ) as create_connection,
        ):
            connection.connect()

        resolve.assert_called_once_with("http://public.example:80/")
        create_connection.assert_called_once_with(
            ("203.0.113.10", 80),
            3,
            None,
        )
        self.assertIs(connection.sock, connected_socket)

    def test_initial_private_url_is_rejected_before_opening(self) -> None:
        request = Request("http://127.0.0.1/private")

        with (
            patch(
                "deepmate.public_http.validate_public_url",
                side_effect=ValueError("private URL"),
            ),
            patch("deepmate.public_http.build_opener") as build_opener,
        ):
            with self.assertRaisesRegex(ValueError, "private URL"):
                open_public_request(request)

        build_opener.assert_not_called()

    def test_private_redirect_is_rejected_before_following(self) -> None:
        request = Request("https://public.example/start")
        handler = _PublicRedirectHandler()

        with (
            patch(
                "deepmate.public_http.validate_public_url",
                side_effect=ValueError("private redirect"),
            ),
            patch.object(HTTPRedirectHandler, "redirect_request") as parent_redirect,
        ):
            with self.assertRaisesRegex(ValueError, "private redirect"):
                handler.redirect_request(
                    request,
                    None,
                    302,
                    "Found",
                    {},
                    "http://127.0.0.1/private",
                )

        parent_redirect.assert_not_called()

    def test_cross_origin_redirect_strips_sensitive_headers(self) -> None:
        request = Request("https://public.example/start")
        redirected = Request(
            "https://other.example/next",
            headers={
                "Authorization": "Bearer secret",
                "Cookie": "session=secret",
                "X-Public": "keep",
            },
        )
        handler = _PublicRedirectHandler()

        with (
            patch("deepmate.public_http.validate_public_url"),
            patch.object(
                HTTPRedirectHandler,
                "redirect_request",
                return_value=redirected,
            ),
        ):
            result = handler.redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                redirected.full_url,
            )

        self.assertIs(result, redirected)
        self.assertIsNone(redirected.get_header("Authorization"))
        self.assertIsNone(redirected.get_header("Cookie"))
        self.assertEqual(redirected.get_header("X-public"), "keep")


if __name__ == "__main__":
    unittest.main()
