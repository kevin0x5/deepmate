"""Shared SVG safety checks for built-in artifact tools."""

from __future__ import annotations

import re

UNSAFE_SVG_PATTERN = re.compile(
    r"<\s*(?:\?xml-stylesheet|(?:[\w.-]+:)?(?:script|foreignObject|iframe|object|embed|base|link|meta|use|image|feImage|animate|animateMotion|animateTransform|set)\b)|"
    r"<\s*(?:[\w.-]+:)?a\b[^>]*\btarget\s*=|"
    r"[\s/\"']on[a-zA-Z]+\s*=|"
    r"\b(?:requiredExtensions|externalResourcesRequired)\s*=|"
    r"(?:href|[\w.-]+:href)\s*=\s*['\"]\s*(?!#)|"
    r"@import\b|"
    r"url\s*\(\s*['\"]?(?!#)|"
    r"(?:expression\s*\(|javascript\s*:|data\s*:)",
    re.IGNORECASE,
)
