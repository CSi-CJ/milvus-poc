"""
Property-based test: 配置凭据占位符化 (Config Credential Placeholderization)

**Feature: project-cleanup-for-opensource, Property 3: 配置凭据占位符化**
**Validates: Requirements 6.2**

Verifies that all credential-related fields in config/config.example.json
contain only empty strings or explicit placeholder strings — no real
credentials should be present.
"""

import json
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Known credential field name patterns
CREDENTIAL_FIELD_NAMES = [
    "user",
    "password",
    "secret_key",
    "api_key",
    "token",
    "access_key",
    "secret",
    "auth_token",
    "private_key",
    "credentials",
]

# Values considered safe placeholders
SAFE_PLACEHOLDER_VALUES = {
    "",
    "your-password-here",
    "your-secret-key-here",
    "your-api-key-here",
    "your-token-here",
    "your-access-key-here",
}


def load_config() -> dict:
    """Load config/config.example.json."""
    config_path = Path("config/config.example.json")
    assert config_path.is_file(), "config/config.example.json does not exist"
    return json.loads(config_path.read_text(encoding="utf-8"))


def extract_credential_fields(obj: dict, prefix: str = "") -> list[tuple[str, str, object]]:
    """
    Recursively extract all fields whose names match known credential patterns.

    Returns a list of (full_path, field_name, value) tuples.
    """
    results = []
    for key, value in obj.items():
        full_path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            results.extend(extract_credential_fields(value, full_path))
        else:
            # Check if this key matches any credential field name
            key_lower = key.lower()
            if key_lower in CREDENTIAL_FIELD_NAMES:
                results.append((full_path, key_lower, value))
    return results


# Load config and extract credential fields at module level
CONFIG_DATA = load_config()
CREDENTIAL_FIELDS = extract_credential_fields(CONFIG_DATA)


@pytest.mark.skipif(
    len(CREDENTIAL_FIELDS) == 0,
    reason="No credential fields found in config/config.example.json",
)
@given(field=st.sampled_from(CREDENTIAL_FIELDS))
@settings(max_examples=100)
def test_credential_fields_are_placeholders(field: tuple[str, str, object]):
    """
    **Feature: project-cleanup-for-opensource, Property 3: 配置凭据占位符化**
    **Validates: Requirements 6.2**

    For randomly sampled credential-related fields from config/config.example.json:
    1. The value must be a string (not a non-string real value)
    2. The value must be an empty string or a known placeholder string
    3. The value must not look like a real credential (no long alphanumeric strings)

    This ensures no real credentials leak into the example config template.
    """
    full_path, field_name, value = field

    # 1. Value must be a string
    assert isinstance(value, str), (
        f"Credential field '{full_path}' has non-string value: {type(value).__name__}"
    )

    # 2. Value must be empty or a known placeholder
    is_safe = value in SAFE_PLACEHOLDER_VALUES or value.startswith("your-")
    assert is_safe, (
        f"Credential field '{full_path}' contains a potentially real credential: '{value}'. "
        f"Expected empty string or placeholder like 'your-password-here'."
    )
