"""OAuth login primitives for xAI's public Grok CLI client.

Mirrors the official Grok CLI flow: authorization-code + PKCE against `auth.x.ai` with the
public client id, a localhost callback (ephemeral ports allowed, unlike the pinned Codex
redirect), and a bearer access token served against the regular REST API. Endpoints are pinned
from the OIDC discovery document (`https://auth.x.ai/.well-known/openid-configuration`,
live-verified 2026-08-27).

Shares its shape with [`OpenAICodexOAuthFlow`][pydantic_ai.providers.openai_codex.OpenAICodexOAuthFlow]
via [`OAuthFlow`][pydantic_ai.providers._oauth.OAuthFlow]: construction does no I/O, and core owns
none of the interactive parts.
"""

from __future__ import annotations as _annotations

import secrets
from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx2
from pydantic import BaseModel, SecretStr

from ._oauth import CredentialsError, OAuthFlow, post_token_request

__all__ = (
    'CredentialsRefreshError',
    'XaiOAuthCredentials',
    'XaiOAuthFlow',
    'refresh_credentials',
)

# Pinned from the OIDC discovery document; the public Grok CLI client (unlike Codex) also
# advertises a device-code grant, which is intentionally out of scope here.
_AUTHORIZE_URL = 'https://auth.x.ai/oauth2/authorize'
_TOKEN_URL = 'https://auth.x.ai/oauth2/token'
# The public OAuth client used by the official Grok CLI.
_PUBLIC_CLIENT_ID = 'b1a00492-073a-47ea-816f-4c329264a828'
_DEFAULT_SCOPE = 'openid profile email offline_access grok-cli:access api:access'
# xAI accepts ephemeral localhost ports, so this is a default, not a pin.
_DEFAULT_REDIRECT_URI = 'http://127.0.0.1:56121/callback'
# Refresh this many seconds before the wall-clock expiry from the token response.
_TOKEN_EXPIRY_BUFFER = timedelta(seconds=120)


class CredentialsRefreshError(CredentialsError):
    """Refreshing xAI credentials against the token endpoint failed.

    When the underlying error is `invalid_grant`, the stored grant is no longer usable and a
    fresh authorization is required (rerun the flow below).
    """

    provider_name = 'xai'


@dataclass
class XaiOAuthCredentials:
    """xAI subscription credentials.

    Secrets use `SecretStr` so tokens never leak through reprs, logs, or accidental serialization.
    """

    _: KW_ONLY
    access_token: SecretStr
    refresh_token: SecretStr
    expires_at: datetime | None
    """Wall-clock expiry derived from the token response's `expires_in`; `None` when absent."""

    def is_stale(self) -> bool:
        """Whether the access token is within the pre-expiry refresh buffer."""
        return self.expires_at is not None and datetime.now(timezone.utc) >= self.expires_at - _TOKEN_EXPIRY_BUFFER


class _TokenResponse(BaseModel):
    """The fields of an OAuth token-endpoint response that credentials are built from."""

    access_token: str | None = None
    refresh_token: str | None = None
    expires_in: float | None = None


def _credentials_from_token_response(
    data: _TokenResponse, fallback_refresh_token: str | None = None
) -> XaiOAuthCredentials:
    """Build credentials from an OAuth token-endpoint response.

    xAI may omit `refresh_token` when rotation is not required, so a refresh keeps the grant it
    already holds via `fallback_refresh_token`.
    """
    if not data.access_token:
        raise CredentialsRefreshError('Token endpoint response is missing `access_token`.')
    refresh_token = data.refresh_token or fallback_refresh_token
    if not refresh_token:
        raise CredentialsRefreshError(
            'Token endpoint response is missing `refresh_token`; request the `offline_access` scope.'
        )
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=data.expires_in) if data.expires_in is not None else None
    )
    return XaiOAuthCredentials(
        access_token=SecretStr(data.access_token), refresh_token=SecretStr(refresh_token), expires_at=expires_at
    )


async def refresh_credentials(
    credentials: XaiOAuthCredentials, *, http_client: httpx2.AsyncClient | None = None
) -> XaiOAuthCredentials:
    """Exchange the refresh token for a new credential set against the public Grok CLI client.

    Raises [`CredentialsRefreshError`][pydantic_ai.providers.xai_oauth.CredentialsRefreshError]
    when the token endpoint rejects the grant (`invalid_grant` means a fresh authorization is
    required) or returns a malformed response.
    """
    data = await post_token_request(
        _TOKEN_URL,
        {
            'grant_type': 'refresh_token',
            'refresh_token': credentials.refresh_token.get_secret_value(),
            'client_id': _PUBLIC_CLIENT_ID,
        },
        response_type=_TokenResponse,
        error=CredentialsRefreshError,
        http_client=http_client,
    )
    return _credentials_from_token_response(data, fallback_refresh_token=credentials.refresh_token.get_secret_value())


class XaiOAuthFlow(OAuthFlow[XaiOAuthCredentials]):
    """Pure authorization-code + PKCE context for the xAI public Grok CLI client.

    Construction does no I/O: build the context anywhere, send the user to
    `authorization_url()`, then call `exchange_code()` from your redirect handler. Unlike the
    Codex client's pinned redirect, xAI accepts any localhost port, so pass `redirect_uri=` to
    use one other than the default.
    """

    def __init__(self, *, redirect_uri: str = _DEFAULT_REDIRECT_URI, state: str | None = None) -> None:
        super().__init__(redirect_uri=redirect_uri, state=state)
        self.nonce = secrets.token_hex(16)

    def authorization_url(self, *, scope: str | None = None, extra_params: Mapping[str, str] | None = None) -> str:
        """The URL to send the user to.

        Args:
            scope: The OAuth scopes to request; `None` means the standard Grok CLI scopes.
            extra_params: Additional query parameters, merged over the defaults (so they can also
                override them), except `client_id` and `redirect_uri`: `exchange_code()` always
                posts the public client id and the flow's `redirect_uri`, so overriding either
                here would make the authorization code unusable.
        """
        params: dict[str, str] = {
            'response_type': 'code',
            'client_id': _PUBLIC_CLIENT_ID,
            'redirect_uri': self.redirect_uri,
            'scope': scope if scope is not None else _DEFAULT_SCOPE,
            'state': self.state,
            'nonce': self.nonce,
            'code_challenge': self.code_challenge,
            'code_challenge_method': 'S256',
        }
        return f'{_AUTHORIZE_URL}?{urlencode(self._merge_extra_params(params, extra_params))}'

    async def exchange_code(self, code: str) -> XaiOAuthCredentials:
        """Exchange an authorization code for credentials (call this in your callback handler)."""
        data = await post_token_request(
            _TOKEN_URL,
            {
                'grant_type': 'authorization_code',
                'code': code,
                'code_verifier': self.code_verifier,
                'redirect_uri': self.redirect_uri,
                'client_id': _PUBLIC_CLIENT_ID,
            },
            response_type=_TokenResponse,
            error=CredentialsRefreshError,
        )
        return _credentials_from_token_response(data)
