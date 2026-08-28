"""Shared authorization-code + PKCE primitives for provider login flows.

Provider OAuth flows (e.g. OpenAI Codex) subclass [`OAuthFlow`][pydantic_ai.providers._oauth.OAuthFlow]
so they share one shape: construction does no I/O, `authorization_url()` builds the redirect, and
`exchange_code()` turns the callback's authorization code into provider credentials.
"""

from __future__ import annotations as _annotations

import base64
import hashlib
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Generic, TypeVar

import httpx2
from pydantic import BaseModel

from pydantic_ai.exceptions import ModelAPIError, UserError

CredentialsT = TypeVar('CredentialsT')
ResponseT = TypeVar('ResponseT', bound=BaseModel)


class CredentialsError(ModelAPIError):
    """Base for provider credential failures.

    Subclasses [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] so the standard handling of
    provider failures (e.g. [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel]) applies;
    the auth layer runs below any specific model, so `model_name` is the provider name, which
    subclasses pin via `provider_name`.
    """

    provider_name: ClassVar[str]

    def __init__(self, message: str):
        super().__init__(model_name=self.provider_name, message=message)

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.message,)


async def post_token_request(
    url: str,
    form: Mapping[str, str],
    *,
    response_type: type[ResponseT],
    error: Callable[[str], Exception],
    http_client: httpx2.AsyncClient | None = None,
) -> ResponseT:
    """POST a form-urlencoded OAuth token request and validate the JSON response.

    When `http_client` is given the request goes through it (so custom transports and proxies apply
    to refreshes too); otherwise an ephemeral client is used. Failures raise `error(message)`.
    """
    if http_client is None:
        async with httpx2.AsyncClient(timeout=httpx2.Timeout(timeout=30, connect=5)) as client:
            response = await client.post(url, data=dict(form), headers={'Accept': 'application/json'})
    else:
        response = await http_client.post(url, data=dict(form), headers={'Accept': 'application/json'})
    if response.status_code != 200:
        try:
            body = _TokenErrorResponse.model_validate(response.json())
        except ValueError:
            body = _TokenErrorResponse()
        detail = body.error_description or body.error or response.text[:200]
        hint = '; the grant was rejected, rerun the authorization flow' if body.error == 'invalid_grant' else ''
        raise error(f'Token request to {url} failed with status {response.status_code}: {detail}{hint}')
    try:
        return response_type.model_validate(response.json())
    except ValueError:
        raise error(f'Token endpoint {url} returned an unexpected response.') from None


class _TokenErrorResponse(BaseModel):
    """An OAuth token-endpoint error body."""

    error: str | None = None
    error_description: str | None = None


class OAuthFlow(ABC, Generic[CredentialsT]):
    """Authorization-code + PKCE context for a provider's public OAuth client.

    Subclasses pin their provider's endpoints, client id, and credential type; the base carries
    the pieces every flow needs: the `state` guarding the callback, the PKCE verifier/challenge
    pair, and the redirect URI the authorization code is bound to.
    """

    def __init__(self, *, redirect_uri: str, state: str | None = None) -> None:
        self.redirect_uri = redirect_uri
        self.state = state or secrets.token_urlsafe(16)
        self.code_verifier = secrets.token_urlsafe(32)

    @property
    def code_challenge(self) -> str:
        """The S256 PKCE challenge derived from `code_verifier`."""
        digest = hashlib.sha256(self.code_verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b'=').decode()

    def _merge_extra_params(self, params: dict[str, str], extra_params: Mapping[str, str] | None) -> dict[str, str]:
        """Merge caller-supplied query parameters over the defaults, refusing identity overrides."""
        if extra_params:
            if overridden := sorted({'client_id', 'redirect_uri'} & extra_params.keys()):
                raise UserError(
                    f'`extra_params` cannot override {", ".join(overridden)}: `exchange_code()` always posts '
                    "the public client id and the flow's `redirect_uri`, so the authorization code would be "
                    'unusable. Pass `redirect_uri=` to the constructor instead.'
                )
            params.update(extra_params)
        return params

    @abstractmethod
    def authorization_url(self, *, scope: str | None = None, extra_params: Mapping[str, str] | None = None) -> str:
        """The URL to send the user to; `scope=None` means the provider's default scopes."""
        ...

    @abstractmethod
    async def exchange_code(self, code: str) -> CredentialsT:
        """Exchange an authorization code for provider credentials (call this in your redirect handler)."""
        ...
