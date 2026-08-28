"""Tests for the xAI OAuth login primitives."""

from __future__ import annotations

import base64
import hashlib
import pickle
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx2
import pytest
from pydantic import SecretStr

from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.providers.xai_oauth import (
    CredentialsRefreshError,
    XaiOAuthCredentials,
    XaiOAuthFlow,
    _credentials_from_token_response,  # pyright: ignore[reportPrivateUsage]
    _TokenResponse,  # pyright: ignore[reportPrivateUsage]
    refresh_credentials,
)

pytestmark = pytest.mark.anyio


def make_credentials(expires_at: datetime | None = None) -> XaiOAuthCredentials:
    return XaiOAuthCredentials(
        access_token=SecretStr('access-1'), refresh_token=SecretStr('refresh-1'), expires_at=expires_at
    )


# --- Authorization URL ---


def test_authorization_url_shape():
    flow = XaiOAuthFlow(state='my-state')
    url = flow.authorization_url()

    parsed = urlparse(url)
    assert f'{parsed.scheme}://{parsed.netloc}{parsed.path}' == 'https://auth.x.ai/oauth2/authorize'
    params = {name: values[0] for name, values in parse_qs(parsed.query).items()}
    assert params['response_type'] == 'code'
    assert params['client_id'] == 'b1a00492-073a-47ea-816f-4c329264a828'
    assert params['redirect_uri'] == 'http://127.0.0.1:56121/callback'
    assert params['scope'] == 'openid profile email offline_access grok-cli:access api:access'
    assert params['state'] == 'my-state'
    assert params['nonce'] == flow.nonce
    assert params['code_challenge_method'] == 'S256'
    challenge = base64.urlsafe_b64encode(hashlib.sha256(flow.code_verifier.encode()).digest()).rstrip(b'=')
    assert params['code_challenge'] == challenge.decode()


def test_authorization_url_extra_params_add_and_override():
    flow = XaiOAuthFlow(state='my-state')
    url = flow.authorization_url(scope='openid', extra_params={'prompt': 'login'})

    params = {name: values[0] for name, values in parse_qs(urlparse(url).query).items()}
    assert params['prompt'] == 'login'
    assert params['scope'] == 'openid'


def test_authorization_url_rejects_identity_overrides():
    flow = XaiOAuthFlow()
    with pytest.raises(UserError, match='cannot override client_id, redirect_uri'):
        flow.authorization_url(extra_params={'client_id': 'other', 'redirect_uri': 'https://example.com/cb'})


def test_flexible_redirect_uri():
    """Unlike the pinned Codex redirect, xAI accepts any localhost port."""
    flow = XaiOAuthFlow(redirect_uri='http://127.0.0.1:0/callback')
    params = {name: values[0] for name, values in parse_qs(urlparse(flow.authorization_url()).query).items()}
    assert params['redirect_uri'] == 'http://127.0.0.1:0/callback'


# --- Token responses and credentials ---


def test_token_response_validation_errors():
    with pytest.raises(CredentialsRefreshError, match='access_token'):
        _credentials_from_token_response(_TokenResponse())
    with pytest.raises(CredentialsRefreshError, match='refresh_token'):
        _credentials_from_token_response(_TokenResponse(access_token='a'))


def test_expires_at_derived_from_expires_in():
    before = datetime.now(timezone.utc)
    credentials = _credentials_from_token_response(_TokenResponse(access_token='a', refresh_token='r', expires_in=3600))
    assert credentials.expires_at is not None
    assert before + timedelta(seconds=3500) < credentials.expires_at < before + timedelta(seconds=3700)
    assert not credentials.is_stale()

    no_expiry = _credentials_from_token_response(_TokenResponse(access_token='a', refresh_token='r'))
    assert no_expiry.expires_at is None
    assert not no_expiry.is_stale()


def test_is_stale_honors_buffer():
    now = datetime.now(timezone.utc)
    assert make_credentials(expires_at=now + timedelta(seconds=30)).is_stale()  # inside the 120s buffer
    assert make_credentials(expires_at=now + timedelta(seconds=3600)).is_stale() is False
    assert make_credentials(expires_at=None).is_stale() is False


def test_credentials_repr_hides_secrets():
    rendered = repr(make_credentials())
    assert 'access-1' not in rendered
    assert 'refresh-1' not in rendered


def test_refresh_error_is_model_api_error():
    exc = CredentialsRefreshError('something broke')
    assert isinstance(exc, ModelAPIError)
    assert exc.model_name == 'xai'
    restored = pickle.loads(pickle.dumps(exc))
    assert type(restored) is CredentialsRefreshError
    assert restored.message == 'something broke'


# --- Exchange and refresh against the token endpoint ---


def _token_endpoint_client(response: httpx2.Response, forms: list[dict[str, Any]]) -> httpx2.AsyncClient:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        forms.append({name: values[0] for name, values in parse_qs(request.content.decode()).items()})
        return response

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def test_refresh_posts_refresh_grant_and_keeps_old_refresh_token():
    """xAI may omit `refresh_token` from a refresh response: the held grant is kept."""
    forms: list[dict[str, Any]] = []
    client = _token_endpoint_client(httpx2.Response(200, json={'access_token': 'access-2'}), forms)

    async with client:
        rotated = await refresh_credentials(make_credentials(), http_client=client)

    assert forms == [
        {
            'grant_type': 'refresh_token',
            'refresh_token': 'refresh-1',
            'client_id': 'b1a00492-073a-47ea-816f-4c329264a828',
        }
    ]
    assert rotated.access_token.get_secret_value() == 'access-2'
    assert rotated.refresh_token.get_secret_value() == 'refresh-1'  # kept, not rotated


async def test_refresh_rejected_grant_surfaces_hint():
    forms: list[dict[str, Any]] = []
    client = _token_endpoint_client(
        httpx2.Response(400, json={'error': 'invalid_grant', 'error_description': 'expired'}), forms
    )

    async with client:
        with pytest.raises(CredentialsRefreshError, match='expired; the grant was rejected'):
            await refresh_credentials(make_credentials(), http_client=client)


async def test_exchange_code_posts_pkce_verifier(monkeypatch: pytest.MonkeyPatch):
    flow = XaiOAuthFlow()
    captured: dict[str, Any] = {}

    async def fake_post(url: str, form: dict[str, str], **kwargs: Any) -> _TokenResponse:
        captured['url'] = url
        captured['form'] = form
        return _TokenResponse(access_token='access-1', refresh_token='refresh-1', expires_in=3600)

    monkeypatch.setattr('pydantic_ai.providers.xai_oauth.post_token_request', fake_post)
    credentials = await flow.exchange_code('the-code')

    assert captured['url'] == 'https://auth.x.ai/oauth2/token'
    assert captured['form'] == {
        'grant_type': 'authorization_code',
        'code': 'the-code',
        'code_verifier': flow.code_verifier,
        'redirect_uri': flow.redirect_uri,
        'client_id': 'b1a00492-073a-47ea-816f-4c329264a828',
    }
    assert credentials.access_token.get_secret_value() == 'access-1'
