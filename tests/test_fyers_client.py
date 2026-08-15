import pytest
from unittest.mock import MagicMock
from fyers_client import (
    get_session_model,
    generate_auth_link,
    get_access_token,
    fetch_fyers_quotes
)

def test_get_session_model(mocker):
    mock_session_model = mocker.patch("fyers_apiv3.fyersModel.SessionModel")
    get_session_model("client_id", "secret_key", "redirect_uri")
    mock_session_model.assert_called_once_with(
        client_id="client_id",
        secret_key="secret_key",
        redirect_uri="redirect_uri",
        response_type="code",
        grant_type="authorization_code"
    )

def test_generate_auth_link():
    mock_session = MagicMock()
    mock_session.generate_authcode.return_value = "https://auth-link.com"
    link = generate_auth_link(mock_session)
    assert link == "https://auth-link.com"

def test_get_access_token():
    mock_session = MagicMock()
    mock_session.generate_token.return_value = {"access_token": "mocked_access_token"}
    token = get_access_token(mock_session, "auth_code")
    mock_session.set_token.assert_called_once_with("auth_code")
    assert token == "mocked_access_token"

def test_fetch_fyers_quotes_empty():
    res = fetch_fyers_quotes("token", "client", [])
    assert res == {}

def test_fetch_fyers_quotes_success(mocker):
    # Mock FyersModel
    mock_fyers_instance = MagicMock()
    # Mock return response dictionary
    mock_fyers_instance.quotes.return_value = {
        "s": "ok",
        "d": [
            {
                "n": "NSE:RELIANCE-EQ",
                "v": {
                    "lp": 2500.0,
                    "chp": 1.2,
                    "vol": 500000,
                    "bid": 2499.0,
                    "ask": 2501.0,
                    "ltq": 5
                }
            }
        ]
    }
    
    mocker.patch("fyers_apiv3.fyersModel.FyersModel", return_value=mock_fyers_instance)
    
    res = fetch_fyers_quotes("token", "client", ["NSE:RELIANCE-EQ"])
    assert "RELIANCE.NS" in res
    assert res["RELIANCE.NS"]["symbol"] == "RELIANCE"
    assert res["RELIANCE.NS"]["ltp"] == 2500.0
    assert res["RELIANCE.NS"]["volume"] == 500000
    assert res["RELIANCE.NS"]["bid_price"] == 2499.0
    assert res["RELIANCE.NS"]["ask_price"] == 2501.0
