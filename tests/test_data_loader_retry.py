# tests/test_data_loader_retry.py
from unittest.mock import patch, MagicMock
import requests
import pytest
import pandas as pd
from src.data_loader import fetch_load_forecast


def test_load_forecast_retries_on_503():
    """fetch_load_forecast should retry on 503 and succeed on the 3rd attempt."""
    fake_response = MagicMock()
    fake_response.status_code = 503
    error_503 = requests.exceptions.HTTPError("503 Server Error", response=fake_response)

    # Mock the EntsoePandasClient to fail twice then succeed
    with patch("src.data_loader.load_entsoe_token", return_value="fake_token"):
        with patch("entsoe.EntsoePandasClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            # First two calls raise 503, third returns a DataFrame
            import pandas as pd
            success_df = pd.DataFrame(
                {"Forecasted Load": [1000.0, 1100.0]},
                index=pd.date_range("2026-05-19", periods=2, freq="h", tz="Europe/Athens"),
            )
            mock_client.query_load_forecast.side_effect = [error_503, error_503, success_df]

            # Should succeed after retries
            result = fetch_load_forecast(
                start=pd.Timestamp("2026-05-19", tz="Europe/Athens"),
                end=pd.Timestamp("2026-05-20", tz="Europe/Athens"),
                country_code="GR",
                api_token="fake_token",
            )

            assert len(result) == 2
            assert mock_client.query_load_forecast.call_count == 3


def test_load_forecast_does_not_retry_on_401():
    """4xx errors (auth) should NOT be retried — fail fast."""
    fake_response = MagicMock()
    fake_response.status_code = 401
    error_401 = requests.exceptions.HTTPError("401 Unauthorized", response=fake_response)

    with patch("src.data_loader.load_entsoe_token", return_value="fake_token"):
        with patch("entsoe.EntsoePandasClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.query_load_forecast.side_effect = [error_401, error_401, error_401]

            with pytest.raises(requests.exceptions.HTTPError):
                fetch_load_forecast(
                    start=pd.Timestamp("2026-05-19", tz="Europe/Athens"),
                    end=pd.Timestamp("2026-05-20", tz="Europe/Athens"),
                    country_code="GR",
                    api_token="fake_token",
                )

            # Should fail on the FIRST attempt, not 3
            assert mock_client.query_load_forecast.call_count == 1