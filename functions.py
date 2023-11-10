import pandas as pd
import yfinance as yf
from prophet import Prophet
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.metrics import mean_squared_error


import logging

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


def calculate_rmse(original_df, forecast_df):
    original_values = original_df["y"].values
    predicted_values = forecast_df["yhat"].values[: len(original_values)]
    rmse = mean_squared_error(original_values, predicted_values, squared=False)
    return rmse


def executar():
    tickers = [
        "BTC-USD",
        "ETH-USD",
        "BCH-USD",
        "AAVE-USD",
        "SOL-USD",
        "LINK-USD",
        "LTC-USD",
        "UNI7083-USD",
        "MATIC-USD",
    ]

    analise = pd.DataFrame(
        columns=[
            "Acao",
            "Preco Atual",
            "Indice NERF",
        ]
    )

    for ticker in tickers:
        end = datetime.now() - timedelta(days=1)
        start = datetime.now() - timedelta(days=365 * 4)
        df_temp = yf.download(ticker, start=start, end=end)

        # Preparar o dataframe para o Prophet
        df_temp = df_temp.rename(columns={"Close": "y"})
        df_temp = df_temp[["y"]]
        df_temp.reset_index(inplace=True)
        df_temp = df_temp.rename(columns={"Date": "ds"})

        rsi = RSIIndicator(df_temp["y"], window=14)
        df_temp["rsi"] = rsi.rsi()

        macd = MACD(df_temp["y"])
        df_temp["macd"] = macd.macd()

        df_temp.dropna(inplace=True)

        # Inicializar o modelo Prophet
        print("Treinando modelo para a ação: " + ticker)
        model = Prophet(
            seasonality_mode="multiplicative",
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
        )
        model.add_regressor("rsi")
        model.add_regressor("macd")
        model.fit(df_temp)

        # Criar previsões futuras
        print("Criando previsões para a ação: " + ticker)
        future = model.make_future_dataframe(periods=30, freq="D")
        future["rsi"] = rsi.rsi()
        future["macd"] = macd.macd()
        future.dropna(inplace=True)
        forecast = model.predict(future)
        preco_atual = df_temp["y"].iloc[-1]
        rmse = calculate_rmse(df_temp, forecast)
        indice_nerf = (
            (((forecast["yhat"].iloc[-24] - preco_atual) / preco_atual) * 100) / rmse
        ) * 1000
        lista_temp = [
            ticker,
            preco_atual,
            indice_nerf,
        ]
        analise.loc[len(analise)] = lista_temp
        analise_ordenado = df.sort_values(by="indice_nerf", ascending=False)


    return analise_ordenado
