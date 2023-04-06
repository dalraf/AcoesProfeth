import pandas as pd
import yfinance as yf
from prophet import Prophet
from datetime import datetime, timedelta

import logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)


def executar():
    tickers = [
        "PETR4.SA",
        "AAPL",
        "NU",
        "AMZN",
        "GOOGL",
        "TSLA",
        "META",
        "MSFT",
        "INTC",
        "CSCO",
        "NVDA",
        "PYPL",
        "NFLX",
        "IBM",
        "BTC-USD",
        "ETH-USD",
    ]

    analise = pd.DataFrame(
        columns=[
            "Acao",
            "Preco Atual",
            "Variação (7 dias) %",
            "Variação (15 dias) %",
            "Variação (30 dias) %",
        ]
    )

    for ticker in tickers:
        end = datetime.now() - timedelta(days=1)
        start = datetime.now() - timedelta(days=365 * 2)
        df_temp = yf.download(ticker, start=start, end=end)
        df_temp = df_temp.rename(columns={"Close": "y"})
        df_temp = df_temp[["y"]]
        df_temp.dropna(inplace=True)

        # Preparar o dataframe para o Prophet
        df_temp.reset_index(inplace=True)
        df_temp = df_temp.rename(columns={"Date": "ds"})

        # Inicializar o modelo Prophet
        print("Treinando modelo para a ação: " + ticker)
        model = Prophet(seasonality_mode='multiplicative', daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df_temp)

        # Criar previsões futuras
        print("Criando previsões para a ação: " + ticker)
        future = model.make_future_dataframe(periods=30, freq='D')
        forecast = model.predict(future)
        preco_atual = df_temp["y"].iloc[-1]
        variacao_prevista_30 = ((forecast["yhat"].iloc[-1] - preco_atual) / preco_atual) * 100
        variacao_prevista_15 = ((forecast["yhat"].iloc[-15] - preco_atual) / preco_atual) * 100
        variacao_prevista_7 = ((forecast["yhat"].iloc[-23] - preco_atual) / preco_atual) * 100

        # Analisar previsões e decidir se devemos comprar ou vender ações
        lista_temp = [
            ticker,
            preco_atual,
            variacao_prevista_7,
            variacao_prevista_15,
            variacao_prevista_30,
        ]
        analise.loc[len(analise)] = lista_temp

    return analise
