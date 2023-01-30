import pandas as pd
from prophet import Prophet
import yfinance as yf
import datetime
import streamlit as st

# Lista de tickers para baixar dados de ações
tickers = [
    "AAPL",
    "NU",
    "AMZN",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
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
        "Preco Previsto (7 dias)",
        "Preco Previsto (15 dias)",
        "Preco Previsto (30 dias)",
        "Sugestao",
        "Porcentagem de lucro",
        "Erro médio percentual de previsao",
    ]
)


def executar():

    for ticker in tickers:

        end = datetime.datetime.now() - datetime.timedelta(days=1)
        start = datetime.datetime.now() - datetime.timedelta(days=365 * 5)
        df_temp = yf.download(ticker, start=start, end=end)
        df_temp.index = df_temp.index.tz_localize(None)
        df_temp = df_temp.rename(columns={"Close": "y"})
        df_temp["ds"] = df_temp.index
        df_temp = df_temp[["ds", "y"]]

        # Inicializar o modelo Prophet
        model = Prophet()
        model.fit(df_temp)

        # Criar previsões futuras
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecast.set_index("ds", inplace=True)

        preco_atual = df_temp["y"].iloc[-1]
        preco_previsto_30 = forecast["yhat"].iloc[-1]
        preco_previsto_15 = forecast["yhat"].iloc[-15]
        preco_previsto_7 = forecast["yhat"].iloc[-23]
        sugestao = "Comprar" if preco_atual < preco_previsto_30 else "Vender"
        porcentagem = ((preco_previsto_30 / preco_atual) * 100) - 100

        erro_medio = (
            (abs(forecast["yhat"] - df_temp["y"])).mean(skipna=True) / preco_atual
        ) * 100

        # Analisar previsões e decidir se devemos comprar ou vender ações
        lista_temp = [
            ticker,
            preco_atual,
            preco_previsto_7,
            preco_previsto_15,
            preco_previsto_30,
            sugestao,
            porcentagem,
            erro_medio,
        ]
        analise.loc[len(analise)] = lista_temp

    return analise


st.set_page_config(
    page_title="Ações Previsão",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Ações Previsão")

if st.button("Executar"):
    st.dataframe(executar(), use_container_width=True)
