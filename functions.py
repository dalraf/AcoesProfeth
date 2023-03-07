import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yfinance as yf
import datetime

# Lista de tickers para baixar dados de ações
tickers = [
    "PETR4.SA",
    "AAPL",
    "NU",
    "AMZN",
    "GOOGL",
    "AMZN",
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


def executar():

    for ticker in tickers:

        end = datetime.datetime.now() - datetime.timedelta(days=1)
        start = datetime.datetime.now() - datetime.timedelta(days=365 * 5)
        df_temp = yf.download(ticker, start=start, end=end)
        df_temp = df_temp.rename(columns={"Close": "y"})
        df_temp = df_temp[["y"]]
        df_temp.dropna(inplace=True)
        df_temp.index = df_temp.index.tz_localize(None).to_period('D')

        # Inicializar o modelo SARIMAX
        print("Treinando modelo para a ação: " + ticker)
        model = SARIMAX(df_temp, order=(1,1,1), seasonal_order=(1,1,1,12))
        model_fit = model.fit()


        # Criar previsões futuras
        print('Criando previsões para a ação: ' + ticker)
        futuro_start = datetime.datetime.now() + datetime.timedelta(days=1)
        futuro_end = datetime.datetime.now() + datetime.timedelta(days=30)
        forecast = model_fit.predict(futuro_start, futuro_end, freq="D")
        print(forecast)
        preco_atual = df_temp["y"].iloc[-1]
        variacao_prevista_30 = ((forecast.iloc[-1] - preco_atual) / preco_atual) * 100
        variacao_prevista_15 = ((forecast.iloc[-15] - preco_atual) / preco_atual) * 100
        variacao_prevista_7 = ((forecast.iloc[-23] - preco_atual) / preco_atual) * 100

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