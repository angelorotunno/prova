# Frontiera efficiente di Markowitz su ETF

Costruzione della frontiera efficiente per un paniere di ETF con dati reali di mercato, secondo la teoria di portafoglio media-varianza di Markowitz.

## Cosa fa

1. Scarica i prezzi mensili aggiustati (dividendi e split) da [Alpha Vantage](https://www.alphavantage.co).
2. Calcola rendimenti attesi e matrice di covarianza annualizzati.
3. Trova, con vincolo di sole posizioni lunghe:
   - il **portafoglio a varianza minima**;
   - il **portafoglio a massimo Sharpe** (portafoglio tangente).
4. Disegna la frontiera efficiente, i singoli ETF e la Capital Market Line.

## Paniere di default

| Ticker | Asset class |
|---|---|
| SPY | Azionario USA |
| EFA | Azionario paesi sviluppati ex USA |
| AGG | Obbligazionario USA aggregato |
| GLD | Oro |
| VNQ | Immobiliare (REIT) USA |

## Come eseguirlo

```bash
pip install -r requirements.txt
export ALPHAVANTAGE_API_KEY=la_tua_chiave   # gratuita su alphavantage.co
python efficient_frontier.py --start 2010-01 --rf 0.02
```

Opzioni:

- `--tickers SPY QQQ TLT ...` per cambiare paniere
- `--start AAAA-MM` per scegliere il primo mese del campione
- `--rf 0.03` per il tasso privo di rischio annuo
- `--refresh` per riscaricare i dati (altrimenti usa la cache in `data/`)

## Output

- `results/frontier.png`: grafico della frontiera
- `results/weights.csv`: pesi, rendimento, volatilità e Sharpe dei due portafogli ottimi

## Limiti

- I rendimenti attesi stimati dalla media storica sono molto instabili: piccoli cambi del campione spostano molto i pesi ottimi.
- Il modello assume rendimenti con distribuzione stabile nel tempo e considera solo media e varianza.
- Esercizio a scopo didattico, non è una raccomandazione di investimento.
