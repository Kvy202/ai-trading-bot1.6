import ccxt
from typing import Optional
from config import EXCHANGE_ID, API_KEY, API_SECRET, API_PASSWORD

def live_client(exchange_id: Optional[str] = None):
    ex_id = exchange_id or EXCHANGE_ID
    cls = getattr(ccxt, ex_id)
    params = {
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'} if ex_id == 'mexc' else {}
    }
    if API_PASSWORD:
        params['password'] = API_PASSWORD
    return cls(params)

def market_buy(client, symbol: str, amount: float):
    return client.create_order(symbol=symbol, type='market', side='buy', amount=amount)

def market_sell(client, symbol: str, amount: float):
    return client.create_order(symbol=symbol, type='market', side='sell', amount=amount)
