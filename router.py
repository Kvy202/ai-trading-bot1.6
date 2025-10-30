import os, json, asyncio
from dotenv import load_dotenv
load_dotenv()

from multi_exchange import best_cex_quote_async

ROUTER_ENABLED = os.getenv("ENABLE_ROUTER", "0") == "1"
EX_LIST = [e.strip() for e in os.getenv("ROUTER_EXCHANGES", "binance,bybit,bitget,mexc").split(",") if e.strip()]
DEFAULT_TAKER_BPS = float(os.getenv("ROUTER_DEFAULT_TAKER_BPS", "10"))

# Optional private keys for future use (NOT required for analysis)
EX_KEYS_JSON = os.getenv("EX_KEYS_JSON", "{}")
try:
    EX_KEYS = json.loads(EX_KEYS_JSON)
except Exception:
    EX_KEYS = {}

def best_cex_quote(symbol: str, side: str, base_amount: float, market_type: str = "swap"):
    analyze_dex = os.getenv("ANALYZE_DEX", "0") == "1"
    q = asyncio.run(best_cex_quote_async(EX_LIST, symbol, side, base_amount, DEFAULT_TAKER_BPS, market_type=market_type, analyze_dex=analyze_dex))
    if q is None:
        raise RuntimeError("No venue returned a valid quote")
    # return a consistent dict INCLUDING mtype for your prints
    return {"venue": q.venue, "vwap": q.vwap, "eff_bps": q.eff_bps, "mtype": q.mtype}

# Optional factory if you later want authenticated clients
def client_for(exchange_id: str):
    import ccxt
    opts = {'enableRateLimit': True}
    if exchange_id == 'mexc':
        opts['options'] = {'defaultType': 'spot'}
    keys = EX_KEYS.get(exchange_id, {})
    if keys:
        opts.update({'apiKey': keys.get('apiKey',''), 'secret': keys.get('secret','')})
        if keys.get('password'):
            opts['password'] = keys['password']
    cls = getattr(ccxt, exchange_id)
    c = cls(opts)
    c.load_markets()
    return c
