import redis
import json
import hashlib
import config

CACHE_TTL = 3600  # 1 hour in seconds
_client   = None

def get_redis():
    global _client
    if _client is None:
        _client = redis.Redis(
            host=config.REDIS_HOST,
            port=6379,
            decode_responses=True
        )
    return _client

def make_key(query: str, top_k: int,
             use_reranking: bool) -> str:
    raw = f"{query.lower().strip()}:{top_k}:{use_reranking}:{config.DENSE_MODEL}"
    return "search:" + hashlib.md5(
        raw.encode()
    ).hexdigest()

def get_cached(query: str, top_k: int,
               use_reranking: bool):
    try:
        r   = get_redis()
        key = make_key(query, top_k, use_reranking)
        val = r.get(key)
        if val:
            return json.loads(val)
    except Exception:
        pass
    return None

def set_cache(query: str, top_k: int,
              use_reranking: bool, data: dict):
    try:
        r   = get_redis()
        key = make_key(query, top_k, use_reranking)
        r.setex(key, CACHE_TTL, json.dumps(data))
    except Exception:
        pass

def is_available() -> bool:
    try:
        get_redis().ping()
        return True
    except Exception:
        return False