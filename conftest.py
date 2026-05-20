import time
import pytest

# In-repo conftest so fixtures are available to tests under src/ and tests/

@pytest.fixture
def vector_store():
    class DummyVectorStore:
        def __init__(self):
            self._store = {}

        def embed_and_cache(self, id_: str, type_: str, content: str, ttl_hours: float = 1.0):
            expires_at = time.time() + ttl_hours * 3600
            lst = self._store.setdefault(type_, [])
            lst.append({"id": id_, "content": content, "expires_at": expires_at})

        def get_by_type(self, type_: str):
            now = time.time()
            items = self._store.get(type_, [])
            return [i for i in items if i.get("expires_at", 0) > now]

        def clear_stale(self):
            now = time.time()
            removed = 0
            for t, items in list(self._store.items()):
                keep = [i for i in items if i.get("expires_at", 0) > now]
                removed += len(items) - len(keep)
                if keep:
                    self._store[t] = keep
                else:
                    del self._store[t]
            return removed

        def semantic_search(self, query, namespace, top_k=3):
            # simple heuristic: return high score for queries with 'sad' or 'test'
            if "sad" in query or "test" in query or "crying" in query:
                return [{"similarity": 0.9, "id": "1", "metadata": {}}]
            return [{"similarity": 0.1, "id": "0", "metadata": {}}]

    return DummyVectorStore()


@pytest.fixture
def embeddings():
    return {"dummy": [0.1, 0.2, 0.3]}


@pytest.fixture
def mood_analyzer():
    class DummyMoodAnalyzer:
        def extract_mood_keywords(self, text: str):
            lower = text.lower()
            if "sad" in lower:
                return ["sad"]
            if "workout" in lower:
                return ["workout"]
            return ["chill"]

        def analyze_correlations(self, min_samples=1):
            return {"sad": [{"confidence": 0.8}]}

    return DummyMoodAnalyzer()


@pytest.fixture
def cache_store():
    class SimpleCache:
        def __init__(self):
            self._store = {}
            self._hits = 0
            self._requests = 0

        def set(self, key, value, ttl_seconds=3600):
            self._store[key] = (value, time.time() + ttl_seconds)

        def get(self, key):
            self._requests += 1
            val = self._store.get(key)
            if not val:
                return None
            value, exp = val
            if exp < time.time():
                del self._store[key]
                return None
            self._hits += 1
            return value

        def get_stats(self):
            if self._requests == 0:
                return {"hit_rate": 0.0}
            return {"hit_rate": self._hits / self._requests}

    return SimpleCache()
