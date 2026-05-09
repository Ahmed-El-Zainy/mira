import httpx
import json
from utils.config import get_settings

class HuggingFaceSentimentTool:
    def __init__(self, token: str, model_id: str):
        self.token = token
        self.model_id = model_id
        self.url = f"https://api-inference.huggingface.co/models/{model_id}"

    async def get_sentiment(self, text: str) -> float:
        if not self.token:
            return 0.0
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"inputs": text[:512]} # Truncate for safety
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(self.url, headers=headers, json=payload, timeout=10.0)
                if resp.status_code != 200:
                    return 0.0
                # Finbert-tone returns [[{label: 'Positive', score: ...}, ...]]
                results = resp.json()[0]
                # Mapping logic:
                mapping = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
                top = max(results, key=lambda x: x['score'])
                return mapping.get(top['label'], 0.0) * top['score']
            except Exception:
                return 0.0
