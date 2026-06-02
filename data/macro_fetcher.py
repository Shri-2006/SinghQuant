import os
import requests
#import google.generativeai as genai #DEPRECATED
from google import genai

from dotenv import load_dotenv

load_dotenv()


SEARXNG_URL  = "http://localhost:8080/search"
MACRO_QUERY  = (
    "stock market crash bank failure circuit breaker federal reserve SEC lawsuit "
    "site:reuters.com OR site:bloomberg.com OR site:cnbc.com OR site:apnews.com "
    "OR site:cnn.com OR site:yahoo.com OR site:ft.com OR site:wsj.com"
)


#SAP AI Core stuff
SAP_AUTH_URL= os.getenv("SAP_AUTH_URL")
SAP_CLIENT_ID= os.getenv("SAP_CLIENT_ID")
SAP_CLIENT_SECRET= os.getenv("SAP_CLIENT_SECRET")
SAP_AI_API_URL= os.getenv("SAP_AI_API_URL")
SAP_DEPLOYMENT_ID= os.getenv("SAP_ORCHESTRATION_DEPLOYMENT_ID")
RESOURCE_GROUP= os.getenv("RESOURCE_GROUP", "default")
#aistudios from google as failsafe
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")

#list of words and scores by Claude in regard to this project
KEYWORD_SCORES = {
    "bank failure":          4,
    "exchange halt":         4,
    "market circuit breaker":4,
    "financial crisis":      3,
    "sec lawsuit":           3,
    "fed emergency":         4,
    "market crash":          3,
    "liquidity crisis":      4,
    "bank run":              4,
    "federal reserve interest rate": 2,
    "rate hike":             1,
    "inflation surge":       2,
    "recession fears":       2,
    "market selloff":        2,
    "earnings miss":         1,
    "debt ceiling":          2,
    "credit downgrade":      3,
}
NEGATIVE_KEYWORDS = {
    "all-time high":  3,
    "record high":    3,
    "market rally":   2,
    "stocks rise":    2,
    "bull market":    2,
    "market gains":   2,
    "stocks climb":   2,
    "market soars":   2,
    "new high":       2,
    "stocks surge":   2,
}
SCORE_CLEAR=2 #clear means its fine 
SCORE_DANGER=12 #anything greater than or equal to this is bad

def _search(query):
    """
    Fetches search results from SearXNG and returns all the text as a single lowercase string, and if it fails it will return a empty string.
    """
    try:
        response = requests.get(SEARXNG_URL,params={"q": query, "format": "json"},timeout=10)
        data = response.json()
        text_parts = []
        for result in data.get("results", []):
            text_parts.append(result.get("title", ""))
            text_parts.append(result.get("content", ""))
        return " ".join(text_parts).lower()
    except Exception:
        return ""
    

def _score_keywords(text):
    """
    Scores search result text against keyword lists.
    Positive keywords add to score, negative keywords subtract.
    Returns (total_score, list of matched positive keywords).
    """
    total = 0
    hits = []
    for keyword, weight in KEYWORD_SCORES.items():
        if keyword in text:
            total += weight
            hits.append(keyword)
    for keyword, weight in NEGATIVE_KEYWORDS.items():
        if keyword in text:
            total -= weight
    return (total, hits)

def _sap_is_configured():
    return all([SAP_AUTH_URL, SAP_CLIENT_ID, SAP_CLIENT_SECRET,SAP_AI_API_URL, SAP_DEPLOYMENT_ID])

def _get_sap_token():
    """Fetches OAuth2 token from SAP."""
    response = requests.post(f"{SAP_AUTH_URL}/oauth/token",
                             data={"grant_type": "client_credentials"},auth=(SAP_CLIENT_ID, SAP_CLIENT_SECRET),   timeout=10)
    return response.json()["access_token"]

#adjusted using Gemini 2.5 Flash from the SAP AI Knowledge System I developed a few weeks ago
def _classify_with_sap(text):
    """
    Sends search text to SAP AI Orchestration for classification. Returns DANGER, CAUTION, CLEAR, or None if it fails.
    """
    try:
        token = _get_sap_token()
        url = f"{SAP_AI_API_URL}/v2/inference/deployments/{SAP_DEPLOYMENT_ID}/completion"

        payload = {
            "orchestration_config": {
                "module_configurations": {
                    "templating_module_config": {
                        "template": [
                            {
                                "role": "user",
                                "content": (
                                    "Classify the macro market risk based on this financial news text. "
                                    "Output ONLY one word with no explanation: DANGER, CAUTION, or CLEAR. "
                                    "DANGER = severe market risk event. "
                                    "CAUTION = elevated uncertainty. "
                                    "CLEAR = normal conditions. "
                                    "Text: {{?user_input}}"
                                )
                            }
                        ]
                    },
                    "llm_module_config": {
                        "model_name": "gemini-2.5-flash-lite",
                        "model_params": {"temperature": 0, "max_tokens": 5}
                    }
                }
            },
            "input_params": {"user_input": text[:3000]}
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": RESOURCE_GROUP,
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=15)
        result = response.json()

        signal = result["orchestration_result"]["choices"][0]["message"]["content"].strip().upper()

        if signal in {"DANGER", "CAUTION", "CLEAR"}:
            return signal
        return None

    except Exception as e:
        print(f"[macro_fetcher] SAP classification failed: {e}")
        return None# THIS IS GEMINI FALLBACK BTW

def _gemini_is_configured():
    return bool(GEMINI_API_KEY)

def _classify_with_gemini(text):
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = (
            "Classify the macro market risk based on this financial news text. "
            "Output ONLY one word with no explanation: DANGER, CAUTION, or CLEAR. "
            "DANGER = severe market risk event. "
            "CAUTION = elevated uncertainty. "
            "CLEAR = normal conditions. "
            f"Text: {text[:3000]}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        signal = response.text.strip().upper()
        if signal in {"DANGER", "CAUTION", "CLEAR"}:
            return signal
        return None
    except Exception as e:
        print(f"[macro_fetcher] Gemini classification failed: {e}")
        return None

#call ai

def _classify_with_ai(content):
    """
    Tries SAP orchestration for SAP AI Core first, then tries Gemini. If both of these fails returns None and relies entirely on Keyword Fallback
    """
    if _sap_is_configured():
        res=_classify_with_sap(content)
        if res:
            print(f"[macro_fetcher] SAP classified: {res}")
            return res
        
    if _gemini_is_configured():
        res=_classify_with_gemini(content)
        if res:
            print(f"[macro_fetcher] Gemini classified: {res}")
            return res
    return None

import time
_macro_cache = {"signal": None, "timestamp": 0}
CACHE_TTL = 3600  # 1 hour


def get_macro_signal():
    global _macro_cache
    if _macro_cache["signal"] and (time.time() - _macro_cache["timestamp"]) < CACHE_TTL:
        return _macro_cache["signal"]
    
    try:
        results_text = _search(MACRO_QUERY)
        if not results_text:
            return "CLEAR"  # don't cache failures

        score, hits = _score_keywords(results_text)
        print(f"[macro_fetcher] keyword score={score}, hits={hits}")

        if score <= SCORE_CLEAR:
            signal = "CLEAR"
        elif score >= SCORE_DANGER:
            signal = "DANGER"
        else:
            ai_result = _classify_with_ai(results_text)
            signal = ai_result if ai_result else ("CAUTION" if score >= 3 else "CLEAR")

        # ✅ Cache the result
        _macro_cache = {"signal": signal, "timestamp": time.time()}
        return signal

    except Exception:
        print("[macro_fetcher] Unexpected error - defaulting to CLEAR")
        return "CLEAR"