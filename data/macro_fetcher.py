import os
import requests
#import google.generativeai as genai #DEPRECATED
from google import genai

from dotenv import load_dotenv

load_dotenv()


SEARXNG_URL  = "http://localhost:8080/search"
MACRO_QUERY  = "stock market crash bank failure circuit breaker federal reserve SEC lawsuit"


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

SCORE_CLEAR=1 #clear means its fine 
SCORE_DANGER=8 #anything greater than or equal to this is bad

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
    Scores the search result text against the keyword list.  Returns (total_score, list of matched keywords).
    """
    total = 0
    hits= []
    for keyword, weight in KEYWORD_SCORES.items():
        if keyword in text:
            total += weight
            hits.append(keyword)
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
        url   = f"{SAP_AI_API_URL}/v2/inference/deployments/{SAP_DEPLOYMENT_ID}/completion"

        payload = {
            "orchestration_config": {
                "templating_module_config": {
                    "template": (
                        "Classify the macro market risk based on this financial news text. "
                        "Output ONLY one word with no explanation: DANGER, CAUTION, or CLEAR. "
                        "DANGER = severe market risk event. "
                        "CAUTION = elevated uncertainty. "
                        "CLEAR = normal conditions. "
                        "Text: {{?user_input}}"
                    )
                },
                "llm_module_config": {
                    "model_name": "gemini-2.5-flash-lite",
                    "model_version": "latest",
                    "params": {"temperature": 0, "max_tokens": 5}
                }
            },
            "input_params": {"user_input": text[:3000]},
            "messages_history": []
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": RESOURCE_GROUP,
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=15)
        result   = response.json()

        # Extract the text response from orchestration output
        signal = result["choices"][0]["message"]["content"].strip().upper()
        if signal in {"DANGER", "CAUTION", "CLEAR"}:
            return signal
        return None

    except Exception as e:
        print(f"[macro_fetcher] SAP classification failed: {e}")
        return None

# THIS IS GEMINI FALLBACK BTW

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



def get_macro_signal():
    """
    Main function — call once per trading cycle in run().

    Flow:
        SearXNG fetch goes to keyword scoring which calls
            score <= 1  then CLEAR
            score >= 8  thenDANGER
            if ambiguous   then AI classification which if that fails goes to keyword fallback

    Always returns CLEAR on any failure -never blocks the bot, this plugs into existing VIX logic: DANGER raises effective VIX to 35, CAUTION raises it to 22.
    """
    try:
        results_text = _search(MACRO_QUERY)

        if not results_text:
            print("[macro_fetcher] SearXNG unavailable - defaulting to CLEAR")
            return "CLEAR"

        score, hits = _score_keywords(results_text)
        print(f"[macro_fetcher] keyword score={score}, hits={hits}")

        # Clear cut cases — no AI needed
        if score <= SCORE_CLEAR:
            return "CLEAR"
        if score >= SCORE_DANGER:
            return "DANGER"

        # Ambiguous zone — ask AI
        ai_result = _classify_with_ai(results_text)
        if ai_result:
            return ai_result

        # AI failed — fall back to keyword logic
        if score >= 3:
            return "CAUTION"
        return "CLEAR"

    except Exception:
        print("[macro_fetcher] Unexpected error -defaulting to CLEAR")
        return "CLEAR"