"""
This file will send hehartbeat pings and critical alert to the discord channel serer through webhook. It will never crash the bot, just logs error. This will give real time viisbility into bot health without needing to check the bot every single time. It will use doscrod incoming webhook api
https://support.discord.com/hc/en-us/articles/228383668-Intro-to-Webhooks
"""
import requests
import logging
from datetime import datetime, timezone
from core.config import DISCORD_WEBHOOK_URL

logger=logging.getLogger(__name__)


def _utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _post_to_discord(payload:dict):
    """
    This will post a json payload to discord webhook and if it wokrs it returns true else fail, never an exception that will crash bot
    """
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD WEBHOOK URL IS INVALID or NOT SET YET, SKIPPING NOTI")
        return False
    try:
        response=requests.post(DISCORD_WEBHOOK_URL,json=payload,timeout=10)
        response.raise_for_status()
        return True
    except requests.exceptions.Timeout:
        logger.warning("Discord webhook has timed out, skipping right now")
        return False
    except requests.exceptions.RequestException as e:
        logger.warning(f"Discord webhook has failed to work: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpeted Discord Error that was not accounted for: {e}")
        return False
    
def send_heartbeat(bot_name: str,is_alive: bool,portfolio_value: float, last_trade_time: str, last_sync_time: str, extra_info: str = ""  ):
    """
    per cycle send a heartbeat to Discord
      the arguments are:
         bot_name : "stable" or "risky1" or ... whatever else is there"
        is_alive: True = healthy cycle, False = error state
        portfolio_value : Current total portfolio value in USD
        last_trade_time : ISO timestamp string of last trade
        last_sync_time : ISO timestamp string of last data sync
        extra_info: Optional extra context line that is included

    """

    status_emoji="🟢" if is_alive else "🔴"
    status_text="ALIVE" if is_alive else "DEAD and ERROR PROBLEM NOW"
    timestamp=_utc_stamp()

    #clean discord embded using online template
    embed = {
        "title": f"{status_emoji} SinghQuant Heartbeat — {bot_name}",
        "color": 0x00FF00 if is_alive else 0xFF0000,  # Green or Red
        "fields": [
            {
                "name": "📡 Status",
                "value": status_text,
                "inline": True
            },
            {
                "name": "💰 Portfolio Value",
                "value": f"${portfolio_value:,.2f}",
                "inline": True
            },
            {
                "name": "🕐 Last Trade",
                "value": last_trade_time or "N/A",
                "inline": False
            },
            {
                "name": "🔄 Last Data Sync",
                "value": last_sync_time or "N/A",
                "inline": False
            },
        ],
        "footer": {"text": f"Oracle ARM • {timestamp}"},
    }

    if extra_info:
        embed["fields"].append({
            "name": "📝 Notes",
            "value": extra_info,
            "inline": False
        })

    return _post_to_discord({"embeds": [embed]})


def send_alert(bot_name: str,alert_type: str,message: str,portfolio_value: float = 0.0):
    """
    sends critical alert to the discord such as a kill switch being triggered
    bot_name: Which bot triggered the alert
    alert_type: e.g. "KILL SWITCH", "DRAWDOWN", "API ERROR"
    message: Human-readable description
    portfolio_value Portfolio value at time of alert
    """
    timestamp=_utc_stamp()
    embed={
        "title": f"🚨 ALERT — {bot_name} | {alert_type}",
        "description": message,
        "color": 0xFF4500,  # Orange-red for alerts
        "fields": [
            {
                "name": "💰 Portfolio at Alert",
                "value": f"${portfolio_value:,.2f}",
                "inline": True
            },
            {
                "name": "⏰ Time",
                "value": timestamp,
                "inline": True
            }
        ],
        "footer": {"text": "SinghQuant auto alert system"}
    }
    return _post_to_discord({"embeds": [embed]})