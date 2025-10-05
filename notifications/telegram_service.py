"""Service de notifications Telegram pour signaux BUY"""

import os
import time
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Gère l'envoi de notifications Telegram pour les signaux de trading"""

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # Anti-spam: tracker des dernières notifications par symbole
        self._last_notification: Dict[str, datetime] = {}
        self._cooldown_minutes = 5  # Cooldown de 5 minutes entre notifications pour le même symbole

        if not self.bot_token or not self.chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID doivent être définis dans .env")

    def _can_send_notification(self, symbol: str) -> bool:
        """Vérifie si on peut envoyer une notification (anti-spam)"""
        if symbol not in self._last_notification:
            return True

        time_since_last = datetime.now() - self._last_notification[symbol]
        return time_since_last > timedelta(minutes=self._cooldown_minutes)

    def send_buy_signal(
        self,
        symbol: str,
        score: int,
        price: float,
        action: str,
        targets: Dict[str, float],
        stop_loss: float,
        reason: str,
        momentum: Optional[float] = None,
        volume_ratio: Optional[float] = None,
        regime: Optional[str] = None,
        estimated_hold_time: Optional[str] = None
    ) -> bool:
        """
        Envoie une notification Telegram pour un signal BUY

        Args:
            symbol: Symbole de la crypto (ex: BTCUSDC)
            score: Score du signal (0-100)
            price: Prix actuel
            action: Action recommandée
            targets: Dict avec tp1, tp2, tp3
            stop_loss: Prix du stop loss
            reason: Raison du signal
            momentum: Score de momentum (optionnel)
            volume_ratio: Ratio de volume (optionnel)
            regime: Régime de marché (optionnel)
            estimated_hold_time: Durée de hold estimée (optionnel)

        Returns:
            True si notification envoyée, False sinon
        """
        # Vérifier anti-spam
        if not self._can_send_notification(symbol):
            logger.info(f"⏸️ Notification ignorée pour {symbol} (cooldown actif)")
            return False

        # Construire le message
        message = self._build_message(
            symbol, score, price, action, targets, stop_loss, reason,
            momentum, volume_ratio, regime, estimated_hold_time
        )

        # Envoyer la notification
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                },
                timeout=10
            )

            if response.status_code == 200:
                self._last_notification[symbol] = datetime.now()
                logger.info(f"✅ Notification Telegram envoyée pour {symbol}")
                return True
            else:
                logger.error(f"❌ Erreur Telegram API: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"❌ Erreur lors de l'envoi Telegram: {e}")
            return False

    def _build_message(
        self,
        symbol: str,
        score: int,
        price: float,
        action: str,
        targets: Dict[str, float],
        stop_loss: float,
        reason: str,
        momentum: Optional[float],
        volume_ratio: Optional[float],
        regime: Optional[str],
        estimated_hold_time: Optional[str]
    ) -> str:
        """Construit le message formaté pour Telegram"""

        # Emoji et titre selon l'action
        action_config = {
            "BUY_NOW": {"emoji": "🟢", "title": "SIGNAL BUY", "score_emojis": True},
            "WAIT_PULLBACK": {"emoji": "🟡", "title": "ATTENDRE BAISSE", "score_emojis": False},
            "WAIT_BREAKOUT": {"emoji": "🔵", "title": "ATTENDRE CASSURE", "score_emojis": False},
            "WAIT_OVERSOLD": {"emoji": "🔵", "title": "ATTENDRE REBOND", "score_emojis": False},
            "WAIT": {"emoji": "⚪", "title": "OBSERVER", "score_emojis": False},
            "SELL_OVERBOUGHT": {"emoji": "🔴", "title": "VENDRE/ÉVITER", "score_emojis": False},
            "AVOID": {"emoji": "⚫", "title": "NE PAS TOUCHER", "score_emojis": False},
        }

        config = action_config.get(action, {"emoji": "⚪", "title": action, "score_emojis": False})

        # Emoji selon le score (seulement pour BUY_NOW) - Ajusté pour /142
        score_emoji = ""
        if config["score_emojis"]:
            if score >= 120:  # 120/142 = 85%
                score_emoji = " 🔥🔥🔥"
            elif score >= 105:  # 105/142 = 74%
                score_emoji = " 🔥🔥"
            elif score >= 90:   # 90/142 = 63%
                score_emoji = " 🔥"

        # Formater le prix intelligemment selon sa valeur
        if price >= 1:
            price_str = f"${price:,.2f}"
        elif price >= 0.01:
            price_str = f"${price:.4f}"
        elif price >= 0.0001:
            price_str = f"${price:.6f}"
        else:
            price_str = f"${price:.8f}"

        # Calculer les gains potentiels en %
        tp1_gain = ((targets.get("tp1", 0) - price) / price) * 100
        tp2_gain = ((targets.get("tp2", 0) - price) / price) * 100
        tp3_gain = ((targets.get("tp3", 0) - price) / price) * 100
        sl_loss = ((stop_loss - price) / price) * 100

        # Formater targets et SL avec même logique
        def format_price(p):
            if p >= 1:
                return f"${p:,.2f}"
            elif p >= 0.01:
                return f"${p:.4f}"
            elif p >= 0.0001:
                return f"${p:.6f}"
            else:
                return f"${p:.8f}"

        message = f"""{config['emoji']} <b>{config['title']}</b>{score_emoji}

<b>{symbol}</b>
📊 Score: <b>{score:.0f}/142</b>{score_emoji}
💰 Prix: <b>{price_str}</b>

<b>🎯 TARGETS:</b>
TP1: {format_price(targets.get('tp1', 0))} (+{tp1_gain:.2f}%)
TP2: {format_price(targets.get('tp2', 0))} (+{tp2_gain:.2f}%)
TP3: {format_price(targets.get('tp3', 0))} (+{tp3_gain:.2f}%)

<b>🛑 STOP LOSS:</b>
SL: {format_price(stop_loss)} ({sl_loss:.2f}%)

<b>📈 ANALYSE:</b>"""

        if momentum is not None:
            message += f"\n• Momentum: {momentum:.0f}/35"

        if volume_ratio is not None:
            message += f"\n• Volume: {volume_ratio:.1f}x"

        if regime:
            message += f"\n• Régime: {regime}"

        if estimated_hold_time:
            message += f"\n• Durée estimée: {estimated_hold_time}"

        message += f"""

💡 <b>Raison:</b>
{reason}

⏰ {datetime.now().strftime('%H:%M:%S')}"""

        return message

    def send_test_notification(self) -> bool:
        """Envoie une notification de test"""
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": "✅ <b>RootTrading Notifications</b>\n\nLe système de notifications Telegram est opérationnel ! 🚀",
                    "parse_mode": "HTML"
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ Erreur test notification: {e}")
            return False


# Instance globale
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Retourne l'instance du notifier (singleton)"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
