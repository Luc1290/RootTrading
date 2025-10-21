"""Service de notifications Telegram pour signaux BUY"""

import builtins
import contextlib
import logging
import os
from datetime import datetime, timedelta, timezone

import requests  # type: ignore[import-untyped]
from dotenv import load_dotenv
from psycopg2.extras import Json

load_dotenv()

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Gère l'envoi de notifications Telegram pour les signaux de trading"""

    def __init__(self, db_connection=None):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.db_connection = db_connection

        # Anti-spam: tracker des dernières notifications par symbole
        self._last_notification: dict[str, dict] = {}  # Stocke {timestamp, action, score}
        self._cooldown_minutes = (
            5  # Cooldown de 5 minutes entre notifications similaires
        )

        if not self.bot_token or not self.chat_id:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID doivent être définis dans .env"
            )

    def _can_send_notification(self, symbol: str, action: str, score: float) -> tuple[bool, str]:  # noqa: PLR0911
        """
        Vérifie si on peut envoyer une notification (anti-spam intelligent)
        Returns:
            tuple[bool, str]: (peut_envoyer, raison)
        """
        # Première notification pour ce symbole
        if symbol not in self._last_notification:
            return True, "Première notification"

        last_notif = self._last_notification[symbol]
        last_time = last_notif["timestamp"]
        last_action = last_notif["action"]
        last_score = last_notif["score"]

        time_since_last = datetime.now(tz=timezone.utc) - last_time
        time_remaining = timedelta(minutes=self._cooldown_minutes) - time_since_last

        # PRIORITÉ 1: Toujours envoyer les EARLY_ENTRY (urgents)
        if action == "EARLY_ENTRY":
            return True, "EARLY_ENTRY toujours prioritaire"

        # PRIORITÉ 2: Changement d'action significatif
        if action != last_action:
            # BUY_DCA → BUY_NOW est une escalade importante
            if last_action == "BUY_DCA" and action == "BUY_NOW":
                return True, f"Escalade {last_action} → {action}"
            # BUY_NOW → BUY_DCA peut indiquer une faiblesse
            if last_action == "BUY_NOW" and action == "BUY_DCA":
                return True, f"Changement {last_action} → {action}"

        # PRIORITÉ 3: Changement de score significatif (>15 points)
        score_diff = abs(score - last_score)
        if score_diff > 15:
            return True, f"Changement de score significatif: {score_diff:.0f} points"

        # PRIORITÉ 4: Cooldown standard si conditions similaires
        if time_since_last > timedelta(minutes=self._cooldown_minutes):
            return True, "Cooldown expiré"

        # Cooldown actif
        minutes_remaining = int(time_remaining.total_seconds() / 60)
        seconds_remaining = int(time_remaining.total_seconds() % 60)
        return False, f"Cooldown actif ({minutes_remaining}m{seconds_remaining}s restantes)"

    def send_buy_signal(
        self,
        symbol: str,
        score: int,
        price: float,
        action: str,
        targets: dict[str, float],
        stop_loss: float,
        reason: str,
        momentum: float | None = None,
        volume_ratio: float | None = None,
        regime: str | None = None,
        estimated_hold_time: str | None = None,
        grade: str | None = None,
        rr_ratio: float | None = None,
        risk_level: str | None = None,
        early_signal: dict | None = None,
    ) -> bool:
        """
        Envoie une notification Telegram pour un signal BUY

        Args:
            symbol: Symbole de la crypto (ex: BTCUSDC)
            score: Score du signal (0-100)
            price: Prix actuel
            action: Action recommandée (BUY_NOW, BUY_DCA, etc.)
            targets: Dict avec tp1, tp2, tp3 (optionnel)
            stop_loss: Prix du stop loss
            reason: Raison du signal
            momentum: Score de momentum (optionnel)
            volume_ratio: Ratio de volume (optionnel)
            regime: Régime de marché (optionnel)
            estimated_hold_time: Durée de hold estimée (optionnel)
            grade: Grade S/A/B/C/D/F (optionnel, système PRO)
            rr_ratio: Ratio Risk/Reward (optionnel, système PRO)
            risk_level: Niveau de risque LOW/MEDIUM/HIGH (optionnel, système PRO)

        Returns:
            True si notification envoyée, False sinon
        """
        # Vérifier anti-spam intelligent
        can_send, reason = self._can_send_notification(symbol, action, score)
        if not can_send:
            logger.info(f"⏸️ {symbol}: {reason}")
            return False

        # Log la raison de l'envoi si c'est une exception au cooldown
        if "prioritaire" in reason or "Changement" in reason or "Escalade" in reason:
            logger.info(f"📬 {symbol}: Envoi autorisé - {reason}")

        # Construire le message
        message = self._build_message(
            symbol,
            score,
            price,
            action,
            targets,
            stop_loss,
            reason,
            momentum,
            volume_ratio,
            regime,
            estimated_hold_time,
            grade,
            rr_ratio,
            risk_level,
            early_signal,
        )

        # Créer les boutons inline pour Binance
        base_asset = symbol.replace("USDC", "").replace("USDT", "").replace("BUSD", "")
        quote_asset = "USDC" if "USDC" in symbol else ("USDT" if "USDT" in symbol else "BUSD")

        # 2 boutons: App (deeplink) et Web (fallback)
        inline_keyboard = {
            "inline_keyboard": [
                [
                    # Bouton 1: Deeplink app mobile (iOS/Android)
                    {
                        "text": "📱 App",
                        "url": f"https://app.binance.com/en/trade/{base_asset}_{quote_asset}"
                    },
                    # Bouton 2: Web desktop/fallback
                    {
                        "text": "🌐 Web",
                        "url": f"https://www.binance.com/en/trade/{base_asset}_{quote_asset}?type=spot"
                    }
                ]
            ]
        }

        # Envoyer la notification
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                    "reply_markup": inline_keyboard,
                },
                timeout=10,
            )

            if response.status_code == 200:
                # Stocker timestamp, action et score pour cooldown intelligent
                self._last_notification[symbol] = {
                    "timestamp": datetime.now(tz=timezone.utc),
                    "action": action,
                    "score": score,
                }
                logger.info(f"✅ Notification Telegram envoyée pour {symbol}")

                # Stocker le signal en DB
                response_data = response.json()
                message_id = response_data.get("result", {}).get("message_id")
                self._store_signal_in_db(
                    symbol=symbol,
                    score=score,
                    price=price,
                    action=action,
                    targets=targets,
                    stop_loss=stop_loss,
                    reason=reason,
                    momentum=momentum,
                    volume_ratio=volume_ratio,
                    regime=regime,
                    estimated_hold_time=estimated_hold_time,
                    grade=grade,
                    rr_ratio=rr_ratio,
                    risk_level=risk_level,
                    message_id=str(message_id) if message_id else None,
                )
            else:
                logger.error(
                    f"❌ Erreur Telegram API: {response.status_code} - {response.text}"
                )
                return False
        except Exception:
            logger.exception("❌ Erreur lors de l'envoi Telegram")
            return False
        else:
            return True

    def _build_message(
        self,
        symbol: str,
        score: int,
        price: float,
        action: str,
        targets: dict[str, float],
        stop_loss: float,
        reason: str,
        momentum: float | None,
        volume_ratio: float | None,
        regime: str | None,
        estimated_hold_time: str | None,
        _grade: str | None,
        _rr_ratio: float | None,
        _risk_level: str | None,
        early_signal: dict | None,
    ) -> str:
        """Construit le message formaté pour Telegram"""

        # Emoji et titre selon l'action (PRO: BUY_DCA renommé en BUY)
        action_config = {
            "BUY_NOW": {"emoji": "🟢", "title": "SIGNAL BUY NOW", "score_emojis": True},
            "BUY_DCA": {"emoji": "🔵", "title": "SIGNAL BUY", "score_emojis": True},
            "EARLY_ENTRY": {
                "emoji": "🟣",
                "title": "⚡ EARLY ENTRY",
                "score_emojis": True,
            },
            "WAIT_PULLBACK": {
                "emoji": "🟡",
                "title": "ATTENDRE BAISSE",
                "score_emojis": False,
            },
            "WAIT_BREAKOUT": {
                "emoji": "🔵",
                "title": "ATTENDRE CASSURE",
                "score_emojis": False,
            },
            "WAIT_OVERSOLD": {
                "emoji": "🔵",
                "title": "ATTENDRE REBOND",
                "score_emojis": False,
            },
            "WAIT": {"emoji": "⚪", "title": "OBSERVER", "score_emojis": False},
            "SELL_OVERBOUGHT": {
                "emoji": "🔴",
                "title": "VENDRE/ÉVITER",
                "score_emojis": False,
            },
            "AVOID": {"emoji": "⚫", "title": "NE PAS TOUCHER", "score_emojis": False},
        }

        config = action_config.get(
            action, {"emoji": "⚪", "title": action, "score_emojis": False}
        )

        # Emoji selon le score (seulement pour BUY_NOW) - Score sur 100
        score_emoji = ""
        if config["score_emojis"]:
            if score >= 85:  # 85/100 = 85%
                score_emoji = " 🔥🔥🔥"
            elif score >= 74:  # 74/100 = 74%
                score_emoji = " 🔥🔥"
            elif score >= 63:  # 63/100 = 63%
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
        tp1_val = targets.get("tp1") or 0
        tp2_val = targets.get("tp2") or 0
        tp3_val = targets.get("tp3") or 0

        tp1_gain = ((tp1_val - price) / price) * 100 if tp1_val else 0
        tp2_gain = ((tp2_val - price) / price) * 100 if tp2_val else 0
        tp3_gain = ((tp3_val - price) / price) * 100 if tp3_val else 0
        sl_loss = ((stop_loss - price) / price) * 100 if stop_loss else 0

        # Formater targets et SL avec même logique
        def format_price(p):
            if p is None or p == 0:
                return "$0.00"
            if p >= 1:
                return f"${p:,.2f}"
            if p >= 0.01:
                return f"${p:.4f}"
            if p >= 0.0001:
                return f"${p:.6f}"
            return f"${p:.8f}"

        message = f"""{config['emoji']} <b>{config['title']}</b>{score_emoji}

<b>{symbol}</b>
📊 Score: <b>{score:.0f}/100</b>{score_emoji}
💰 Prix: <b>{price_str}</b>"""

        # BADGE EARLY ENTRY (NOUVEAU)
        if early_signal and early_signal.get("level") in ["entry_now", "prepare"]:
            early_level = early_signal.get("level", "").upper()
            early_score = early_signal.get("score", 0)
            entry_window = early_signal.get("estimated_entry_window_seconds", 0)

            early_emoji = "🚀" if early_level == "ENTRY_NOW" else "⚡"
            message += f"""

{early_emoji} <b>EARLY ENTRY SIGNAL</b> - {early_level}
   Score Early: <b>{early_score:.0f}/100</b> | Entry window: ~{entry_window}s"""

        message += f"""

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

⏰ {datetime.now(tz=timezone.utc).strftime('%H:%M:%S')}"""

        return message

    def _store_signal_in_db(
        self,
        symbol: str,
        score: int,
        price: float,
        action: str,
        targets: dict[str, float],
        stop_loss: float,
        reason: str,
        momentum: float | None,
        volume_ratio: float | None,
        regime: str | None,
        estimated_hold_time: str | None,
        grade: str | None,
        rr_ratio: float | None,
        risk_level: str | None,
        message_id: str | None,
    ) -> None:
        """Stocke le signal Telegram en base de données"""
        if not self.db_connection:
            logger.warning("Pas de connexion DB, signal non stocké")
            return

        try:
            with self.db_connection.cursor() as cursor:
                # Déterminer le side (BUY pour BUY_NOW, BUY_DCA et EARLY_ENTRY,
                # sinon déduire du contexte)
                side = (
                    "BUY"
                    if action in ["BUY_NOW", "BUY_DCA", "EARLY_ENTRY"]
                    else "SELL" if action in ["SELL_OVERBOUGHT", "AVOID"] else "BUY"
                )

                # Métadonnées additionnelles
                metadata = {
                    "targets": targets,
                    "grade": grade,
                    "rr_ratio": rr_ratio,
                    "risk_level": risk_level,
                }

                insert_query = """
                    INSERT INTO telegram_signals
                    (symbol, side, score, price, action, tp1, tp2, tp3, stop_loss,
                     reason, momentum, volume_ratio, regime, estimated_hold_time,
                     grade, rr_ratio, risk_level, telegram_message_id, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                cursor.execute(
                    insert_query,
                    (
                        symbol,
                        side,
                        score,
                        price,
                        action,
                        targets.get("tp1"),
                        targets.get("tp2"),
                        targets.get("tp3"),
                        stop_loss,
                        reason,
                        momentum,
                        volume_ratio,
                        regime,
                        estimated_hold_time,
                        grade,
                        rr_ratio,
                        risk_level,
                        message_id,
                        Json(metadata),
                    ),
                )

                self.db_connection.commit()
                logger.debug(f"Signal Telegram stocké en DB pour {symbol}")

        except Exception:
            logger.exception("❌ Erreur stockage signal Telegram en DB")
            with contextlib.suppress(builtins.BaseException):
                self.db_connection.rollback()

    def send_test_notification(self) -> bool:
        """Envoie une notification de test"""
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": "✅ <b>RootTrading Notifications</b>\n\nLe système de notifications Telegram est opérationnel ! 🚀",
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
        except Exception:
            logger.exception("❌ Erreur test notification")
            return False
        else:
            return response.status_code == 200


class _NotifierSingleton:
    """Singleton pour gérer l'instance unique du TelegramNotifier"""
    _instance: TelegramNotifier | None = None

    @classmethod
    def get_instance(cls, db_connection=None) -> TelegramNotifier:
        """
        Retourne l'instance du notifier (singleton).

        Args:
            db_connection: Connexion PostgreSQL (psycopg2) pour stocker les signaux.
                          Si fournie, met à jour la connexion de l'instance existante.
        """
        if cls._instance is None:
            cls._instance = TelegramNotifier(db_connection=db_connection)
        elif db_connection is not None:
            # Mettre à jour la connexion DB si fournie
            cls._instance.db_connection = db_connection
        return cls._instance


def get_notifier(db_connection=None) -> TelegramNotifier:
    """
    Retourne l'instance du notifier (singleton).

    Args:
        db_connection: Connexion PostgreSQL (psycopg2) pour stocker les signaux.
    """
    return _NotifierSingleton.get_instance(db_connection)
