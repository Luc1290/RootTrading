"""
Coordinator simplifié pour RootTrading.
Rôle : Valider la faisabilité des signaux et les transmettre au trader.
"""
import logging
import time
import json
import asyncio
import threading
import os
from typing import Dict, Any, Optional
import psycopg2

from shared.src.redis_client import RedisClient
from shared.src.enums import OrderSide
from shared.src.schemas import StrategySignal
from service_client import ServiceClient
from trailing_sell_manager import TrailingSellManager

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Coordinateur simplifié : reçoit les signaux, vérifie la faisabilité, transmet au trader.
    """
    
    def __init__(self, trader_api_url: str = "http://trader:5002", 
                 portfolio_api_url: str = "http://portfolio:8000"):
        """
        Initialise le coordinator.
        
        Args:
            trader_api_url: URL du service trader
            portfolio_api_url: URL du service portfolio
        """
        self.service_client = ServiceClient(trader_api_url, portfolio_api_url)
        self.redis_client = RedisClient()
        
        # Configuration base de données
        self.db_config = {
            'host': os.getenv('DB_HOST', 'db'),
            'port': os.getenv('DB_PORT', 5432),
            'database': os.getenv('DB_NAME', 'trading'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
        self.db_connection = None
        
        # Thread de monitoring stop-loss
        self.stop_loss_active = True
        self.stop_loss_thread: Optional[threading.Thread] = None
        
        # Configuration dynamique basée sur le capital total
        self.fee_rate = 0.001  # 0.1% de frais estimés par trade
        
        # Allocation dynamique intelligente
        self.base_allocation_percent = 8.0  # 8% par défaut du capital total
        self.max_trade_percent = 15.0  # 15% maximum du capital total
        self.min_absolute_trade_usdc = 10.0  # 10 USDC minimum Binance
        
        # Initialiser la connexion DB d'abord
        self._init_db_connection()
        
        # Initialiser le gestionnaire de trailing sell avec la connexion DB
        self.trailing_manager = TrailingSellManager(
            redis_client=self.redis_client,
            service_client=self.service_client,
            db_connection=self.db_connection
        )
        
        # Configuration stop-loss automatique adaptatif
        self.stop_loss_percent_base = 0.015  # 1.5% de base
        self.stop_loss_percent_bullish = 0.025  # 2.5% en tendance haussière (plus souple)
        self.stop_loss_percent_strong_bullish = 0.035  # 3.5% en tendance très haussière (encore plus souple)
        self.price_check_interval = 60  # Vérification des prix toutes les 60 secondes (aligné sur la fréquence des données)
        
        # Démarrer le monitoring stop-loss
        self.start_stop_loss_monitoring()
        
        logger.info("✅ Coordinator initialisé (version simplifiée) avec stop-loss automatique")
        
        # Stats
        self.stats = {
            "signals_received": 0,
            "signals_processed": 0,
            "signals_rejected": 0,
            "orders_sent": 0,
            "errors": 0
        }
        
    def _init_db_connection(self):
        """Initialise la connexion à la base de données."""
        try:
            self.db_connection = psycopg2.connect(**self.db_config)
            logger.info("Connexion à la base de données établie")
        except Exception as e:
            logger.error(f"Erreur connexion DB: {e}")
            self.db_connection = None
            
    def _mark_signal_as_processed(self, signal_id: int) -> bool:
        """
        Marque un signal comme traité en base de données.
        
        Args:
            signal_id: ID du signal à marquer
            
        Returns:
            True si le marquage a réussi, False sinon
        """
        if not self.db_connection:
            logger.warning("Pas de connexion DB pour marquer le signal")
            return False
            
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE trading_signals SET processed = true WHERE id = %s",
                    (signal_id,)
                )
                
                self.db_connection.commit()
                
                if cursor.rowcount > 0:
                    logger.debug(f"Signal {signal_id} marqué comme traité")
                    return True
                else:
                    logger.warning(f"Signal {signal_id} non trouvé pour marquage")
                    return False
                    
        except Exception as e:
            logger.error(f"Erreur marquage signal {signal_id}: {e}")
            try:
                self.db_connection.rollback()
            except Exception as rollback_error:
                logger.error(f"Erreur rollback: {rollback_error}")
            return False
    
    def process_signal(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Traite un signal reçu via Redis.
        
        Args:
            channel: Canal Redis
            data: Données du signal
        """
        try:
            self.stats["signals_received"] += 1
            
            # Parser le signal
            try:
                if 'side' in data and isinstance(data['side'], str):
                    data['side'] = OrderSide(data['side'])
                    
                signal = StrategySignal(**data)
            except ValueError as e:
                logger.error(f"❌ Erreur parsing signal: {e}")
                self.stats["signals_rejected"] += 1
                return
            except Exception as e:
                logger.error(f"❌ Erreur création signal: {e}")
                self.stats["signals_rejected"] += 1
                return
            logger.info(f"📨 Signal reçu: {signal.strategy} {signal.side} {signal.symbol} @ {signal.price}")
            logger.debug(f"Signal metadata: {signal.metadata}")
            if signal.metadata and 'db_id' in signal.metadata:
                logger.info(f"DB ID trouvé dans signal: {signal.metadata['db_id']}")
            else:
                logger.warning("Pas de db_id trouvé dans les métadonnées du signal")
            
            # Vérifier la faisabilité
            is_feasible, reason = self._check_feasibility(signal)
            
            if not is_feasible:
                logger.warning(f"❌ Signal rejeté: {reason}")
                self.stats["signals_rejected"] += 1
                
                # Marquer le signal comme traité même s'il est rejeté
                if signal.metadata and 'db_id' in signal.metadata:
                    db_id = signal.metadata['db_id']
                    self._mark_signal_as_processed(db_id)
                
                return
            
            # Vérifier l'efficacité du trade (logique simplifiée)
            is_efficient, efficiency_reason = self._check_trade_efficiency(signal)
            
            if not is_efficient:
                logger.warning(f"❌ Signal rejeté: {efficiency_reason}")
                self.stats["signals_rejected"] += 1
                
                # Marquer le signal comme traité même s'il est rejeté
                if signal.metadata and 'db_id' in signal.metadata:
                    db_id = signal.metadata['db_id']
                    self._mark_signal_as_processed(db_id)
                
                return
            
            # Calculer la quantité à trader
            quantity = self._calculate_quantity(signal)
            if not quantity or quantity <= 0:
                logger.error("Impossible de calculer la quantité")
                self.stats["signals_rejected"] += 1
                
                # Marquer le signal comme traité même en cas d'erreur
                if signal.metadata and 'db_id' in signal.metadata:
                    db_id = signal.metadata['db_id']
                    self._mark_signal_as_processed(db_id)
                
                return
            
            # Préparer l'ordre pour le trader (MARKET pour exécution immédiate)
            side_value = signal.side.value if hasattr(signal.side, 'value') else str(signal.side)
            order_data = {
                "symbol": signal.symbol,
                "side": side_value,
                "quantity": float(quantity),
                "price": None,  # Force ordre MARKET pour exécution immédiate
                "strategy": signal.strategy,
                "timestamp": int(time.time() * 1000),
                "metadata": signal.metadata or {}
            }
            
            # Ajouter les stops si disponibles
            if signal.metadata:
                if "stop_price" in signal.metadata:
                    order_data["stop_price"] = signal.metadata["stop_price"]
                if "trailing_delta" in signal.metadata:
                    order_data["trailing_delta"] = signal.metadata["trailing_delta"]
            
            # Envoyer l'ordre au trader
            logger.info(f"📤 Envoi ordre au trader: {order_data['side']} {quantity:.8f} {signal.symbol}")
            order_id = self.service_client.create_order(order_data)
            
            if order_id:
                logger.info(f"✅ Ordre créé: {order_id}")
                self.stats["orders_sent"] += 1
                self.stats["signals_processed"] += 1
                
                # Marquer le signal comme traité en DB si on a le db_id
                if signal.metadata and 'db_id' in signal.metadata:
                    db_id = signal.metadata['db_id']
                    if self._mark_signal_as_processed(db_id):
                        logger.debug(f"Signal {db_id} marqué comme traité en DB")
                    else:
                        logger.warning(f"Impossible de marquer le signal {db_id} comme traité")
                else:
                    logger.warning("Pas de db_id dans les métadonnées du signal")
            else:
                logger.error("❌ Échec création ordre")
                self.stats["errors"] += 1
                
                # Marquer le signal comme traité même en cas d'échec
                if signal.metadata and 'db_id' in signal.metadata:
                    db_id = signal.metadata['db_id']
                    self._mark_signal_as_processed(db_id)
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal: {str(e)}")
            self.stats["errors"] += 1
    
    def _check_feasibility(self, signal: StrategySignal) -> tuple[bool, str]:
        """
        Vérifie si un trade est faisable.
        
        Args:
            signal: Signal à vérifier
            
        Returns:
            (is_feasible, reason)
        """
        try:
            # Récupérer les balances
            balances = self.service_client.get_all_balances()
            if not balances:
                return False, "Impossible de récupérer les balances"
            
            # Extraire l'asset de base (quote toujours USDC)
            base_asset = self._get_base_asset(signal.symbol)
            
            if signal.side == OrderSide.BUY:
                # Pour un BUY, on a besoin d'USDC
                if isinstance(balances, dict):
                    usdc_balance = balances.get('USDC', {}).get('free', 0)
                else:
                    usdc_balance = next((b.get('free', 0) for b in balances if b.get('asset') == 'USDC'), 0)
                
                if usdc_balance < self.min_absolute_trade_usdc:
                    return False, f"Balance USDC insuffisante: {usdc_balance:.2f} < {self.min_absolute_trade_usdc}"
                
                # Vérifier s'il y a déjà un cycle actif pour ce symbole
                active_cycle = self._check_active_cycle(signal.symbol)
                if active_cycle:
                    return False, f"Cycle déjà actif pour {signal.symbol}: {active_cycle}"
                
            else:  # SELL
                # FILTRE TRAILING SELL: Vérifier si on doit vendre maintenant (AVANT balance)
                # Récupérer les infos de la position active pour le trailing sell
                active_positions = self.service_client.get_active_cycles(signal.symbol)
                if active_positions:
                    position = active_positions[0]
                    entry_price = float(position.get('entry_price', 0))
                    entry_time = position.get('timestamp')
                    
                    should_sell, sell_reason = self.trailing_manager.check_trailing_sell(
                        symbol=signal.symbol,
                        current_price=signal.price,
                        entry_price=entry_price,
                        entry_time=entry_time
                    )
                    if not should_sell:
                        return False, sell_reason
                else:
                    # Pas de position active, autoriser le SELL
                    logger.info(f"✅ Pas de position active pour {signal.symbol}, SELL autorisé")
                
                # Pour un SELL, on a besoin de la crypto
                if isinstance(balances, dict):
                    crypto_balance = balances.get(base_asset, {}).get('free', 0)
                else:
                    crypto_balance = next((b.get('free', 0) for b in balances if b.get('asset') == base_asset), 0)
                
                if crypto_balance <= 0:
                    return False, f"Pas de {base_asset} à vendre"
                
                # Vérifier la valeur en USDC
                value_usdc = crypto_balance * signal.price
                if value_usdc < self.min_absolute_trade_usdc:
                    return False, f"Valeur trop faible: {value_usdc:.2f} USDC"
            
            return True, "OK"
            
        except Exception as e:
            logger.error(f"Erreur vérification faisabilité: {str(e)}")
            return False, f"Erreur: {str(e)}"
    
    def _check_trade_efficiency(self, signal: StrategySignal) -> tuple[bool, str]:
        """
        Vérifications basiques pour l'exécution du trade.
        Le Coordinator EXÉCUTE, il ne décide pas de la stratégie.
        
        Args:
            signal: Signal à analyser
            
        Returns:
            (is_efficient, reason)
        """
        try:
            # Calculer la quantité et valeur du trade
            quantity = self._calculate_quantity(signal)
            if not quantity:
                return False, "Impossible de calculer la quantité"
            
            # Valeur totale du trade
            trade_value = quantity * signal.price
            
            # Filtre 1: Valeur minimum du trade (simple)
            if trade_value < self.min_absolute_trade_usdc:
                return False, f"Trade trop petit: {trade_value:.2f} USDC < {self.min_absolute_trade_usdc:.2f} USDC (minimum Binance)"
            
            # Filtre 2: Ratio frais/valeur acceptable
            estimated_fees = trade_value * self.fee_rate * 2  # Aller-retour
            fee_percentage = (estimated_fees / trade_value) * 100
            
            if fee_percentage > 1.0:  # Si frais > 1% de la valeur du trade
                return False, f"Frais trop élevés: {fee_percentage:.2f}% de la valeur du trade"
            
            logger.info(f"✅ Trade valide: {trade_value:.2f} USDC, frais {fee_percentage:.2f}%")
            return True, "Trade valide"
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification trade: {str(e)}")
            return True, "Erreur technique - trade autorisé par défaut"
    
    def _calculate_quantity(self, signal: StrategySignal) -> Optional[float]:
        """
        Calcule la quantité à trader.
        
        Args:
            signal: Signal de trading
            
        Returns:
            Quantité à trader ou None
        """
        try:
            balances = self.service_client.get_all_balances()
            if not balances:
                return None
            
            base_asset = self._get_base_asset(signal.symbol)
            
            if signal.side == OrderSide.BUY:
                # Pour un BUY, calculer combien on peut acheter (allocation dynamique)
                if isinstance(balances, dict):
                    usdc_balance = balances.get('USDC', {}).get('free', 0)
                    total_capital = sum(b.get('value_usdc', 0) for b in balances.values())
                else:
                    usdc_balance = next((b.get('free', 0) for b in balances if b.get('asset') == 'USDC'), 0)
                    total_capital = sum(b.get('value_usdc', 0) for b in balances)
                
                # Allocation dynamique basée sur la force du signal
                allocation_percent = self.base_allocation_percent  # 8% par défaut
                
                # Ajuster selon la force du signal si disponible
                if signal.metadata and "signal_strength" in signal.metadata:
                    strength = signal.metadata["signal_strength"]
                    if strength == "VERY_STRONG":
                        allocation_percent = 12.0
                    elif strength == "STRONG":
                        allocation_percent = 10.0
                    elif strength == "MODERATE":
                        allocation_percent = 8.0
                    elif strength == "WEAK":
                        allocation_percent = 5.0
                
                # Calculer le montant à trader (% du capital total)
                trade_amount = total_capital * (allocation_percent / 100)
                
                # Limiter par l'USDC disponible (utiliser jusqu'à 98% pour garder une petite marge)
                trade_amount = min(trade_amount, usdc_balance * 0.98)
                
                # Appliquer seulement le maximum (pas de minimum % car on veut pouvoir épuiser l'USDC)
                max_trade_value = total_capital * (self.max_trade_percent / 100)
                trade_amount = min(trade_amount, max_trade_value)
                
                # Mais toujours respecter le minimum absolu Binance
                trade_amount = max(self.min_absolute_trade_usdc, trade_amount)
                
                # Convertir en quantité
                quantity = trade_amount / signal.price
                
            else:  # SELL
                # Pour un SELL, vendre toute la position
                if isinstance(balances, dict):
                    quantity = balances.get(base_asset, {}).get('free', 0)
                else:
                    quantity = next((b.get('free', 0) for b in balances if b.get('asset') == base_asset), 0)
            
            return quantity
            
        except Exception as e:
            logger.error(f"Erreur calcul quantité: {str(e)}")
            return None
    
    def _get_base_asset(self, symbol: str) -> str:
        """Extrait l'asset de base du symbole."""
        if symbol.endswith('USDC'):
            return symbol[:-4]
        elif symbol.endswith('USDT'):
            return symbol[:-4]
        elif symbol.endswith('BTC'):
            return symbol[:-3]
        elif symbol.endswith('ETH'):
            return symbol[:-3]
        else:
            return symbol[:-4]  # Par défaut
    
    def _get_quote_asset(self, symbol: str) -> str:
        """Extrait l'asset de quote du symbole."""
        if symbol.endswith('USDC'):
            return 'USDC'
        elif symbol.endswith('USDT'):
            return 'USDT'
        elif symbol.endswith('BTC'):
            return 'BTC'
        elif symbol.endswith('ETH'):
            return 'ETH'
        else:
            return 'USDC'  # Par défaut
    
    def _check_active_cycle(self, symbol: str) -> Optional[str]:
        """
        Vérifie s'il y a un cycle actif pour ce symbole via le portfolio service.
        
        Args:
            symbol: Symbole à vérifier (ex: 'BTCUSDC')
            
        Returns:
            ID du cycle actif si trouvé, None sinon
        """
        try:
            # Récupérer les positions actives depuis le portfolio service
            active_positions = self.service_client.get_active_cycles(symbol)
            
            if active_positions:
                # Retourner l'ID de la première position active trouvée
                position = active_positions[0]
                return position.get('id', f"position_{symbol}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Erreur vérification cycle actif pour {symbol}: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du coordinator."""
        return self.stats.copy()
    
    def start_stop_loss_monitoring(self) -> None:
        """Démarre le thread de monitoring stop-loss."""
        if not self.stop_loss_thread or not self.stop_loss_thread.is_alive():
            self.stop_loss_thread = threading.Thread(
                target=self._stop_loss_monitor_loop,
                daemon=True,
                name="StopLossMonitor"
            )
            self.stop_loss_thread.start()
            logger.info("🛡️ Monitoring stop-loss démarré")
    
    def stop_stop_loss_monitoring(self) -> None:
        """Arrête le monitoring stop-loss."""
        self.stop_loss_active = False
        if self.stop_loss_thread:
            self.stop_loss_thread.join(timeout=10)
        logger.info("🛑 Monitoring stop-loss arrêté")
    
    def _stop_loss_monitor_loop(self) -> None:
        """Boucle principale du monitoring stop-loss."""
        logger.info("🔍 Boucle de monitoring stop-loss active")
        
        while self.stop_loss_active:
            try:
                self._check_all_positions_stop_loss()
                time.sleep(self.price_check_interval)
            except Exception as e:
                logger.error(f"❌ Erreur dans monitoring stop-loss: {e}")
                time.sleep(self.price_check_interval * 2)  # Attendre plus longtemps en cas d'erreur
    
    def _check_all_positions_stop_loss(self) -> None:
        """Vérifie toutes les positions actives pour déclenchement stop-loss."""
        try:
            # Récupérer toutes les positions actives
            all_active_cycles = self.service_client.get_all_active_cycles()
            
            if not all_active_cycles:
                return
            
            logger.debug(f"🔍 Vérification stop-loss pour {len(all_active_cycles)} positions actives")
            
            for cycle in all_active_cycles:
                try:
                    self._check_position_stop_loss(cycle)
                except Exception as e:
                    logger.error(f"❌ Erreur vérification stop-loss pour cycle {cycle.get('id', 'unknown')}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Erreur récupération positions actives: {e}")
    
    def _check_position_stop_loss(self, cycle: Dict[str, Any]) -> None:
        """
        Vérifie une position spécifique et déclenche un stop-loss si nécessaire.
        Met aussi à jour la référence trailing automatiquement.
        Utilise maintenant le système de stop-loss adaptatif intelligent.
        
        Args:
            cycle: Données du cycle de trading
        """
        try:
            symbol = cycle.get('symbol')
            entry_price = float(cycle.get('entry_price', 0))
            entry_time = cycle.get('timestamp')
            cycle_id = cycle.get('id')
            
            if not symbol or not entry_price:
                logger.warning(f"⚠️ Données cycle incomplètes: {cycle}")
                return
            
            # Convertir timestamp en epoch si nécessaire
            if isinstance(entry_time, str):
                from datetime import datetime
                entry_time_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                entry_time_epoch = entry_time_dt.timestamp()
            else:
                entry_time_epoch = float(entry_time) if entry_time else time.time()
            
            # Récupérer le prix actuel via TrailingSellManager
            current_price = self.trailing_manager.get_current_price(symbol)
            if not current_price:
                logger.warning(f"⚠️ Prix actuel indisponible pour {symbol}")
                return
            
            # Utiliser le TrailingSellManager pour vérifier si on doit vendre
            should_sell, sell_reason = self.trailing_manager.check_trailing_sell(
                symbol=symbol,
                current_price=current_price,
                entry_price=entry_price,
                entry_time=entry_time_epoch
            )
            
            if should_sell:
                logger.warning(f"🚨 AUTO-SELL DÉCLENCHÉ pour {symbol}: {sell_reason}")
                # Déclencher vente d'urgence
                self._execute_emergency_sell(symbol, current_price, str(cycle_id) if cycle_id else "unknown", sell_reason)
                return
            
            # Mettre à jour le prix max si position gagnante
            if current_price > entry_price:
                self.trailing_manager.update_max_price_if_needed(symbol, current_price)
            
            # Log debug pour positions proches des seuils
            loss_percent = (entry_price - current_price) / entry_price * 100
            # Utiliser le seuil adaptatif pour l'alerte précoce
            base_threshold = self.stop_loss_percent_base * 100 * 0.5
            if loss_percent > base_threshold:  # Si > 50% du seuil de base
                precision = self._get_price_precision(current_price)
                logger.debug(f"⚠️ {symbol} proche stop-loss: {current_price:.{precision}f} (perte {loss_percent:.2f}%)")
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification stop-loss cycle {cycle_id}: {e}")
    
    
    
    
    
    
    
    
    
    
    
    def _get_price_precision(self, price: float) -> int:
        """
        Détermine la précision d'affichage selon le niveau de prix.
        
        Args:
            price: Prix à analyser
            
        Returns:
            Nombre de décimales à afficher
        """
        if price >= 1000:  # BTC, ETH haut
            return 2  # 17000.12
        elif price >= 100:  # ETH, SOL, AVAX
            return 3  # 177.123
        elif price >= 1:   # ADA, XRP, LINK
            return 6  # 3.1234
        elif price >= 0.01:  # Certains altcoins
            return 6  # 0.12345
        elif price >= 0.0001:  # DOGE, SHIB
            return 10  # 0.123456
        else:  # PEPE, BONK (très petits prix)
            return 12  # 0.12345678
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du coordinator."""
        return self.stats.copy()
    
    def start_stop_loss_monitoring(self) -> None:
        """Démarre le thread de monitoring stop-loss."""
        if not self.stop_loss_thread or not self.stop_loss_thread.is_alive():
            self.stop_loss_thread = threading.Thread(
                target=self._stop_loss_monitor_loop,
                daemon=True,
                name="StopLossMonitor"
            )
            self.stop_loss_thread.start()
            logger.info("🛡️ Monitoring stop-loss démarré")
    
    def stop_stop_loss_monitoring(self) -> None:
        """Arrête le monitoring stop-loss."""
        self.stop_loss_active = False
        if self.stop_loss_thread:
            self.stop_loss_thread.join(timeout=10)
        logger.info("🛑 Monitoring stop-loss arrêté")
    
    def _stop_loss_monitor_loop(self) -> None:
        """Boucle principale du monitoring stop-loss."""
        logger.info("🔍 Boucle de monitoring stop-loss active")
        
        while self.stop_loss_active:
            try:
                self._check_all_positions_stop_loss()
                time.sleep(self.price_check_interval)
            except Exception as e:
                logger.error(f"❌ Erreur dans monitoring stop-loss: {e}")
                time.sleep(self.price_check_interval * 2)  # Attendre plus longtemps en cas d'erreur
    
    def _check_all_positions_stop_loss(self) -> None:
        """Vérifie toutes les positions actives pour déclenchement stop-loss."""
        try:
            # Récupérer toutes les positions actives
            all_active_cycles = self.service_client.get_all_active_cycles()
            
            if not all_active_cycles:
                return
            
            logger.debug(f"🔍 Vérification stop-loss pour {len(all_active_cycles)} positions actives")
            
            for cycle in all_active_cycles:
                try:
                    self._check_position_stop_loss(cycle)
                except Exception as e:
                    logger.error(f"❌ Erreur vérification stop-loss pour cycle {cycle.get('id', 'unknown')}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Erreur récupération positions actives: {e}")
    
    def _check_position_stop_loss(self, cycle: Dict[str, Any]) -> None:
        """
        Vérifie une position spécifique et déclenche un stop-loss si nécessaire.
        Met aussi à jour la référence trailing automatiquement.
        Utilise maintenant le système de stop-loss adaptatif intelligent.
        
        Args:
            cycle: Données du cycle de trading
        """
        try:
            symbol = cycle.get('symbol')
            entry_price = float(cycle.get('entry_price', 0))
            entry_time = cycle.get('timestamp')
            cycle_id = cycle.get('id')
            
            if not symbol or not entry_price:
                logger.warning(f"⚠️ Données cycle incomplètes: {cycle}")
                return
            
            # Convertir timestamp en epoch si nécessaire
            if isinstance(entry_time, str):
                from datetime import datetime
                entry_time_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                entry_time_epoch = entry_time_dt.timestamp()
            else:
                entry_time_epoch = float(entry_time) if entry_time else time.time()
            
            # Récupérer le prix actuel via TrailingSellManager
            current_price = self.trailing_manager.get_current_price(symbol)
            if not current_price:
                logger.warning(f"⚠️ Prix actuel indisponible pour {symbol}")
                return
            
            # Utiliser le TrailingSellManager pour vérifier si on doit vendre
            should_sell, sell_reason = self.trailing_manager.check_trailing_sell(
                symbol=symbol,
                current_price=current_price,
                entry_price=entry_price,
                entry_time=entry_time_epoch
            )
            
            if should_sell:
                logger.warning(f"🚨 AUTO-SELL DÉCLENCHÉ pour {symbol}: {sell_reason}")
                # Déclencher vente d'urgence
                self._execute_emergency_sell(symbol, current_price, str(cycle_id) if cycle_id else "unknown", sell_reason)
                return
            
            # Mettre à jour le prix max si position gagnante
            if current_price > entry_price:
                self.trailing_manager.update_max_price_if_needed(symbol, current_price)
            
            # Log debug pour positions proches des seuils
            loss_percent = (entry_price - current_price) / entry_price * 100
            # Utiliser le seuil adaptatif pour l'alerte précoce
            base_threshold = self.stop_loss_percent_base * 100 * 0.5
            if loss_percent > base_threshold:  # Si > 50% du seuil de base
                precision = self._get_price_precision(current_price)
                logger.debug(f"⚠️ {symbol} proche stop-loss: {current_price:.{precision}f} (perte {loss_percent:.2f}%)")
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification stop-loss cycle {cycle_id}: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix actuel d'un symbole depuis Redis.
        
        Args:
            symbol: Symbole à récupérer
            
        Returns:
            Prix actuel ou None
        """
        try:
            # Essayer de récupérer depuis Redis (ticker)
            ticker_key = f"ticker:{symbol}"
            ticker_data = self.redis_client.get(ticker_key)
            
            if ticker_data:
                if isinstance(ticker_data, dict):
                    return float(ticker_data.get('price', 0))
                elif isinstance(ticker_data, str):
                    ticker_dict = json.loads(ticker_data)
                    return float(ticker_dict.get('price', 0))
            
            # Fallback: essayer market_data
            market_key = f"market_data:{symbol}:1m"
            market_data = self.redis_client.get(market_key)
            
            if market_data:
                if isinstance(market_data, dict):
                    return float(market_data.get('close', 0))
                elif isinstance(market_data, str):
                    market_dict = json.loads(market_data)
                    return float(market_dict.get('close', 0))
            
            logger.warning(f"⚠️ Prix non trouvé dans Redis pour {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération prix pour {symbol}: {e}")
            return None
    
    def _execute_emergency_sell(self, symbol: str, current_price: float, cycle_id: str, reason: str) -> None:
        """
        Exécute une vente d'urgence (stop-loss).
        
        Args:
            symbol: Symbole à vendre
            current_price: Prix actuel
            cycle_id: ID du cycle
            reason: Raison de la vente
        """
        try:
            logger.warning(f"🚨 VENTE D'URGENCE {symbol} - Raison: {reason}")
            
            # Récupérer la balance à vendre
            balances = self.service_client.get_all_balances()
            if not balances:
                logger.error(f"❌ Impossible de récupérer les balances pour vente d'urgence {symbol}")
                return
            
            base_asset = self._get_base_asset(symbol)
            
            if isinstance(balances, dict):
                quantity = balances.get(base_asset, {}).get('free', 0)
            else:
                quantity = next((b.get('free', 0) for b in balances if b.get('asset') == base_asset), 0)
            
            if quantity <= 0:
                logger.warning(f"⚠️ Aucune quantité à vendre pour {symbol} (balance: {quantity})")
                return
            
            # Créer ordre de vente d'urgence
            order_data = {
                "symbol": symbol,
                "side": "SELL",
                "quantity": float(quantity),
                "price": None,  # Ordre MARKET pour exécution immédiate
                "strategy": f"STOP_LOSS_AUTO_{reason}",
                "timestamp": int(time.time() * 1000),
                "metadata": {
                    "emergency_sell": True,
                    "stop_loss_trigger": True,
                    "cycle_id": cycle_id,
                    "trigger_price": current_price,
                    "reason": reason
                }
            }
            
            # Envoyer l'ordre
            logger.warning(f"📤 Envoi ordre stop-loss: SELL {quantity:.8f} {symbol} @ MARKET")
            order_id = self.service_client.create_order(order_data)
            
            if order_id:
                logger.warning(f"✅ Ordre stop-loss créé: {order_id}")
                self.stats["orders_sent"] += 1
                
                # Nettoyer les références Redis liées à ce symbole
                self.trailing_manager._clear_sell_reference(symbol)
                
            else:
                logger.error(f"❌ Échec création ordre stop-loss pour {symbol}")
                self.stats["errors"] += 1
                
        except Exception as e:
            logger.error(f"❌ Erreur vente d'urgence {symbol}: {e}")
            self.stats["errors"] += 1