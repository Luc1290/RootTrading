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
        
        # Configuration trailing sell
        self.sell_margin = 0.004  # 0.4% de marge pour laisser plus de marge aux pumps
        
        # Initialiser la connexion DB
        self._init_db_connection()
        
        # Configuration stop-loss automatique adaptatif
        self.stop_loss_percent_base = 0.015  # 1.5% de base
        self.stop_loss_percent_bullish = 0.025  # 2.5% en tendance haussière (plus souple)
        self.stop_loss_percent_strong_bullish = 0.035  # 3.5% en tendance très haussière (encore plus souple)
        self.price_check_interval = 60  # Vérification des prix toutes les 60 secondes (aligné sur la fréquence des données)
        
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
        
        # Démarrer le monitoring stop-loss
        self.start_stop_loss_monitoring()
        
        logger.info("✅ Coordinator initialisé (version simplifiée) avec stop-loss automatique")
    
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
                should_sell, sell_reason = self._check_trailing_sell(signal)
                if not should_sell:
                    return False, sell_reason
                
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
    
    
    def _check_trailing_sell(self, signal: StrategySignal) -> tuple[bool, str]:
        """
        Vérifie si on doit exécuter le SELL selon la logique de trailing sell intelligent.
        Utilise maintenant un stop-loss adaptatif basé sur les données d'analyse de marché.
        
        Args:
            signal: Signal SELL à vérifier
            
        Returns:
            (should_sell, reason)
        """
        logger.info(f"🔍 DEBUT _check_trailing_sell pour {signal.symbol} @ {signal.price}")
        try:
            # Récupérer la position active pour vérifier si elle est gagnante
            logger.info(f"🔍 Appel get_active_cycles pour {signal.symbol}")
            active_positions = self.service_client.get_active_cycles(signal.symbol)
            logger.info(f"🔍 Résultat get_active_cycles: {active_positions}")
            
            if not active_positions:
                # Pas de position active, autoriser le SELL
                logger.info(f"✅ Pas de position active pour {signal.symbol}, SELL autorisé")
                return True, "Pas de position active, SELL autorisé"
            
            logger.info(f"🔍 Position active trouvée: {active_positions[0]}")
            position = active_positions[0]
            entry_price = float(position.get('entry_price', 0))
            entry_time = position.get('timestamp')  # Format ISO string
            current_price = signal.price
            
            # Convertir timestamp ISO en epoch si nécessaire
            if isinstance(entry_time, str):
                from datetime import datetime
                entry_time_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                entry_time_epoch = entry_time_dt.timestamp()
            else:
                entry_time_epoch = float(entry_time) if entry_time else time.time()
            
            precision = self._get_price_precision(current_price)
            logger.info(f"🔍 Prix entrée: {entry_price:.{precision}f}, Prix actuel: {current_price:.{precision}f}")
            
            # Calculer la perte actuelle en pourcentage
            loss_percent = (entry_price - current_price) / entry_price
            
            # === NOUVEAU: STOP-LOSS ADAPTATIF INTELLIGENT ===
            adaptive_threshold = self._calculate_adaptive_threshold(signal.symbol, entry_price, entry_time_epoch)
            
            logger.info(f"🧠 Stop-loss adaptatif pour {signal.symbol}: {adaptive_threshold*100:.2f}% (perte actuelle: {loss_percent*100:.2f}%)")
            
            # Si perte dépasse le seuil adaptatif : SELL immédiat
            if loss_percent >= adaptive_threshold:
                logger.info(f"📉 Stop-loss adaptatif déclenché pour {signal.symbol}: perte {loss_percent*100:.2f}% ≥ seuil {adaptive_threshold*100:.2f}%")
                return True, f"Stop-loss adaptatif déclenché (perte {loss_percent*100:.2f}% ≥ {adaptive_threshold*100:.2f}%)"
            
            # Si position perdante mais dans la tolérance adaptative : garder
            if current_price <= entry_price:
                logger.info(f"🟡 Position perdante mais dans tolérance adaptative pour {signal.symbol}: perte {loss_percent*100:.2f}% < seuil {adaptive_threshold*100:.2f}%")
                return False, f"Position perdante mais dans tolérance adaptative (perte {loss_percent*100:.2f}% < {adaptive_threshold*100:.2f}%)"
            
            logger.info(f"🔍 Position gagnante détectée, vérification trailing sell")
            # === LOGIQUE TRAILING SELL INCHANGÉE POUR POSITIONS GAGNANTES ===
            previous_sell_price = self._get_previous_sell_price(signal.symbol)
            logger.info(f"🔍 Prix SELL précédent: {previous_sell_price}")
            
            if previous_sell_price is None:
                # Premier SELL gagnant : stocker comme référence, ne pas vendre
                logger.info(f"🔍 Premier SELL gagnant, stockage référence")
                self._update_sell_reference(signal.symbol, current_price)
                logger.info(f"🎯 Premier SELL gagnant pour {signal.symbol} @ {current_price:.{precision}f}, stocké comme référence")
                return False, f"Position gagnante, premier SELL @ {current_price:.{precision}f} stocké comme référence"
            
            # Comparer avec le SELL précédent (avec marge de tolérance)
            sell_threshold = previous_sell_price * (1 - self.sell_margin)
            logger.info(f"🔍 Seuil de vente calculé: {sell_threshold:.{precision}f} (marge {self.sell_margin*100:.1f}%)")
            
            if current_price > previous_sell_price:
                # Prix monte : mettre à jour référence, ne pas vendre
                logger.info(f"🔍 Prix monte, mise à jour référence")
                self._update_sell_reference(signal.symbol, current_price)
                logger.info(f"📈 Prix monte pour {signal.symbol}: {current_price:.{precision}f} > {previous_sell_price:.{precision}f}, référence mise à jour")
                return False, f"Prix monte ({current_price:.{precision}f} > {previous_sell_price:.{precision}f}), référence mise à jour"
            elif current_price > sell_threshold:
                # Prix légèrement en baisse mais dans la marge de tolérance
                logger.info(f"🟡 Prix stable pour {signal.symbol}: {current_price:.{precision}f} > seuil {sell_threshold:.{precision}f} (marge {self.sell_margin*100:.1f}%), GARDE")
                return False, f"Prix dans marge de tolérance ({current_price:.{precision}f} > {sell_threshold:.{precision}f}), position gardée"
            else:
                # Prix baisse significativement : VENDRE !
                logger.info(f"🔍 Prix baisse significative, nettoyage référence")
                logger.info(f"📉 Prix baisse significative pour {signal.symbol}: {current_price:.{precision}f} ≤ {sell_threshold:.{precision}f}, SELL exécuté !")
                # Nettoyer la référence après vente
                self._clear_sell_reference(signal.symbol)
                return True, f"Prix baisse significative ({current_price:.{precision}f} ≤ {sell_threshold:.{precision}f}), SELL exécuté"
            
        except Exception as e:
            logger.error(f"❌ EXCEPTION dans _check_trailing_sell pour {signal.symbol}: {str(e)}")
            logger.error(f"❌ Type exception: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            # En cas d'erreur, autoriser le SELL par sécurité
            return True, f"Erreur technique, SELL autorisé par défaut"
    
    def _get_previous_sell_price(self, symbol: str) -> Optional[float]:
        """
        Récupère le prix du SELL précédent stocké en référence.
        
        Args:
            symbol: Symbole à vérifier
            
        Returns:
            Prix du SELL précédent ou None
        """
        try:
            ref_key = f"sell_reference:{symbol}"
            price_data = self.redis_client.get(ref_key)
            
            if not price_data:
                return None
            
            logger.debug(f"🔍 Récupération sell reference {symbol}: type={type(price_data)}, data={price_data}")
            
            # Gérer tous les cas possibles de retour Redis
            if isinstance(price_data, dict):
                # Déjà un dictionnaire Python
                if "price" in price_data:
                    return float(price_data["price"])
                else:
                    logger.warning(f"Clé 'price' manquante dans dict Redis pour {symbol}: {price_data}")
                    return None
            
            elif isinstance(price_data, (str, bytes)):
                # String JSON à parser
                try:
                    if isinstance(price_data, bytes):
                        price_data = price_data.decode('utf-8')
                    
                    parsed_data = json.loads(price_data)
                    if isinstance(parsed_data, dict) and "price" in parsed_data:
                        return float(parsed_data["price"])
                    else:
                        logger.warning(f"Format JSON invalide pour {symbol}: {parsed_data}")
                        return None
                except json.JSONDecodeError as e:
                    logger.error(f"Erreur JSON decode pour {symbol}: {e}, data: {price_data}")
                    return None
            
            else:
                logger.warning(f"Type Redis inattendu pour {symbol}: {type(price_data)}, data: {price_data}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur récupération sell reference pour {symbol}: {e}")
            logger.error(f"Type: {type(price_data) if 'price_data' in locals() else 'undefined'}, Data: {price_data if 'price_data' in locals() else 'undefined'}")
            return None
    
    def _update_sell_reference(self, symbol: str, price: float) -> None:
        """
        Met à jour la référence de prix SELL pour un symbole.
        
        Args:
            symbol: Symbole
            price: Nouveau prix de référence
        """
        try:
            ref_key = f"sell_reference:{symbol}"
            ref_data = {
                "price": price,
                "timestamp": int(time.time() * 1000)
            }
            # TTL de 2 heures pour éviter les références obsolètes
            self.redis_client.set(ref_key, json.dumps(ref_data), expiration=7200)
        except Exception as e:
            logger.error(f"Erreur mise à jour sell reference pour {symbol}: {e}")
    
    def _clear_sell_reference(self, symbol: str) -> None:
        """
        Supprime la référence de prix SELL pour un symbole.
        
        Args:
            symbol: Symbole
        """
        try:
            ref_key = f"sell_reference:{symbol}"
            self.redis_client.delete(ref_key)
            logger.info(f"🧹 Référence SELL supprimée pour {symbol}")
        except Exception as e:
            logger.error(f"Erreur suppression sell reference pour {symbol}: {e}")
    
    def _analyze_symbol_trend(self, symbol: str, current_price: float, entry_price: float) -> str:
        """
        Analyse simple de la tendance d'un symbole pour adapter le stop-loss.
        
        Args:
            symbol: Symbole à analyser
            current_price: Prix actuel
            entry_price: Prix d'entrée
            
        Returns:
            'STRONG_BULLISH', 'BULLISH', ou 'NEUTRAL'
        """
        try:
            # 1. Performance depuis l'entrée
            performance_pct = ((current_price - entry_price) / entry_price) * 100
            
            # 2. Récupérer l'historique de prix récent depuis Redis
            price_history_key = f"price_history:{symbol}"
            try:
                price_history = self.redis_client.get(price_history_key)
                if price_history:
                    if isinstance(price_history, str):
                        price_history = json.loads(price_history)
                    
                    if isinstance(price_history, list) and len(price_history) >= 10:
                        prices = [float(p) for p in price_history[-20:]]  # 20 derniers prix
                        
                        # Calculer la pente de tendance (régression linéaire simple)
                        n = len(prices)
                        x_sum = sum(range(n))
                        y_sum = sum(prices)
                        xy_sum = sum(i * prices[i] for i in range(n))
                        x2_sum = sum(i * i for i in range(n))
                        
                        if n * x2_sum - x_sum * x_sum != 0:
                            slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
                            slope_pct = (slope / current_price) * 100  # Normaliser
                        else:
                            slope_pct = 0
                    else:
                        slope_pct = 0
                else:
                    slope_pct = 0
                    
            except Exception:
                slope_pct = 0
            
            # 3. Logique de classification
            if performance_pct > 5.0 and slope_pct > 0.1:  # +5% performance + pente positive
                return "STRONG_BULLISH"
            elif performance_pct > 2.0 or slope_pct > 0.05:  # +2% ou pente légèrement positive
                return "BULLISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Erreur analyse tendance {symbol}: {e}")
            return "NEUTRAL"
    
    def _calculate_adaptive_threshold(self, symbol: str, entry_price: float, entry_time: float) -> float:
        """
        Calcule le seuil de stop-loss adaptatif basé sur les données d'analyse de marché.
        
        Args:
            symbol: Symbole à analyser (ex: 'ETHUSDC')
            entry_price: Prix d'entrée de la position
            entry_time: Timestamp d'entrée (epoch)
            
        Returns:
            Seuil de perte acceptable avant stop-loss (ex: 0.015 = 1.5%)
        """
        try:
            # Récupérer les données d'analyse les plus récentes
            analysis = self._get_latest_analysis_data(symbol)
            if not analysis:
                logger.warning(f"⚠️ Pas de données d'analyse pour {symbol}, utilisation seuil par défaut")
                return 0.010  # 1% par défaut si pas de données
                
            # === FACTEUR 1: RÉGIME DE MARCHÉ ===
            regime_factor = self._calculate_regime_factor(analysis)
            
            # === FACTEUR 2: VOLATILITÉ ===
            volatility_factor = self._calculate_volatility_factor(analysis)
            
            # === FACTEUR 3: SUPPORT TECHNIQUE ===
            support_factor = self._calculate_support_factor(analysis, entry_price)
            
            # === FACTEUR 4: TEMPS ÉCOULÉ ===
            time_factor = self._calculate_time_factor(entry_time)
            
            # === CALCUL DU SEUIL FINAL ===
            base_threshold = 0.008  # 0.8% de base
            
            # Application des facteurs (conversion explicite en float)
            adaptive_threshold = float(base_threshold) * float(regime_factor) * float(volatility_factor) * float(support_factor) * float(time_factor)
            
            # Contraintes min/max
            adaptive_threshold = max(0.003, min(0.025, adaptive_threshold))  # Entre 0.3% et 2.5%
            
            logger.debug(f"🧠 Stop-loss adaptatif {symbol}: {adaptive_threshold*100:.2f}% "
                        f"(régime:{regime_factor:.2f}, vol:{volatility_factor:.2f}, "
                        f"support:{support_factor:.2f}, temps:{time_factor:.2f})")
            
            return adaptive_threshold
            
        except Exception as e:
            logger.error(f"Erreur calcul stop-loss adaptatif {symbol}: {e}")
            return 0.010  # Fallback sécurisé
    
    def _get_latest_analysis_data(self, symbol: str) -> Optional[Dict]:
        """
        Récupère les données d'analyse les plus récentes depuis analyzer_data.
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Dictionnaire avec les données d'analyse ou None
        """
        try:
            if not self.db_connection:
                return None
                
            with self.db_connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        market_regime, regime_strength, regime_confidence,
                        volatility_regime, atr_percentile,
                        nearest_support, support_strength,
                        trend_alignment, directional_bias
                    FROM analyzer_data 
                    WHERE symbol = %s AND timeframe = '1m'
                    ORDER BY time DESC 
                    LIMIT 1
                """, (symbol,))
                
                result = cursor.fetchone()
                if result:
                    # Conversion explicite de tous les types Decimal en float
                    return {
                        'market_regime': result[0],
                        'regime_strength': result[1], 
                        'regime_confidence': float(result[2]) if result[2] is not None else 50.0,
                        'volatility_regime': result[3],
                        'atr_percentile': float(result[4]) if result[4] is not None else 50.0,
                        'nearest_support': float(result[5]) if result[5] is not None else None,
                        'support_strength': result[6],
                        'trend_alignment': result[7],
                        'directional_bias': result[8]
                    }
                return None
        except Exception as e:
            logger.error(f"Erreur récupération données analyse {symbol}: {e}")
            return None
    
    def _calculate_regime_factor(self, analysis: Dict) -> float:
        """
        Calcule le facteur d'ajustement basé sur le régime de marché.
        
        Args:
            analysis: Données d'analyse de marché
            
        Returns:
            Facteur multiplicateur (0.5 à 2.0)
        """
        regime = analysis.get('market_regime', 'UNKNOWN')
        strength = analysis.get('regime_strength', 'WEAK')
        
        # Conversion sécurisée en float pour éviter les erreurs Decimal
        try:
            confidence = float(analysis.get('regime_confidence', 50))
        except (ValueError, TypeError):
            confidence = 50.0
        
        # Base selon le régime
        regime_multipliers = {
            'TRENDING_BULL': 1.8,      # Plus patient en tendance haussière
            'BREAKOUT_BULL': 1.6,      # Patient sur les breakouts haussiers
            'RANGING': 1.2,            # Légèrement plus patient en range
            'TRANSITION': 0.9,         # Plus strict en transition
            'TRENDING_BEAR': 0.7,      # Plus strict en baisse
            'VOLATILE': 0.8,           # Plus strict en volatilité
            'BREAKOUT_BEAR': 0.6       # Très strict sur breakouts baissiers
        }
        
        base_factor = regime_multipliers.get(regime, 1.0)
        
        # Ajustement selon la force
        strength_multipliers = {
            'EXTREME': 1.3,
            'STRONG': 1.1, 
            'MODERATE': 1.0,
            'WEAK': 0.8
        }
        
        strength_factor = strength_multipliers.get(strength, 1.0)
        
        # Ajustement selon la confiance (conversion float sécurisée)
        confidence_factor = 0.7 + (float(confidence) / 100.0) * 0.6  # 0.7 à 1.3
        
        return float(base_factor * strength_factor * confidence_factor)
    
    def _calculate_volatility_factor(self, analysis: Dict) -> float:
        """
        Calcule le facteur d'ajustement basé sur la volatilité.
        
        Args:
            analysis: Données d'analyse de marché
            
        Returns:
            Facteur multiplicateur (0.6 à 2.0)
        """
        volatility_regime = analysis.get('volatility_regime', 'normal')
        
        # Conversion sécurisée en float pour éviter les erreurs Decimal
        try:
            atr_percentile = float(analysis.get('atr_percentile', 50))
        except (ValueError, TypeError):
            atr_percentile = 50.0
        
        # Base selon régime de volatilité
        volatility_multipliers = {
            'low': 0.7,        # Plus strict si volatilité faible
            'normal': 1.0,     # Standard
            'high': 1.4,       # Plus patient si volatilité élevée
            'extreme': 1.8     # Très patient si volatilité extrême
        }
        
        base_factor = volatility_multipliers.get(volatility_regime, 1.0)
        
        # Ajustement fin selon percentile ATR (conversion float sécurisée)
        percentile_factor = 0.6 + (float(atr_percentile) / 100.0) * 0.8  # 0.6 à 1.4
        
        return float(base_factor * percentile_factor)
    
    def _calculate_support_factor(self, analysis: Dict, entry_price: float) -> float:
        """
        Calcule le facteur d'ajustement basé sur la proximité des supports.
        
        Args:
            analysis: Données d'analyse de marché
            entry_price: Prix d'entrée de la position
            
        Returns:
            Facteur multiplicateur (0.8 à 2.0)
        """
        nearest_support = analysis.get('nearest_support')
        support_strength = analysis.get('support_strength', 'WEAK')
        
        if not nearest_support:
            return 1.0
            
        # Conversion sécurisée en float pour éviter les erreurs Decimal
        try:
            support_price = float(nearest_support)
            entry_price_float = float(entry_price)
        except (ValueError, TypeError):
            return 1.0
            
        # Distance au support (en %)
        support_distance = abs(entry_price_float - support_price) / entry_price_float
        
        # Plus on est proche d'un support fort, plus on est patient
        strength_multipliers = {
            'MAJOR': 1.6,      # Support majeur = très patient
            'STRONG': 1.3,     # Support fort = patient
            'MODERATE': 1.1,   # Support modéré = légèrement patient
            'WEAK': 0.9        # Support faible = légèrement strict
        }
        
        strength_factor = strength_multipliers.get(support_strength, 1.0)
        
        # Facteur de distance (plus proche = plus patient)
        if support_distance < 0.01:      # < 1%
            distance_factor = 1.4
        elif support_distance < 0.02:    # < 2%
            distance_factor = 1.2
        elif support_distance < 0.05:    # < 5%
            distance_factor = 1.0
        else:                            # > 5%
            distance_factor = 0.8
            
        return float(strength_factor * distance_factor)
    
    def _calculate_time_factor(self, entry_time: float) -> float:
        """
        Calcule le facteur d'ajustement basé sur le temps écoulé.
        
        Args:
            entry_time: Timestamp d'entrée (epoch)
            
        Returns:
            Facteur multiplicateur (0.8 à 1.3)
        """
        time_elapsed = float(time.time() - float(entry_time))
        minutes_elapsed = time_elapsed / 60.0
        
        # Plus patient sur les trades récents (bruit de marché)
        # Plus strict sur les trades anciens
        if minutes_elapsed < 5:      # < 5 min
            return 1.3               # Très patient
        elif minutes_elapsed < 15:   # < 15 min
            return 1.1               # Patient
        elif minutes_elapsed < 60:   # < 1h
            return 1.0               # Standard
        elif minutes_elapsed < 240:  # < 4h
            return 0.9               # Légèrement strict
        else:                        # > 4h
            return 0.8               # Plus strict
    
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
            
            # Récupérer le prix actuel depuis Redis
            current_price = self._get_current_price(symbol)
            if not current_price:
                logger.warning(f"⚠️ Prix actuel indisponible pour {symbol}")
                return
            
            # 1. VÉRIFICATION STOP-LOSS ADAPTATIF INTELLIGENT
            loss_percent = (entry_price - current_price) / entry_price
            adaptive_threshold = self._calculate_adaptive_threshold(symbol, entry_price, entry_time_epoch)
            
            precision = self._get_price_precision(current_price)
            
            if loss_percent >= adaptive_threshold:
                logger.warning(f"🚨 STOP-LOSS ADAPTATIF DÉCLENCHÉ pour {symbol}!")
                logger.warning(f"📉 Prix entrée: {entry_price:.{precision}f}")
                logger.warning(f"📉 Prix actuel: {current_price:.{precision}f}")
                logger.warning(f"📉 Seuil adaptatif: {adaptive_threshold*100:.2f}%")
                logger.warning(f"📉 Perte: {loss_percent*100:.2f}%")
                
                # Déclencher vente d'urgence avec mention du système adaptatif
                self._execute_emergency_sell(symbol, current_price, str(cycle_id) if cycle_id else "unknown", "ADAPTIVE_STOP_LOSS_AUTO")
                return
            
            # 2. MISE À JOUR AUTOMATIQUE DU TRAILING SELL
            # Seulement si position gagnante
            if current_price > entry_price:
                previous_sell_price = self._get_previous_sell_price(symbol)
                
                if previous_sell_price is None or current_price > previous_sell_price:
                    # Nouveau pic détecté, mettre à jour la référence
                    self._update_sell_reference(symbol, current_price)
                    precision = self._get_price_precision(current_price)
                    logger.debug(f"📈 Trailing auto-mis à jour pour {symbol}: {current_price:.{precision}f}")
                else:
                    # Vérifier si on doit déclencher le trailing sell
                    sell_threshold = previous_sell_price * (1 - self.sell_margin)
                    
                    if current_price <= sell_threshold:
                        precision = self._get_price_precision(current_price)
                        logger.warning(f"🚨 TRAILING SELL DÉCLENCHÉ pour {symbol}!")
                        logger.warning(f"📉 Prix référence: {previous_sell_price:.{precision}f}")
                        logger.warning(f"📉 Prix actuel: {current_price:.{precision}f}")
                        logger.warning(f"📉 Seuil trailing: {sell_threshold:.{precision}f}")
                        
                        # Déclencher vente trailing
                        self._execute_emergency_sell(symbol, current_price, str(cycle_id) if cycle_id else "unknown", "TRAILING_SELL_AUTO")
                        return
            
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
                self._clear_sell_reference(symbol)
                
            else:
                logger.error(f"❌ Échec création ordre stop-loss pour {symbol}")
                self.stats["errors"] += 1
                
        except Exception as e:
            logger.error(f"❌ Erreur vente d'urgence {symbol}: {e}")
            self.stats["errors"] += 1