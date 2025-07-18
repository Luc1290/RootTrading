"""
Coordinator simplifié pour RootTrading.
Rôle : Valider la faisabilité des signaux et les transmettre au trader.
"""
import logging
import time
from typing import Dict, Any, Optional
from decimal import Decimal

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
        
        # Configuration dynamique basée sur le capital total
        self.fee_rate = 0.001  # 0.1% de frais estimés par trade
        
        # Allocation dynamique intelligente
        self.base_allocation_percent = 8.0  # 8% par défaut du capital total
        self.max_trade_percent = 15.0  # 15% maximum du capital total
        self.min_absolute_trade_usdc = 10.0  # 10 USDC minimum Binance
        
        # Stats
        self.stats = {
            "signals_received": 0,
            "signals_processed": 0,
            "signals_rejected": 0,
            "orders_sent": 0,
            "errors": 0
        }
        
        logger.info("✅ Coordinator initialisé (version simplifiée)")
    
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
            
            # Vérifier la faisabilité
            is_feasible, reason = self._check_feasibility(signal)
            
            if not is_feasible:
                logger.warning(f"❌ Signal rejeté: {reason}")
                self.stats["signals_rejected"] += 1
                return
            
            # Vérifier l'efficacité du trade (logique simplifiée)
            is_efficient, efficiency_reason = self._check_trade_efficiency(signal)
            
            if not is_efficient:
                logger.warning(f"❌ Signal rejeté: {efficiency_reason}")
                self.stats["signals_rejected"] += 1
                return
            
            # Calculer la quantité à trader
            quantity = self._calculate_quantity(signal)
            if not quantity or quantity <= 0:
                logger.error("Impossible de calculer la quantité")
                self.stats["signals_rejected"] += 1
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
            else:
                logger.error("❌ Échec création ordre")
                self.stats["errors"] += 1
                
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
            
            # Extraire les assets
            base_asset = self._get_base_asset(signal.symbol)
            quote_asset = self._get_quote_asset(signal.symbol)
            
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
        Vérifie s'il y a un cycle actif pour ce symbole.
        
        Args:
            symbol: Symbole à vérifier (ex: 'BTCUSDC')
            
        Returns:
            ID du cycle actif si trouvé, None sinon
        """
        try:
            # Appeler le service trader pour vérifier les cycles actifs
            response = self.service_client.get_active_cycles(symbol)
            
            if response and response.get('active_cycles'):
                # Retourner l'ID du premier cycle actif trouvé
                active_cycle = response['active_cycles'][0]
                return active_cycle.get('id', 'unknown')
            
            return None
            
        except Exception as e:
            logger.warning(f"Erreur vérification cycle actif pour {symbol}: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du coordinator."""
        return self.stats.copy()