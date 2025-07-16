"""
Gestionnaire de cycles de trading pour ROOT.
Gère la création et le suivi des cycles BUY->SELL sans impacter l'exécution des trades.
"""
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal
import json

from shared.src.db_pool import transaction
from shared.src.enums import OrderSide

logger = logging.getLogger(__name__)


class CycleManager:
    """
    Gère les cycles de trading (BUY->SELL) pour le calcul des P&L.
    Fonctionne de manière indépendante sans influencer l'exécution des trades.
    """
    
    def __init__(self):
        """Initialise le gestionnaire de cycles."""
        logger.info("✅ CycleManager initialisé")
    
    def process_trade_execution(self, trade_data: Dict[str, Any]) -> Optional[str]:
        """
        Traite une exécution de trade pour mettre à jour les cycles.
        Cette méthode est appelée APRÈS l'exécution réussie d'un trade.
        
        Args:
            trade_data: Données du trade exécuté
                - symbol: Symbole tradé
                - side: BUY ou SELL
                - price: Prix d'exécution
                - quantity: Quantité exécutée
                - order_id: ID de l'ordre
                - timestamp: Timestamp d'exécution
                - strategy: Stratégie utilisée (optionnel)
                
        Returns:
            ID du cycle créé/mis à jour ou None si erreur
        """
        try:
            symbol = trade_data.get('symbol')
            side = trade_data.get('side')
            price = Decimal(str(trade_data.get('price', 0)))
            quantity = Decimal(str(trade_data.get('quantity', 0)))
            order_id = trade_data.get('order_id')
            strategy = trade_data.get('strategy', 'Unknown')
            
            if not all([symbol, side, price > 0, quantity > 0]):
                logger.warning(f"⚠️ Données de trade incomplètes: {trade_data}")
                return
            
            # Convertir side en OrderSide si nécessaire
            if isinstance(side, str):
                side = OrderSide(side.upper())
            
            if side == OrderSide.BUY:
                return self._handle_buy_trade(
                    symbol=symbol,
                    price=price,
                    quantity=quantity,
                    order_id=order_id,
                    strategy=strategy
                )
            elif side == OrderSide.SELL:
                return self._handle_sell_trade(
                    symbol=symbol,
                    price=price,
                    quantity=quantity,
                    order_id=order_id,
                    strategy=strategy
                )
                
        except Exception as e:
            logger.error(f"❌ Erreur dans process_trade_execution: {str(e)}")
            # On ne propage pas l'erreur pour ne pas impacter le trading
            return None
    
    def _handle_buy_trade(self, symbol: str, price: Decimal, quantity: Decimal, 
                         order_id: str, strategy: str) -> Optional[str]:
        """
        Gère un trade BUY en créant ou mettant à jour un cycle.
        
        Args:
            symbol: Symbole tradé
            price: Prix d'achat
            quantity: Quantité achetée
            order_id: ID de l'ordre
            strategy: Stratégie utilisée
            
        Returns:
            ID du cycle créé/mis à jour
        """
        try:
            # Chercher un cycle actif pour ce symbole
            active_cycle = self._get_active_cycle(symbol)
            
            if active_cycle:
                # Mettre à jour le cycle existant (ajout de position)
                self._update_cycle_for_buy(
                    cycle_id=active_cycle['id'],
                    new_price=price,
                    new_quantity=quantity,
                    order_id=order_id
                )
                logger.info(f"📈 Cycle mis à jour pour {symbol}: +{quantity} @ {price}")
                return active_cycle['id']
            else:
                # Créer un nouveau cycle
                cycle_id = self._create_new_cycle(
                    symbol=symbol,
                    price=price,
                    quantity=quantity,
                    order_id=order_id,
                    strategy=strategy
                )
                logger.info(f"🆕 Nouveau cycle créé pour {symbol}: {quantity} @ {price}")
                return cycle_id
                
        except Exception as e:
            logger.error(f"❌ Erreur dans _handle_buy_trade: {str(e)}")
            return None
    
    def _handle_sell_trade(self, symbol: str, price: Decimal, quantity: Decimal,
                          order_id: str, strategy: str) -> Optional[str]:
        """
        Gère un trade SELL en fermant le(s) cycle(s) correspondant(s).
        
        Args:
            symbol: Symbole tradé
            price: Prix de vente
            quantity: Quantité vendue
            order_id: ID de l'ordre
            strategy: Stratégie utilisée
            
        Returns:
            ID du cycle fermé ou None
        """
        try:
            # Récupérer tous les cycles actifs pour ce symbole
            active_cycles = self._get_active_cycles_for_symbol(symbol)
            
            if not active_cycles:
                logger.warning(f"⚠️ Aucun cycle actif trouvé pour SELL {symbol}")
                # Créer un cycle orphelin pour tracker ce SELL
                return self._create_orphan_sell_cycle(
                    symbol=symbol,
                    price=price,
                    quantity=quantity,
                    order_id=order_id,
                    strategy=strategy
                )
            
            # Fermer tous les cycles du symbole vendu (on vend toujours tout sur un symbole)
            closed_cycle_id = None
            total_cycle_quantity = Decimal('0')
            
            for cycle in active_cycles:
                cycle_quantity = Decimal(str(cycle['quantity']))
                total_cycle_quantity += cycle_quantity
                
                # Fermer complètement ce cycle
                self._close_cycle(
                    cycle_id=cycle['id'],
                    exit_price=price,
                    exit_quantity=cycle_quantity,
                    order_id=order_id
                )
                closed_cycle_id = cycle['id']
            
            # Log si différence de quantité (max 1 USDC de valeur)
            quantity_diff = abs(quantity - total_cycle_quantity)
            if quantity_diff > 0:
                # Calculer la valeur en USDC de la différence
                diff_value_usdc = quantity_diff * price
                if diff_value_usdc < Decimal('1.0'):
                    logger.debug(f"📊 Dust négligeable: {quantity_diff} {symbol} (~{diff_value_usdc:.4f} USDC)")
                else:
                    logger.warning(f"⚠️ Différence importante: vendu {quantity}, cycles {total_cycle_quantity}, diff {quantity_diff} (~{diff_value_usdc:.2f} USDC)")
                
            return closed_cycle_id
                
        except Exception as e:
            logger.error(f"❌ Erreur dans _handle_sell_trade: {str(e)}")
            return None
    
    def _get_active_cycle(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Récupère le cycle actif le plus récent pour un symbole.
        
        Args:
            symbol: Symbole à rechercher
            
        Returns:
            Cycle actif ou None
        """
        try:
            with transaction() as cursor:
                cursor.execute("""
                    SELECT * FROM trade_cycles 
                    WHERE symbol = %s 
                    AND status IN ('active_buy', 'waiting_sell')
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (symbol,))
                
                result = cursor.fetchone()
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, result))
                return None
        except Exception as e:
            logger.error(f"❌ Erreur _get_active_cycle: {e}")
            return None
    
    def _get_active_cycles_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Récupère tous les cycles actifs pour un symbole (ordre FIFO).
        
        Args:
            symbol: Symbole à rechercher
            
        Returns:
            Liste des cycles actifs
        """
        try:
            with transaction() as cursor:
                cursor.execute("""
                    SELECT * FROM trade_cycles 
                    WHERE symbol = %s 
                    AND status IN ('active_buy', 'waiting_sell')
                    ORDER BY created_at ASC
                """, (symbol,))
                
                results = cursor.fetchall()
                if results:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in results]
                return []
        except Exception as e:
            logger.error(f"❌ Erreur _get_active_cycles_for_symbol: {e}")
            return []
    
    def _create_new_cycle(self, symbol: str, price: Decimal, quantity: Decimal,
                         order_id: str, strategy: str) -> str:
        """
        Crée un nouveau cycle de trading.
        
        Args:
            symbol: Symbole tradé
            price: Prix d'entrée
            quantity: Quantité
            order_id: ID de l'ordre d'entrée
            strategy: Stratégie utilisée
            
        Returns:
            ID du cycle créé
        """
        try:
            cycle_id = f"cycle_{uuid.uuid4().hex[:16]}"
            
            with transaction() as cursor:
                cursor.execute("""
                    INSERT INTO trade_cycles (
                        id, symbol, strategy, status, side, entry_order_id,
                        entry_price, quantity, min_price, max_price,
                        created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        NOW(), NOW()
                    )
                """, (
                    cycle_id, symbol, strategy, 'active_buy', 'BUY', order_id,
                    price, quantity, price, price
                ))
            
            return cycle_id
        except Exception as e:
            logger.error(f"❌ Erreur _create_new_cycle: {e}")
            return ""
    
    def _update_cycle_for_buy(self, cycle_id: str, new_price: Decimal,
                             new_quantity: Decimal, order_id: str) -> None:
        """
        Met à jour un cycle existant avec un nouveau BUY (ajout de position).
        
        Args:
            cycle_id: ID du cycle à mettre à jour
            new_price: Prix du nouveau BUY
            new_quantity: Quantité du nouveau BUY
            order_id: ID de l'ordre
        """
        try:
            with transaction() as cursor:
                # Récupérer le cycle actuel
                cursor.execute("""
                    SELECT entry_price, quantity, min_price, max_price, metadata
                    FROM trade_cycles WHERE id = %s
                """, (cycle_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.error(f"Cycle {cycle_id} non trouvé")
                    return
                
                current_price = Decimal(str(result[0]))
                current_quantity = Decimal(str(result[1]))
                min_price = Decimal(str(result[2])) if result[2] else current_price
                max_price = Decimal(str(result[3])) if result[3] else current_price
                
                try:
                    metadata = json.loads(result[4]) if result[4] else {}
                except:
                    metadata = {}
                
                # Calculer le nouveau prix moyen pondéré
                total_value = (current_price * current_quantity) + (new_price * new_quantity)
                new_total_quantity = current_quantity + new_quantity
                new_avg_price = total_value / new_total_quantity
                
                # Mettre à jour min/max
                min_price = min(min_price, new_price)
                max_price = max(max_price, new_price)
                
                # Ajouter l'historique des BUY dans metadata
                if 'buy_history' not in metadata:
                    metadata['buy_history'] = []
                
                metadata['buy_history'].append({
                    'order_id': order_id,
                    'price': str(new_price),
                    'quantity': str(new_quantity),
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # Mettre à jour le cycle
                cursor.execute("""
                    UPDATE trade_cycles SET
                        entry_price = %s,
                        quantity = %s,
                        min_price = %s,
                        max_price = %s,
                        metadata = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (
                    new_avg_price, new_total_quantity,
                    min_price, max_price, json.dumps(metadata),
                    cycle_id
                ))
                
        except Exception as e:
            logger.error(f"❌ Erreur _update_cycle_for_buy: {e}")
    
    def _close_cycle(self, cycle_id: str, exit_price: Decimal,
                    exit_quantity: Decimal, order_id: str) -> None:
        """
        Ferme complètement un cycle et calcule le P&L.
        
        Args:
            cycle_id: ID du cycle à fermer
            exit_price: Prix de sortie
            exit_quantity: Quantité vendue
            order_id: ID de l'ordre de sortie
        """
        try:
            with transaction() as cursor:
                # Récupérer les données du cycle
                cursor.execute("""
                    SELECT entry_price, quantity, metadata
                    FROM trade_cycles WHERE id = %s
                """, (cycle_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.error(f"Cycle {cycle_id} non trouvé")
                    return
                
                entry_price = Decimal(str(result[0]))
                quantity = Decimal(str(result[1]))
                
                try:
                    metadata = json.loads(result[2]) if result[2] else {}
                except:
                    metadata = {}
                
                # Calculer le P&L
                profit_loss = (exit_price - entry_price) * exit_quantity
                profit_loss_percent = ((exit_price - entry_price) / entry_price) * 100
                
                # Ajouter les infos de SELL dans metadata
                metadata['sell_info'] = {
                    'order_id': order_id,
                    'price': str(exit_price),
                    'quantity': str(exit_quantity),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Mettre à jour le cycle
                cursor.execute("""
                    UPDATE trade_cycles SET
                        status = 'completed',
                        exit_order_id = %s,
                        exit_price = %s,
                        profit_loss = %s,
                        profit_loss_percent = %s,
                        metadata = %s,
                        completed_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                """, (
                    order_id, exit_price, profit_loss,
                    profit_loss_percent, json.dumps(metadata), cycle_id
                ))
                
                logger.info(f"✅ Cycle {cycle_id} fermé: P&L={profit_loss:.2f} ({profit_loss_percent:.2f}%)")
                
        except Exception as e:
            logger.error(f"❌ Erreur _close_cycle: {e}")
    
    def _partial_close_cycle(self, cycle_id: str, exit_price: Decimal,
                            exit_quantity: Decimal, order_id: str) -> None:
        """
        Ferme partiellement un cycle (vente partielle de la position).
        
        Args:
            cycle_id: ID du cycle
            exit_price: Prix de sortie
            exit_quantity: Quantité vendue
            order_id: ID de l'ordre
        """
        try:
            with transaction() as cursor:
                # Récupérer les données du cycle
                cursor.execute("""
                    SELECT * FROM trade_cycles WHERE id = %s
                """, (cycle_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.error(f"Cycle {cycle_id} non trouvé")
                    return
                
                columns = [desc[0] for desc in cursor.description]
                cycle_data = dict(zip(columns, result))
                
                current_quantity = Decimal(str(cycle_data['quantity']))
                remaining_quantity = current_quantity - exit_quantity
                
                if remaining_quantity <= 0:
                    # Si on vend tout ou plus, fermer complètement
                    self._close_cycle(cycle_id, exit_price, current_quantity, order_id)
                    return
                
                # Fermer le cycle actuel avec la quantité vendue
                self._close_cycle(cycle_id, exit_price, exit_quantity, order_id)
                
                # Créer un nouveau cycle avec la quantité restante
                new_cycle_id = f"cycle_{uuid.uuid4().hex[:16]}"
                
                try:
                    metadata = json.loads(cycle_data.get('metadata', '{}') or '{}')
                except:
                    metadata = {}
                
                metadata['partial_from_cycle'] = cycle_id
                
                cursor.execute("""
                    INSERT INTO trade_cycles (
                        id, symbol, strategy, status, side, entry_order_id,
                        entry_price, quantity, min_price, max_price,
                        metadata, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, NOW(), NOW()
                    )
                """, (
                    new_cycle_id, cycle_data['symbol'], cycle_data['strategy'],
                    'active_buy', 'BUY', cycle_data['entry_order_id'],
                    cycle_data['entry_price'], remaining_quantity,
                    cycle_data['min_price'], cycle_data['max_price'],
                    json.dumps(metadata)
                ))
                
                logger.info(f"📊 Fermeture partielle: {exit_quantity} @ {exit_price}, reste {remaining_quantity}")
                
        except Exception as e:
            logger.error(f"❌ Erreur _partial_close_cycle: {e}")
    
    def _create_orphan_sell_cycle(self, symbol: str, price: Decimal,
                                 quantity: Decimal, order_id: str, strategy: str) -> Optional[str]:
        """
        Crée un cycle "orphelin" pour un SELL sans BUY correspondant.
        Utile pour tracker les ventes de positions préexistantes.
        
        Args:
            symbol: Symbole vendu
            price: Prix de vente
            quantity: Quantité vendue
            order_id: ID de l'ordre
            strategy: Stratégie utilisée
            
        Returns:
            ID du cycle orphelin créé
        """
        try:
            cycle_id = f"cycle_orphan_{uuid.uuid4().hex[:16]}"
            
            metadata = {
                'orphan_sell': True,
                'reason': 'No active buy cycle found',
                'order_id': order_id
            }
            
            with transaction() as cursor:
                cursor.execute("""
                    INSERT INTO trade_cycles (
                        id, symbol, strategy, status, side,
                        exit_order_id, exit_price, quantity,
                        metadata, created_at, updated_at, completed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, NOW(), NOW(), NOW()
                    )
                """, (
                    cycle_id, symbol, strategy, 'completed', 'SELL',
                    order_id, price, quantity, json.dumps(metadata)
                ))
            
            logger.warning(f"⚠️ Cycle orphelin créé pour SELL {symbol}: {quantity} @ {price}")
            return cycle_id
            
        except Exception as e:
            logger.error(f"❌ Erreur _create_orphan_sell_cycle: {e}")
            return None