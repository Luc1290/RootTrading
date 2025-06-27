"""
Module de traitement et validation des signaux.
Centralise la logique de validation et de décision pour les signaux.
"""
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from datetime import datetime

from shared.src.schemas import StrategySignal
from shared.src.enums import OrderSide, SignalStrength
from coordinator.src.service_client import ServiceClient

logger = logging.getLogger(__name__)


class SignalProcessor:
    """
    Traite et valide les signaux de trading.
    Gère l'historique, la validation et les décisions basées sur les signaux.
    """
    
    def __init__(self, service_client: ServiceClient, 
                 max_cycles_per_symbol_side: int = 3,
                 signal_expiry_seconds: float = 10.0):
        """
        Initialise le processeur de signaux.
        
        Args:
            service_client: Client pour accéder aux services externes
            max_cycles_per_symbol_side: Nombre max de cycles par symbole/côté
            signal_expiry_seconds: Durée avant expiration d'un signal
        """
        self.service_client = service_client
        self.max_cycles_per_symbol_side = max_cycles_per_symbol_side
        self.signal_expiry_seconds = signal_expiry_seconds
        
        # Historique des signaux récents pour anti-spam
        self.recent_signals_history: Dict[str, List[Tuple[StrategySignal, float]]] = defaultdict(lambda: deque(maxlen=10))
        
        # Métriques de performance
        self.metrics = {
            "signals_received": 0,
            "signals_processed": 0,
            "signals_rejected": 0,
            "cycles_created": 0
        }
        
    def validate_signal(self, signal: StrategySignal) -> Tuple[bool, str]:
        """
        Valide un signal selon plusieurs critères.
        
        Args:
            signal: Signal à valider
            
        Returns:
            Tuple (is_valid, reason)
        """
        # Incrémenter le compteur
        self.metrics["signals_received"] += 1
        
        # 1. Vérifier l'expiration du signal
        current_time = time.time()
        
        # Gérer différents formats de timestamp
        if hasattr(signal.timestamp, 'timestamp'):
            # datetime object
            signal_time = signal.timestamp.timestamp()
        elif isinstance(signal.timestamp, (int, float)):
            # Unix timestamp en secondes ou millisecondes
            signal_time = signal.timestamp
            if signal_time > 1e12:  # Si en millisecondes
                signal_time = signal_time / 1000.0
        else:
            # String - essayer de parser
            try:
                from datetime import datetime
                signal_time = datetime.fromisoformat(str(signal.timestamp).replace('Z', '+00:00')).timestamp()
            except:
                # Si on ne peut pas parser, ignorer la vérification d'expiration
                signal_time = current_time
                
        signal_age = current_time - signal_time
        if signal_age > self.signal_expiry_seconds:
            self.metrics["signals_rejected"] += 1
            return False, f"Signal expiré ({signal_age:.1f}s > {self.signal_expiry_seconds}s)"
            
        # 2. Vérifier les cycles existants
        active_cycles = self.service_client.get_active_cycles(signal.symbol)
        if not self._check_cycle_limits(signal, active_cycles):
            self.metrics["signals_rejected"] += 1
            return False, "Limite de cycles atteinte"
            
        # 3. Vérifier l'anti-spam (sauf pour signaux agrégés)
        if not signal.strategy.startswith("Aggregated_"):
            if self._is_spam_signal(signal):
                self.metrics["signals_rejected"] += 1
                return False, "Trop de signaux récents similaires"
                
        # 4. Vérifier les balances
        balance_check = self._validate_balance_for_signal(signal)
        if not balance_check["can_trade"]:
            self.metrics["signals_rejected"] += 1
            return False, balance_check.get("reason", "Balance insuffisante")
            
        self.metrics["signals_processed"] += 1
        return True, "Signal valide"
        
    def _check_cycle_limits(self, signal: StrategySignal, active_cycles: List[Dict]) -> bool:
        """
        Vérifie si on peut créer un nouveau cycle selon les limites.
        
        Args:
            signal: Signal à traiter
            active_cycles: Liste des cycles actifs
            
        Returns:
            True si on peut créer un nouveau cycle
        """
        # Compter les cycles par position
        position_counts = {"BUY": 0, "SELL": 0}
        
        for cycle in active_cycles:
            if cycle.get("symbol") != signal.symbol:
                continue
                
            status = cycle.get("status", "")
            if status in ["waiting_sell", "active_sell"]:
                position_counts["BUY"] += 1
            elif status in ["waiting_buy", "active_buy"]:
                position_counts["SELL"] += 1
                
        # Vérifier la limite pour la position du signal
        signal_position = signal.side.value if hasattr(signal.side, 'value') else str(signal.side)
        current_count = position_counts.get(signal_position, 0)
        
        if current_count >= self.max_cycles_per_symbol_side:
            logger.warning(f"Limite atteinte: {current_count}/{self.max_cycles_per_symbol_side} "
                         f"cycles {signal_position} pour {signal.symbol}")
            return False
            
        return True
        
    def _is_spam_signal(self, signal: StrategySignal) -> bool:
        """
        Vérifie si le signal est du spam (trop de signaux similaires récents).
        
        Args:
            signal: Signal à vérifier
            
        Returns:
            True si c'est du spam
        """
        current_time = time.time()
        
        # Nettoyer l'historique
        self._clean_signal_history()
        
        # Vérifier les signaux récents du même côté
        recent_same_side = [
            s for s, t in self.recent_signals_history[signal.symbol]
            if (s.side.value if hasattr(s.side, 'value') else str(s.side)) == (signal.side.value if hasattr(signal.side, 'value') else str(signal.side)) and current_time - t < 10.0
        ]
        
        # Ajouter le signal actuel à l'historique
        self.recent_signals_history[signal.symbol].append((signal, current_time))
        
        # Si plus de 2 signaux similaires en 10 secondes, c'est du spam
        return len(recent_same_side) > 2
        
    def _clean_signal_history(self):
        """
        Nettoie l'historique des signaux expirés.
        """
        current_time = time.time()
        
        for symbol in list(self.recent_signals_history.keys()):
            # Garder seulement les signaux des 30 dernières secondes
            self.recent_signals_history[symbol] = deque(
                [(s, t) for s, t in self.recent_signals_history[symbol] 
                 if current_time - t < 30.0],
                maxlen=10
            )
            
            # Supprimer l'entrée si vide
            if not self.recent_signals_history[symbol]:
                del self.recent_signals_history[symbol]
                
    def _validate_balance_for_signal(self, signal: StrategySignal) -> Dict[str, Any]:
        """
        Valide que les balances sont suffisantes pour le signal.
        
        Args:
            signal: Signal à valider
            
        Returns:
            Dict avec can_trade et reason
        """
        # NE PAS utiliser un montant fixe pour la vérification
        # La vérification réelle avec le montant dynamique sera faite dans signal_handler
        # via AllocationManager. Ici on vérifie juste qu'il y a une balance minimum
        
        # Montants minimums par devise pour la vérification basique
        min_amounts = {
            'USDC': 10.0,
            'BTC': 0.0001,
            'ETH': 0.003,
            'BNB': 0.02,
            'SOL': 0.1,
            'XRP': 10.0,
            'ADA': 20.0,
            'DOT': 1.0
        }
        
        # Déterminer quelle devise on a besoin selon le côté
        if signal.side == OrderSide.BUY or (hasattr(signal.side, 'value') and signal.side.value == 'BUY'):
            # Pour BUY, on a besoin de la devise de quote (USDC, BTC, etc.)
            quote_asset = 'USDC'  # Par défaut
            if signal.symbol.endswith('USDT'):
                quote_asset = 'USDT'
            elif signal.symbol.endswith('BTC'):
                quote_asset = 'BTC'
            elif signal.symbol.endswith('ETH'):
                quote_asset = 'ETH'
            
            # Utiliser le montant minimum pour la vérification basique
            min_amount = min_amounts.get(quote_asset, 10.0)
        else:
            # Pour SELL, on a besoin de la devise de base (SOL, BTC, ETH, etc.)
            # Extraire l'actif de base du symbole
            base_asset = signal.symbol.replace('USDC', '').replace('USDT', '').replace('BTC', '').replace('ETH', '')
            
            # Utiliser le montant minimum pour l'actif de base
            min_amount = min_amounts.get(base_asset, 0.1)
        
        return self.service_client.check_balance_for_trade(
            symbol=signal.symbol,
            side=signal.side.value if hasattr(signal.side, 'value') else str(signal.side),
            amount=min_amount
        )
        
    def should_process_signal_strategically(self, signal: StrategySignal,
                                          active_cycles: List[Dict]) -> Tuple[bool, str]:
        """
        Décide si un signal doit être traité selon la stratégie globale.
        
        Args:
            signal: Signal à évaluer
            active_cycles: Cycles actifs
            
        Returns:
            Tuple (should_process, reason)
        """
        # Compter les positions actuelles
        position_counts = {"BUY": 0, "SELL": 0}
        opposite_cycles = []
        
        signal_position = signal.side.value if hasattr(signal.side, 'value') else str(signal.side)
        opposite_position = "SELL" if signal_position == "BUY" else "BUY"
        
        for cycle in active_cycles:
            if cycle.get("symbol") != signal.symbol:
                continue
                
            status = cycle.get("status", "")
            if status in ["waiting_sell", "active_sell"]:
                position_counts["BUY"] += 1
                if signal_position == "SELL":
                    opposite_cycles.append(cycle)
            elif status in ["waiting_buy", "active_buy"]:
                position_counts["SELL"] += 1
                if signal_position == "BUY":
                    opposite_cycles.append(cycle)
                    
        # S'il y a des positions opposées, évaluer le retournement
        if opposite_cycles:
            # Calculer la force du signal pour décider du retournement
            signal_strength = self._calculate_signal_strength_score(signal)
            
            # Log pour debug
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"🔍 Signal {signal.symbol} {signal.side}: strength={signal.strength}, confidence={getattr(signal, 'confidence', 'N/A')}, metadata={getattr(signal, 'metadata', {})}")
            
            # Seuils adaptés selon le type de signal
            threshold = 0.85  # Seuil par défaut
            
            if signal.metadata and signal.metadata.get("ultra_confluence"):
                score = signal.metadata.get("total_score", 0)
                if score >= 95:
                    threshold = 0.75  # Plus permissif pour signaux institutionnels
                elif score >= 85:
                    threshold = 0.80  # Légèrement plus permissif pour excellents
                    
            if signal_strength >= threshold:
                # Fermer les cycles opposés avant d'accepter le nouveau signal
                try:
                    self._close_opposite_cycles(opposite_cycles, signal)
                    return True, f"Retournement accepté (force: {signal_strength:.2f}) - positions opposées fermées"
                except Exception as e:
                    logger.error(f"❌ Échec fermeture cycles opposés: {str(e)}")
                    return False, f"Retournement rejeté - impossible de fermer cycles opposés: {str(e)}"
            else:
                return False, f"Signal trop faible pour retournement (force: {signal_strength:.2f})"
                
        return True, "Pas de conflit de position"
        
    def _calculate_signal_strength_score(self, signal: StrategySignal) -> float:
        """
        Calcule un score de force pour le signal.
        
        Args:
            signal: Signal à évaluer
            
        Returns:
            Score entre 0 et 1
        """
        # Si c'est un signal ultra-confluent avec score
        if signal.metadata and signal.metadata.get("ultra_confluence"):
            score = signal.metadata.get("total_score", 0)
            return min(score / 100.0, 1.0)
            
        # Sinon, utiliser la force et la confidence
        strength_weights = {
            SignalStrength.WEAK: 0.25,
            SignalStrength.MODERATE: 0.5,
            SignalStrength.STRONG: 0.75,
            SignalStrength.VERY_STRONG: 1.0
        }
        
        base_score = strength_weights.get(signal.strength, 0.5)
        confidence = getattr(signal, 'confidence', None) or 1.0  # Défaut à 1.0 si None
        return base_score * confidence
        
    def calculate_portfolio_conviction(self, symbol: str, position: str, 
                                      active_cycles: List[Dict]) -> float:
        """
        Calcule la conviction du portfolio sur une position.
        
        Args:
            symbol: Symbole concerné
            position: Position à évaluer (BUY/SELL)
            active_cycles: Cycles actifs pour ce symbole
            
        Returns:
            Score de conviction entre 0 et 1
        """
        # Compter les cycles actifs pour cette position
        position_cycles = 0
        for cycle in active_cycles:
            if cycle.get("symbol") != symbol:
                continue
            status = cycle.get("status", "")
            if status in ["waiting_sell", "active_sell"] and position == "BUY":
                position_cycles += 1
            elif status in ["waiting_buy", "active_buy"] and position == "SELL":
                position_cycles += 1
                
        # 1. Facteur basé sur le nombre de cycles (normalisé sur 3 max)
        cycles_factor = min(position_cycles / 3.0, 1.0)
        
        # 2. Performance récente (pour l'instant valeur fixe)
        performance_factor = 0.5
        
        # 3. Durée des positions (pour l'instant valeur fixe)
        duration_factor = 0.6
        
        # 4. Cohérence des signaux récents
        recent_coherence = self._calculate_recent_signal_coherence(symbol, position)
        
        # Moyenne pondérée
        conviction = (
            cycles_factor * 0.4 +
            performance_factor * 0.2 +
            duration_factor * 0.2 +
            recent_coherence * 0.2
        )
        
        return conviction
        
    def _calculate_recent_signal_coherence(self, symbol: str, position: str) -> float:
        """
        Calcule la cohérence des signaux récents pour une position.
        
        Args:
            symbol: Symbole concerné
            position: Position à évaluer (BUY/SELL)
            
        Returns:
            Score de cohérence entre 0 et 1
        """
        if symbol not in self.recent_signals_history:
            return 0.5  # Valeur neutre si pas d'historique
            
        recent_signals = self.recent_signals_history[symbol]
        if not recent_signals:
            return 0.5
            
        # Compter les signaux de cette position dans les 10 dernières minutes
        current_time = time.time()
        matching_signals = 0
        total_recent = 0
        
        for signal, timestamp in recent_signals:
            if current_time - timestamp < 600:  # 10 minutes
                total_recent += 1
                if (signal.side.value if hasattr(signal.side, 'value') else str(signal.side)) == position:
                    matching_signals += 1
                    
        if total_recent == 0:
            return 0.5
            
        return matching_signals / total_recent

    def _close_opposite_cycles(self, opposite_cycles: List[Dict], signal: StrategySignal) -> None:
        """
        Ferme les cycles opposés avant d'ouvrir une nouvelle position.
        
        Args:
            opposite_cycles: Liste des cycles opposés à fermer
            signal: Signal qui justifie la fermeture
        """
        import logging
        logger = logging.getLogger(__name__)
        
        for cycle in opposite_cycles:
            cycle_id = cycle.get('id')
            cycle_status = cycle.get('status', '')
            if not cycle_id:
                continue
                
            # Vérifier si le cycle peut être fermé
            if cycle_status in ['completed', 'canceled', 'failed']:
                logger.info(f"🔍 Cycle {cycle_id} déjà fermé (statut: {cycle_status}), ignoré")
                continue
                
            try:
                # Préparer les données de fermeture
                close_data = {
                    "reason": f"Fermeture pour retournement - Signal {signal.strategy} force {self._calculate_signal_strength_score(signal):.2f}",
                    "price": signal.price  # Fermer au prix du signal
                }
                
                # Fermeture comptable via l'API
                result = self.service_client.close_cycle_accounting(
                    cycle_id, 
                    signal.price, 
                    f"Retournement vers {signal.side} - Signal {signal.strategy}"
                )
                
                # Vérifier le succès de la fermeture comptable
                success = False
                if result:
                    # Vérifier les différents indicateurs de succès
                    if result.get('success') is True:
                        success = True
                    elif result.get('status') in ['closed_accounting', 'already_closed']:
                        success = True
                    elif 'error' not in result and result.get('cycle_id'):
                        success = True
                
                if success:
                    logger.info(f"✅ Cycle opposé {cycle_id} fermé pour retournement")
                else:
                    error_msg = result.get('error', 'Erreur inconnue') if result else 'Aucune réponse'
                    logger.warning(f"⚠️ Échec fermeture cycle opposé {cycle_id}: {error_msg}")
                    # Ne pas traiter le signal si on n'a pas pu fermer les cycles opposés
                    raise Exception(f"Impossible de fermer le cycle opposé {cycle_id}: {error_msg}")
                    
            except Exception as e:
                logger.error(f"❌ Erreur fermeture cycle opposé {cycle_id}: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retourne les métriques du processeur.
        
        Returns:
            Dict des métriques
        """
        return {
            **self.metrics,
            "signal_history_size": sum(len(hist) for hist in self.recent_signals_history.values()),
            "monitored_symbols": len(self.recent_signals_history)
        }