"""
Module de traitement des signaux - VERSION SIMPLIFIÉE ET PROPRE.
Utilise uniquement le nouveau système avec buffer intelligent et consensus adaptatif.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
import json
from hierarchical_validator import HierarchicalValidator

logger = logging.getLogger(__name__)
validation_logger = logging.getLogger('signal_aggregator.validation')


class SignalProcessor:
    """
    Processeur simplifié pour la validation des signaux.
    Ne gère QUE les signaux individuels - le buffering est géré par SignalBuffer.
    """
    
    def __init__(self, validator_loader, context_manager, database_manager=None):
        """
        Initialise le processeur de signaux.
        
        Args:
            validator_loader: Chargeur de validators
            context_manager: Gestionnaire de contexte de marché
            database_manager: Gestionnaire de base de données (optionnel)
        """
        self.validator_loader = validator_loader
        self.context_manager = context_manager
        self.database_manager = database_manager
        
        # Système hiérarchique uniquement
        self.hierarchical_validator = HierarchicalValidator()
        
        # Statistiques simplifiées
        self.stats = {
            'signals_processed': 0,
            'signals_validated': 0,
            'signals_rejected': 0,
            'signals_vetoed': 0,
            'validation_errors': 0,
            'veto_reasons': {}
        }
        
    async def process_signal(self, signal_data: str) -> Optional[Dict[str, Any]]:
        """
        Traite un signal individuel reçu depuis Redis.
        
        Args:
            signal_data: Données du signal au format JSON
            
        Returns:
            Signal validé et scoré ou None si rejeté
        """
        try:
            # Parsing du message
            signal = json.loads(signal_data)
            
            # Validation directe - plus de distinction batch/individuel
            return await self._validate_signal(signal)
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON signal: {e}")
            self.stats['validation_errors'] += 1
            return None
        except Exception as e:
            logger.error(f"Erreur traitement signal: {e}")
            self.stats['validation_errors'] += 1
            return None
            
    async def _validate_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Valide un signal avec le système hiérarchique.
        
        Args:
            signal: Signal à valider
            
        Returns:
            Signal validé ou None si rejeté
        """
        try:
            self.stats['signals_processed'] += 1
            
            # Validation structure de base
            if not self._validate_signal_structure(signal):
                logger.warning("Signal rejeté: structure invalide")
                self.stats['signals_rejected'] += 1
                return None
                
            symbol = signal['symbol']
            timeframe = signal['timeframe']
            
            # Récupération du contexte de marché
            context = self.context_manager.get_market_context(symbol, timeframe)
            if not context:
                logger.warning(f"Impossible de récupérer le contexte pour {symbol} {timeframe}")
                self.stats['signals_rejected'] += 1
                return None
                
            # Chargement des validators
            validators = self.validator_loader.get_all_validators()
            if not validators:
                logger.warning("Aucun validator disponible")
                self.stats['signals_rejected'] += 1
                return None
                
            # Validation avec tous les validators
            validation_results = await self._run_all_validators(signal, context, validators)
            
            # Validation hiérarchique avec pouvoir de veto
            is_valid, final_score, detailed_analysis = self.hierarchical_validator.validate_with_hierarchy(
                validation_results, signal
            )
            
            # Gestion du veto (Market Structure Validator peut bloquer)
            if detailed_analysis.get('veto', False):
                veto_reason = detailed_analysis.get('veto_reason', 'Veto sans raison')
                logger.warning(f"Signal REJETÉ par VETO: {signal['strategy']} {symbol} {timeframe} "
                             f"{signal['side']} - {veto_reason}")
                
                self.stats['signals_vetoed'] += 1
                if veto_reason not in self.stats['veto_reasons']:
                    self.stats['veto_reasons'][veto_reason] = 0
                self.stats['veto_reasons'][veto_reason] += 1
                
                return None
            
            if is_valid:
                # Construction du signal validé
                validated_signal = self._build_validated_signal(
                    signal, detailed_analysis, validation_results, final_score
                )
                
                self.stats['signals_validated'] += 1
                
                # SAUVEGARDE EN BASE DE DONNÉES
                if self.database_manager:
                    try:
                        signal_id = self.database_manager.store_validated_signal(validated_signal)
                        if signal_id:
                            validated_signal['db_id'] = signal_id
                            logger.debug(f"Signal sauvegardé en DB avec ID: {signal_id}")
                        else:
                            logger.warning(f"Échec sauvegarde signal {symbol} {timeframe}")
                    except Exception as e:
                        logger.error(f"Erreur sauvegarde signal en DB: {e}")
                
                logger.info(f"Signal VALIDÉ: {signal['strategy']} {symbol} {timeframe} "
                          f"{signal['side']} (score={final_score:.2f})")
                
                return validated_signal
            else:
                self.stats['signals_rejected'] += 1
                
                logger.info(f"Signal REJETÉ: {signal['strategy']} {symbol} {timeframe} "
                          f"{signal['side']} (score={final_score:.2f})")
                
                return None
                
        except Exception as e:
            logger.error(f"Erreur validation signal: {e}")
            self.stats['validation_errors'] += 1
            return None
        
    def _validate_signal_structure(self, signal: Dict[str, Any]) -> bool:
        """Valide la structure de base du signal."""
        required_fields = ['symbol', 'timeframe', 'strategy', 'side', 'confidence', 'timestamp']
        
        for field in required_fields:
            if field not in signal:
                logger.warning(f"Champ manquant dans le signal: {field}")
                return False
                
        # Validation des valeurs
        if signal['side'] not in ['BUY', 'SELL']:
            logger.warning(f"Side invalide: {signal['side']}")
            return False
            
        if not 0 <= signal['confidence'] <= 1:
            logger.warning(f"Confidence invalide: {signal['confidence']}")
            return False
            
        return True
        
    async def _run_all_validators(self, signal: Dict[str, Any], context: Dict[str, Any], 
                                 validators: Dict) -> list:
        """Exécute tous les validators sur le signal."""
        validation_results = []
        
        for validator_name, validator_class in validators.items():
            try:
                # Préparation du contexte aplati
                flat_context = {}
                if 'indicators' in context:
                    flat_context.update(context['indicators'])
                if 'market_structure' in context:
                    flat_context.update(context['market_structure'])
                if 'volume_profile' in context:
                    flat_context.update(context['volume_profile'])
                
                # Données OHLCV
                ohlcv_data = context.get('ohlcv_data', [])
                data_dict = {}
                
                if ohlcv_data:
                    latest_candle = ohlcv_data[-1]
                    data_dict = {
                        'open': latest_candle.get('open'),
                        'high': latest_candle.get('high'),
                        'low': latest_candle.get('low'),
                        'close': latest_candle.get('close'),
                        'volume': latest_candle.get('volume'),
                        'quote_volume': latest_candle.get('quote_volume')
                    }
                    flat_context['current_price'] = latest_candle.get('close', 0)
                    
                # Instanciation et validation
                validator = validator_class(
                    symbol=signal['symbol'],
                    data=data_dict,
                    context=flat_context
                )
                
                is_valid = validator.validate_signal(signal)
                score = validator.get_validation_score(signal)
                reason = validator.get_validation_reason(signal, is_valid)
                
                validation_results.append({
                    'validator_name': validator_name,
                    'is_valid': is_valid,
                    'score': score,
                    'reason': reason
                })
                
            except Exception as e:
                logger.error(f"Erreur validator {validator_name}: {e}")
                validation_results.append({
                    'validator_name': validator_name,
                    'is_valid': False,
                    'score': 0.0,
                    'reason': f"ERREUR: {str(e)}"
                })
                
        return validation_results
        
    def _build_validated_signal(self, signal: Dict[str, Any], detailed_analysis: Dict[str, Any], 
                               validation_results: list, final_score: float) -> Dict[str, Any]:
        """Construit le signal validé final."""
        # Prix requis pour le coordinator
        price = self._extract_signal_price(signal)
        
        return {
            'strategy': signal['strategy'],
            'symbol': signal['symbol'],
            'side': signal['side'],
            'timestamp': signal['timestamp'],
            'price': price,
            'confidence': min(1.0, signal.get('confidence', 0) * final_score),
            'strength': signal.get('strength', 'moderate'),
            'timeframe': signal.get('timeframe'),
            'metadata': {
                **signal.get('metadata', {}),
                'validation_timestamp': datetime.utcnow().isoformat(),
                'validation_score': final_score,
                'hierarchical_analysis': detailed_analysis,
                'validators_passed': sum(1 for r in validation_results if r['is_valid']),
                'total_validators': len(validation_results)
            }
        }
        
    def _extract_signal_price(self, signal: Dict[str, Any]) -> float:
        """Extrait le prix du signal."""
        try:
            if 'price' in signal and signal['price'] is not None:
                return float(signal['price'])
                
            # Récupérer depuis le contexte
            symbol = signal.get('symbol')
            timeframe = signal.get('timeframe')
            
            if symbol and timeframe and self.context_manager:
                context = self.context_manager.get_market_context(symbol, timeframe)
                ohlcv_data = context.get('ohlcv_data', [])
                if ohlcv_data and len(ohlcv_data) > 0:
                    return float(ohlcv_data[-1].get('close', 0))
                    
            return 0.0
            
        except Exception as e:
            logger.error(f"Erreur extraction prix signal: {e}")
            return 0.0
        
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques simplifiées."""
        stats = self.stats.copy()
        
        if stats['signals_processed'] > 0:
            stats['veto_rate'] = stats['signals_vetoed'] / stats['signals_processed']
            stats['validation_rate'] = stats['signals_validated'] / stats['signals_processed']
            stats['rejection_rate'] = stats['signals_rejected'] / stats['signals_processed']
        
        return stats