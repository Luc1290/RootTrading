"""
Module de traitement et validation des signaux avec système hiérarchique.
Contient la logique principale de validation, scoring et filtrage des signaux.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import json
from hierarchical_validator import HierarchicalValidator

logger = logging.getLogger(__name__)

# Logger spécialisé pour la validation
validation_logger = logging.getLogger('signal_aggregator.validation')


class SignalProcessor:
    """Processeur principal pour la validation et le scoring des signaux avec hiérarchie."""
    
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
        
        # Initialisation du système hiérarchique
        self.hierarchical_validator = HierarchicalValidator()
        
        # Configuration des seuils de base
        self.min_strategies_consensus = 2    # Minimum de stratégies pour consensus
        
        # Statistiques de validation
        self.stats = {
            'signals_processed': 0,
            'signals_validated': 0,
            'signals_rejected': 0,
            'signals_vetoed': 0,
            'validation_errors': 0,
            'avg_validation_score': 0.0,
            'validator_performance': {},
            'veto_reasons': {}
        }
        
    async def process_signal(self, signal_data: str) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Traite un signal ou un batch de signaux reçu depuis Redis.
        
        Args:
            signal_data: Données du signal au format JSON
            
        Returns:
            Signal validé et scoré ou liste de signaux validés, ou None si rejeté
        """
        try:
            # Parsing du message
            message = json.loads(signal_data)
            
            # Router vers la logique appropriée
            if isinstance(message, dict) and message.get('type') == 'signal_batch':
                # Traitement d'un batch de signaux avec consensus intelligent
                return await self._process_signal_batch(message)
            else:
                # Traitement d'un signal individuel
                return await self._process_individual_signal(message)
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur parsing JSON signal: {e}")
            self.stats['validation_errors'] += 1
            return None
        except Exception as e:
            logger.error(f"Erreur traitement signal: {e}")
            self.stats['validation_errors'] += 1
            return None
            
    async def _process_individual_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Traite un signal individuel sans logique de consensus.
        
        Args:
            signal: Signal à traiter
            
        Returns:
            Signal validé et scoré ou None si rejeté
        """
        try:
            self.stats['signals_processed'] += 1
            
            logger.debug(f"Traitement signal individuel: {signal.get('strategy')} {signal.get('symbol')} "
                        f"{signal.get('timeframe')} {signal.get('side')}")
            
            # Validation de base du signal
            if not self._validate_signal_structure(signal):
                logger.warning("Signal rejeté: structure invalide")
                return None
                
            # Marquer comme signal individuel (pas de consensus requis)
            if 'metadata' not in signal:
                signal['metadata'] = {}
            signal['metadata']['is_individual'] = True
            signal['metadata']['strategy_count'] = 1
                
            # Validation complète avec contexte
            validated_signal = await self._validate_signal_core(signal)
            
            return validated_signal
            
        except Exception as e:
            logger.error(f"Erreur traitement signal individuel: {e}")
            self.stats['validation_errors'] += 1
            return None
            
    async def _process_signal_batch(self, batch_message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Traite un batch de signaux avec logique de consensus intelligente.
        
        Args:
            batch_message: Message contenant le batch de signaux
            
        Returns:
            Liste des signaux validés ou None si erreur
        """
        try:
            signals = batch_message.get('signals', [])
            if not signals:
                logger.warning("Batch vide reçu")
                return None
                
            logger.info(f"Traitement batch de {len(signals)} signaux")
            
            # Étape 1: Grouper les signaux par symbole et direction
            groups = self._group_signals_by_symbol_and_side(signals)
            
            # Étape 2: Analyser les consensus et conflits
            consensus_analysis = self._analyze_consensus_and_conflicts(groups)
            
            # Étape 3: Sélectionner les signaux à traiter selon la stratégie de résolution
            selected_signals = self._select_signals_from_analysis(consensus_analysis)
            
            # Étape 4: Valider les signaux sélectionnés
            validated_signals = []
            for signal in selected_signals:
                validated_signal = await self._validate_signal_core(signal)
                if validated_signal:
                    validated_signals.append(validated_signal)
                    
            logger.info(f"Batch traité: {len(validated_signals)}/{len(signals)} signaux validés")
            
            return validated_signals if validated_signals else None
            
        except Exception as e:
            logger.error(f"Erreur traitement batch: {e}")
            self.stats['validation_errors'] += 1
            return None
            
    def _group_signals_by_symbol_and_side(self, signals: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Groupe les signaux par symbole et par direction (BUY/SELL).
        
        Args:
            signals: Liste des signaux à grouper
            
        Returns:
            Structure: {symbol: {side: [signals]}}
        """
        groups = {}
        
        for signal in signals:
            symbol = signal.get('symbol')
            side = signal.get('side')
            
            if symbol not in groups:
                groups[symbol] = {}
            if side not in groups[symbol]:
                groups[symbol][side] = []
                
            groups[symbol][side].append(signal)
            
        return groups
        
    def _analyze_consensus_and_conflicts(self, groups: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Dict]:
        """
        Analyse les consensus et les conflits BUY/SELL pour chaque symbole.
        
        Args:
            groups: Groupes de signaux par symbole et direction
            
        Returns:
            Analyse des consensus avec stratégie de résolution pour chaque symbole
        """
        analysis = {}
        
        for symbol, sides in groups.items():
            buy_signals = sides.get('BUY', [])
            sell_signals = sides.get('SELL', [])
            
            buy_count = len(buy_signals)
            sell_count = len(sell_signals)
            
            buy_consensus = buy_count >= self.min_strategies_consensus
            sell_consensus = sell_count >= self.min_strategies_consensus
            
            analysis[symbol] = {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'buy_count': buy_count,
                'sell_count': sell_count,
                'buy_consensus': buy_consensus,
                'sell_consensus': sell_consensus,
                'conflict': buy_consensus and sell_consensus,
                'resolution_strategy': None,
                'selected_signals': []
            }
            
            # Déterminer la stratégie de résolution
            if buy_consensus and sell_consensus:
                # Conflit : choisir le groupe avec la meilleure confidence moyenne
                buy_avg_conf = sum(s.get('confidence', 0) for s in buy_signals) / buy_count
                sell_avg_conf = sum(s.get('confidence', 0) for s in sell_signals) / sell_count
                
                if buy_avg_conf > sell_avg_conf:
                    analysis[symbol]['resolution_strategy'] = 'prioritize_buy'
                    analysis[symbol]['selected_signals'] = buy_signals
                    logger.warning(f"Conflit {symbol}: Priorité BUY (conf={buy_avg_conf:.2f}) > SELL (conf={sell_avg_conf:.2f})")
                else:
                    analysis[symbol]['resolution_strategy'] = 'prioritize_sell'
                    analysis[symbol]['selected_signals'] = sell_signals
                    logger.warning(f"Conflit {symbol}: Priorité SELL (conf={sell_avg_conf:.2f}) > BUY (conf={buy_avg_conf:.2f})")
                    
            elif buy_consensus:
                analysis[symbol]['resolution_strategy'] = 'buy_only'
                analysis[symbol]['selected_signals'] = buy_signals
                logger.debug(f"Consensus BUY uniquement pour {symbol}: {buy_count} stratégies")
                
            elif sell_consensus:
                analysis[symbol]['resolution_strategy'] = 'sell_only'
                analysis[symbol]['selected_signals'] = sell_signals
                logger.debug(f"Consensus SELL uniquement pour {symbol}: {sell_count} stratégies")
                
            else:
                analysis[symbol]['resolution_strategy'] = 'no_consensus'
                analysis[symbol]['selected_signals'] = []
                logger.debug(f"Aucun consensus pour {symbol}: BUY={buy_count}, SELL={sell_count}")
                
        return analysis
        
    def _select_signals_from_analysis(self, analysis: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Sélectionne et prépare les signaux à valider selon l'analyse de consensus.
        
        Args:
            analysis: Analyse des consensus par symbole
            
        Returns:
            Liste des signaux préparés pour validation
        """
        selected_signals = []
        
        for symbol, data in analysis.items():
            strategy = data['resolution_strategy']
            signals = data['selected_signals']
            
            if strategy == 'no_consensus':
                # Pas de consensus : ne pas traiter les signaux
                continue
                
            # Préparer les signaux sélectionnés avec les métadonnées appropriées
            for signal in signals:
                if 'metadata' not in signal:
                    signal['metadata'] = {}
                    
                # Métadonnées de consensus
                signal['metadata']['is_individual'] = False
                signal['metadata']['has_consensus'] = True
                signal['metadata']['strategy_count'] = len(signals)
                signal['metadata']['consensus_group'] = f"{symbol}_{signal.get('side')}"
                signal['metadata']['resolution_strategy'] = strategy
                signal['metadata']['symbol_analysis'] = {
                    'buy_count': data['buy_count'],
                    'sell_count': data['sell_count'],
                    'conflict_detected': data['conflict']
                }
                
                selected_signals.append(signal)
                
        return selected_signals
        
    async def _validate_signal_core(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Fonction de validation centrale sans duplication.
        Gère à la fois les signaux individuels et les signaux de consensus.
        
        Args:
            signal: Signal à valider (avec métadonnées de consensus si applicable)
            
        Returns:
            Signal validé et scoré ou None si rejeté
        """
        try:
            symbol = signal['symbol']
            timeframe = signal['timeframe']
            
            # Vérification du consensus si requis
            is_individual = signal.get('metadata', {}).get('is_individual', False)
            has_consensus = signal.get('metadata', {}).get('has_consensus', False)
            strategy_count = signal.get('metadata', {}).get('strategy_count', 1)
            
            # Logique de consensus unifié
            if not is_individual:  # Signal de batch
                if not has_consensus:
                    logger.info(f"Signal REJETÉ pour manque de consensus: {signal['strategy']} {symbol} "
                              f"{signal['side']} - {strategy_count} stratégies < {self.min_strategies_consensus} requis")
                    self.stats['signals_rejected'] += 1
                    return None
            # Les signaux individuels passent toujours cette étape
            
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
                
            # Validation avec chaque validator
            validation_results = await self._run_all_validators(signal, context, validators)
            
            # Validation hiérarchique
            is_valid, final_score, detailed_analysis = self.hierarchical_validator.validate_with_hierarchy(
                validation_results, signal
            )
            
            # Gestion du veto
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
                validated_signal = self._build_validated_signal_hierarchical(
                    signal, detailed_analysis, validation_results, final_score
                )
                
                # Filtre final : aggregator_confidence >= 70%
                aggregator_confidence = validated_signal['metadata'].get('aggregator_confidence', 0.0)
                min_aggregator_confidence = 0.7
                
                if aggregator_confidence < min_aggregator_confidence:
                    logger.info(f"Signal REJETÉ pour aggregator_confidence insuffisante: {signal['strategy']} {symbol} "
                              f"{signal['side']} - confidence={aggregator_confidence:.1%} < {min_aggregator_confidence:.1%}")
                    self.stats['signals_rejected'] += 1
                    return None
                
                # Stockage en base de données si disponible
                if self.database_manager:
                    try:
                        signal_id = self.database_manager.store_validated_signal(validated_signal)
                        if signal_id:
                            validated_signal['metadata']['db_id'] = signal_id
                            logger.info(f"Signal stocké en DB avec ID: {signal_id}")
                    except Exception as e:
                        logger.error(f"Erreur stockage signal en DB: {e}")
                
                self.stats['signals_validated'] += 1
                self._update_validation_stats_hierarchical(validation_results, final_score, detailed_analysis)
                
                logger.info(f"Signal VALIDÉ: {signal['strategy']} {symbol} {timeframe} "
                          f"{signal['side']} (score={final_score:.2f}, agg_conf={aggregator_confidence:.1%})")
                
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
        """
        Valide la structure de base du signal.
        
        Args:
            signal: Signal à valider
            
        Returns:
            True si la structure est valide, False sinon
        """
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
                                 validators: Dict) -> List[Dict[str, Any]]:
        """
        Exécute tous les validators sur le signal.
        
        Args:
            signal: Signal à valider
            context: Contexte de marché
            validators: Dict des validators disponibles
            
        Returns:
            Liste des résultats de validation
        """
        validation_results = []
        
        for validator_name, validator_class in validators.items():
            try:
                # Préparation du contexte aplati pour le validator
                flat_context = {}
                # Ajouter les indicateurs directement
                if 'indicators' in context:
                    flat_context.update(context['indicators'])
                # Ajouter d'autres données importantes
                if 'market_structure' in context:
                    flat_context.update(context['market_structure'])
                if 'volume_profile' in context:
                    flat_context.update(context['volume_profile'])
                
                # Récupérer les données OHLCV
                ohlcv_data = context.get('ohlcv_data', [])
                
                # Ajouter le prix actuel dans le contexte aplati
                if ohlcv_data and len(ohlcv_data) > 0:
                    flat_context['current_price'] = ohlcv_data[-1].get('close', 0)
                    
                # Préparer les données OHLCV pour les validators
                data_dict = {}
                if ohlcv_data:
                    latest_candle = ohlcv_data[-1]
                    # Format mixte pour compatibilité avec tous les validators
                    data_dict = {
                        # Valeurs scalaires (dernière bougie) - pour la majorité des validators
                        'open': latest_candle.get('open'),
                        'high': latest_candle.get('high'),
                        'low': latest_candle.get('low'),
                        'close': latest_candle.get('close'),
                        'volume': latest_candle.get('volume'),
                        'quote_volume': latest_candle.get('quote_volume'),
                        # Listes complètes pour les validators qui en ont besoin
                        'open_list': [candle.get('open') for candle in ohlcv_data],
                        'high_list': [candle.get('high') for candle in ohlcv_data],
                        'low_list': [candle.get('low') for candle in ohlcv_data],
                        'close_list': [candle.get('close') for candle in ohlcv_data],
                        'volume_list': [candle.get('volume') for candle in ohlcv_data],
                        'quote_volume_list': [candle.get('quote_volume') for candle in ohlcv_data],
                        'ohlcv_list': ohlcv_data  # Liste complète des bougies
                    }
                    
                # Instanciation du validator
                validator = validator_class(
                    symbol=signal['symbol'],
                    data=data_dict,
                    context=flat_context
                )
                
                # Validation
                is_valid = validator.validate_signal(signal)
                score = validator.get_validation_score(signal)
                reason = validator.get_validation_reason(signal, is_valid)
                
                result = {
                    'validator_name': validator_name,
                    'is_valid': is_valid,
                    'score': score,
                    'reason': reason,
                    'execution_time': 0  # TODO: Mesurer le temps d'exécution
                }
                
                validation_results.append(result)
                
                # Log détaillé par validator (DEBUG uniquement)
                if logger.isEnabledFor(logging.DEBUG):
                    importance = self.hierarchical_validator.get_validator_importance(validator_name)
                    validation_logger.debug(f"{signal['symbol']} {signal.get('timeframe', 'N/A')} - "
                                          f"{importance} {validator_name}: {'PASS' if is_valid else 'FAIL'} "
                                          f"(score={score:.2f}) - {reason[:50]}")
                
            except Exception as e:
                logger.error(f"Erreur validator {validator_name}: {e}")
                
                # Ajouter un résultat d'erreur
                validation_results.append({
                    'validator_name': validator_name,
                    'is_valid': False,
                    'score': 0.0,
                    'reason': f"ERREUR: {str(e)}",
                    'execution_time': 0
                })
                
        return validation_results
        
    def _build_validated_signal_hierarchical(self, signal: Dict[str, Any], detailed_analysis: Dict[str, Any], 
                                           validation_results: List[Dict[str, Any]], final_score: float) -> Dict[str, Any]:
        """
        Construit le signal validé final avec les métadonnées hiérarchiques.
        
        Args:
            signal: Signal original
            detailed_analysis: Analyse hiérarchique détaillée
            validation_results: Résultats détaillés
            final_score: Score final
            
        Returns:
            Signal validé complet
        """
        # Extraction et ajout du prix (requis par le coordinator)
        price = self._extract_signal_price(signal)
        
        # Conversion du strength pour correspondre aux enums du coordinator
        strength_mapping = {
            'weak': 'weak',
            'moderate': 'moderate', 
            'strong': 'strong',
            'very_strong': 'very_strong',
            'very_weak': 'very_weak'
        }
        
        original_strength = signal.get('strength', 'moderate')
        mapped_strength = strength_mapping.get(original_strength, 'moderate')
        
        # Extraction des informations hiérarchiques
        level_analysis = detailed_analysis.get('level_analysis', {})
        combination_bonuses = detailed_analysis.get('combination_bonuses', [])
        
        # Construction du signal conforme au schéma StrategySignal
        validated_signal = {
            # Champs obligatoires pour StrategySignal
            'strategy': signal['strategy'],
            'symbol': signal['symbol'],
            'side': signal['side'],
            'timestamp': signal['timestamp'],
            'price': price,
            
            # Champs optionnels pour StrategySignal
            'confidence': min(1.0, signal.get('confidence', 0) * final_score),
            'strength': mapped_strength,
            'metadata': {
                # Métadonnées originales
                **signal.get('metadata', {}),
                
                # Métadonnées de validation hiérarchiques
                'timeframe': signal.get('timeframe'),
                'reason': signal.get('reason'),
                'validation_timestamp': datetime.utcnow().isoformat(),
                'validation_score': final_score,
                'final_score': final_score * signal.get('confidence', 0),
                'aggregator_confidence': min(1.0, signal.get('confidence', 0) * final_score),
                
                # Nouvelle analyse hiérarchique
                'hierarchical_analysis': {
                    'critical_validators': level_analysis.get('critical', {}),
                    'important_validators': level_analysis.get('important', {}),
                    'standard_validators': level_analysis.get('standard', {}),
                    'combination_bonuses': combination_bonuses,
                    'veto_triggered': detailed_analysis.get('veto', False)
                },
                
                # Détails des validators pour compatibilité
                'validators_passed': sum(1 for r in validation_results if r['is_valid']),
                'total_validators': len(validation_results),
                'validation_details': [
                    {
                        'validator': result['validator_name'],
                        'valid': result['is_valid'],
                        'score': result['score'],
                        'importance': self.hierarchical_validator.get_validator_importance(result['validator_name'])
                    }
                    for result in validation_results
                ]
            }
        }
        
        return validated_signal
        
    def _extract_signal_price(self, signal: Dict[str, Any]) -> float:
        """
        Extrait le prix du signal pour le coordinator.
        
        Args:
            signal: Signal à traiter
            
        Returns:
            Prix du signal ou prix depuis le contexte
        """
        try:
            # Si le prix est déjà dans le signal
            if 'price' in signal and signal['price'] is not None:
                return float(signal['price'])
                
            # Essayer de récupérer depuis les métadonnées
            metadata = signal.get('metadata', {})
            if isinstance(metadata, dict) and 'current_price' in metadata:
                return float(metadata['current_price'])
                
            # Récupérer le prix depuis le contexte de marché
            symbol = signal.get('symbol')
            timeframe = signal.get('timeframe')
            
            # Si timeframe n'est pas direct, chercher dans les métadonnées
            if not timeframe:
                timeframe = metadata.get('timeframe')
            
            if symbol and timeframe and self.context_manager:
                context = self.context_manager.get_market_context(symbol, timeframe)
                
                # Prix depuis les données OHLCV
                ohlcv_data = context.get('ohlcv_data', [])
                if ohlcv_data and len(ohlcv_data) > 0:
                    latest_candle = ohlcv_data[-1]
                    return float(latest_candle.get('close', 0))
                    
                # Prix depuis les indicateurs
                indicators = context.get('indicators', {})
                if indicators and 'close' in indicators:
                    return float(indicators['close'])
                    
            # Valeur par défaut - récupérer depuis la DB directement
            if symbol:
                try:
                    # Essayer analyzer_data d'abord (plus récent)
                    if self.context_manager:
                        with self.context_manager.db_connection.cursor() as cursor:
                            cursor.execute("""
                                SELECT close FROM analyzer_data 
                                WHERE symbol = %s 
                                ORDER BY time DESC 
                                LIMIT 1
                            """, (symbol,))
                            
                            result = cursor.fetchone()
                            if result and result[0] is not None:
                                return float(result[0])
                    
                    # Fallback sur market_data
                    if self.context_manager:
                        with self.context_manager.db_connection.cursor() as cursor:
                            cursor.execute("""
                                SELECT close FROM market_data 
                                WHERE symbol = %s 
                                ORDER BY time DESC 
                                LIMIT 1
                            """, (symbol,))
                            
                            result = cursor.fetchone()
                            if result and result[0] is not None:
                                return float(result[0])
                                
                except Exception as e:
                    logger.warning(f"Erreur récupération prix DB pour {symbol}: {e}")
                    
            logger.warning(f"Prix non trouvé pour signal {symbol}, utilisation de 0.0")
            return 0.0
            
        except Exception as e:
            logger.error(f"Erreur extraction prix signal: {e}")
            return 0.0
        
    def _update_validation_stats_hierarchical(self, validation_results: List[Dict[str, Any]], 
                                            final_score: float, detailed_analysis: Dict[str, Any]):
        """
        Met à jour les statistiques de validation avec les nouvelles métriques hiérarchiques.
        
        Args:
            validation_results: Résultats de validation
            final_score: Score final
            detailed_analysis: Analyse hiérarchique détaillée
        """
        try:
            # Mise à jour de la moyenne des scores
            current_avg = self.stats['avg_validation_score']
            total_processed = self.stats['signals_processed']
            
            self.stats['avg_validation_score'] = (
                (current_avg * (total_processed - 1) + final_score) / total_processed
            )
            
            # Mise à jour des performances par validator avec importance
            for result in validation_results:
                validator_name = result['validator_name']
                if validator_name not in self.stats['validator_performance']:
                    self.stats['validator_performance'][validator_name] = {
                        'total_runs': 0,
                        'successful_validations': 0,
                        'avg_score': 0.0,
                        'importance': self.hierarchical_validator.get_validator_importance(validator_name)
                    }
                    
                perf = self.stats['validator_performance'][validator_name]
                perf['total_runs'] += 1
                
                if result['is_valid']:
                    perf['successful_validations'] += 1
                    
                # Mise à jour de la moyenne des scores pour ce validator
                current_avg = perf['avg_score']
                total_runs = perf['total_runs']
                perf['avg_score'] = (
                    (current_avg * (total_runs - 1) + result['score']) / total_runs
                )
                
        except Exception as e:
            logger.error(f"Erreur mise à jour stats: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de validation avec métriques hiérarchiques."""
        stats = self.stats.copy()
        
        # Ajouter des métriques hiérarchiques
        if stats['signals_processed'] > 0:
            stats['veto_rate'] = stats['signals_vetoed'] / stats['signals_processed']
            stats['validation_rate'] = stats['signals_validated'] / stats['signals_processed']
            stats['rejection_rate'] = stats['signals_rejected'] / stats['signals_processed']
        
        return stats
        
    def reset_stats(self):
        """Remet à zéro les statistiques."""
        self.stats = {
            'signals_processed': 0,
            'signals_validated': 0,
            'signals_rejected': 0,
            'signals_vetoed': 0,
            'validation_errors': 0,
            'avg_validation_score': 0.0,
            'validator_performance': {},
            'veto_reasons': {}
        }
        logger.info("Statistiques de validation hiérarchiques remises à zéro")