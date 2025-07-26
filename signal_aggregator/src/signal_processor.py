"""
Module de traitement et validation des signaux.
Contient la logique principale de validation, scoring et filtrage des signaux.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import json

logger = logging.getLogger(__name__)


class SignalProcessor:
    """Processeur principal pour la validation et le scoring des signaux."""
    
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
        
        # Configuration des seuils de validation
        self.min_validation_score = 0.6
        self.min_validators_passed = 3
        self.max_validators_failed = 10
        
        # Pondération des validators par catégorie
        self.validator_weights = {
            'trend': 1.5,      # Validators de tendance sont plus importants
            'volume': 1.3,     # Volume critique pour confirmation
            'structure': 1.2,   # Structure de marché importante
            'regime': 1.4,     # Régime de marché très important
            'volatility': 1.0,  # Volatilité standard
            'technical': 1.1    # Indicateurs techniques standard
        }
        
        # Statistiques de validation
        self.stats = {
            'signals_processed': 0,
            'signals_validated': 0,
            'signals_rejected': 0,
            'validation_errors': 0,
            'avg_validation_score': 0.0,
            'validator_performance': {}
        }
        
    async def process_signal(self, signal_data: str) -> Optional[Dict[str, Any]]:
        """
        Traite un signal ou un batch de signaux reçu depuis Redis.
        
        Args:
            signal_data: Données du signal au format JSON
            
        Returns:
            Signal validé et scoré ou None si rejeté
        """
        try:
            # Parsing du message
            message = json.loads(signal_data)
            
            # Vérifier si c'est un batch de signaux ou un signal individuel
            if isinstance(message, dict) and message.get('type') == 'signal_batch':
                # Traitement d'un batch de signaux
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
        Traite un signal individuel.
        
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
                
            # Validation complète avec contexte
            validated_signal = await self._validate_with_context(signal)
            
            return validated_signal
            
        except Exception as e:
            logger.error(f"Erreur traitement signal individuel: {e}")
            self.stats['validation_errors'] += 1
            return None
            
    async def _process_signal_batch(self, batch_message: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Traite un batch de signaux.
        
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
            
            validated_signals = []
            
            # Traiter chaque signal du batch
            for signal in signals:
                try:
                    validated_signal = await self._process_individual_signal(signal)
                    if validated_signal:
                        validated_signals.append(validated_signal)
                except Exception as e:
                    logger.error(f"Erreur traitement signal dans batch: {e}")
                    
            logger.info(f"Batch traité: {len(validated_signals)}/{len(signals)} signaux validés")
            
            return validated_signals if validated_signals else None
            
        except Exception as e:
            logger.error(f"Erreur traitement batch: {e}")
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
        
    async def _validate_with_context(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Valide le signal avec le contexte de marché complet.
        
        Args:
            signal: Signal à valider
            
        Returns:
            Signal validé et scoré ou None si rejeté
        """
        try:
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
                
            # Validation avec chaque validator
            validation_results = await self._run_all_validators(signal, context, validators)
            
            # Analyse des résultats de validation
            validation_summary = self._analyze_validation_results(validation_results)
            
            # Décision finale de validation
            is_valid, final_score = self._make_validation_decision(
                signal, validation_summary, validation_results
            )
            
            if is_valid:
                # Construction du signal validé
                validated_signal = self._build_validated_signal(
                    signal, validation_summary, validation_results, final_score
                )
                
                # Stockage en base de données si disponible
                if self.database_manager:
                    try:
                        signal_id = self.database_manager.store_validated_signal(validated_signal)
                        if signal_id:
                            # Ajouter l'ID en tant que métadonnée pour le coordinator
                            validated_signal['metadata']['db_id'] = signal_id
                            logger.info(f"Signal stocké en DB avec ID: {signal_id}, métadonnées: {validated_signal['metadata'].get('db_id', 'MANQUANT')}")
                    except Exception as e:
                        logger.error(f"Erreur stockage signal en DB: {e}")
                        # Continuer même si le stockage échoue
                
                self.stats['signals_validated'] += 1
                self._update_validation_stats(validation_results, final_score)
                
                logger.info(f"Signal VALIDÉ: {signal['strategy']} {symbol} {timeframe} "
                          f"{signal['side']} (score={final_score:.2f})")
                
                return validated_signal
            else:
                self.stats['signals_rejected'] += 1
                
                logger.info(f"Signal REJETÉ: {signal['strategy']} {symbol} {timeframe} "
                          f"{signal['side']} (score={final_score:.2f})")
                
                return None
                
        except Exception as e:
            logger.error(f"Erreur validation avec contexte: {e}")
            self.stats['validation_errors'] += 1
            return None
            
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
                # Instanciation du validator
                validator = validator_class(
                    symbol=signal['symbol'],
                    data=context,
                    context=context.get('indicators', {})
                )
                
                # Validation
                is_valid = validator.validate_signal(signal)
                score = validator.get_validation_score(signal)
                reason = validator.get_validation_reason(signal, is_valid)
                
                # Détermination de la catégorie du validator
                category = self._get_validator_category(validator_name)
                weight = self.validator_weights.get(category, 1.0)
                
                result = {
                    'validator_name': validator_name,
                    'category': category,
                    'is_valid': is_valid,
                    'score': score,
                    'weighted_score': score * weight,
                    'weight': weight,
                    'reason': reason,
                    'execution_time': 0  # TODO: Mesurer le temps d'exécution
                }
                
                validation_results.append(result)
                
                logger.debug(f"Validator {validator_name}: {'PASS' if is_valid else 'FAIL'} "
                           f"(score={score:.2f}, weighted={result['weighted_score']:.2f})")
                
            except Exception as e:
                logger.error(f"Erreur validator {validator_name}: {e}")
                
                # Ajouter un résultat d'erreur
                validation_results.append({
                    'validator_name': validator_name,
                    'category': 'error',
                    'is_valid': False,
                    'score': 0.0,
                    'weighted_score': 0.0,
                    'weight': 1.0,
                    'reason': f"ERREUR: {str(e)}",
                    'execution_time': 0
                })
                
        return validation_results
        
    def _get_validator_category(self, validator_name: str) -> str:
        """
        Détermine la catégorie d'un validator basée sur son nom.
        
        Args:
            validator_name: Nom du validator
            
        Returns:
            Catégorie du validator
        """
        name_lower = validator_name.lower()
        
        if 'trend' in name_lower or 'adx' in name_lower or 'macd' in name_lower:
            return 'trend'
        elif 'volume' in name_lower or 'vwap' in name_lower:
            return 'volume'
        elif 'structure' in name_lower or 'pivot' in name_lower or 'level' in name_lower:
            return 'structure'
        elif 'regime' in name_lower or 'market' in name_lower:
            return 'regime'
        elif 'volatility' in name_lower or 'atr' in name_lower or 'bollinger' in name_lower:
            return 'volatility'
        else:
            return 'technical'
            
    def _analyze_validation_results(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyse les résultats de validation pour créer un résumé.
        
        Args:
            validation_results: Liste des résultats de validation
            
        Returns:
            Résumé des résultats de validation
        """
        total_validators = len(validation_results)
        validators_passed = sum(1 for result in validation_results if result['is_valid'])
        validators_failed = total_validators - validators_passed
        
        # Calcul des scores
        raw_scores = [result['score'] for result in validation_results]
        weighted_scores = [result['weighted_score'] for result in validation_results]
        total_weights = sum(result['weight'] for result in validation_results)
        
        avg_raw_score = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
        avg_weighted_score = sum(weighted_scores) / total_weights if total_weights > 0 else 0.0
        
        # Analyse par catégorie
        category_results = {}
        for result in validation_results:
            category = result['category']
            if category not in category_results:
                category_results[category] = {
                    'passed': 0,
                    'total': 0,
                    'scores': [],
                    'weight': result['weight']
                }
            
            category_results[category]['total'] += 1
            category_results[category]['scores'].append(result['score'])
            if result['is_valid']:
                category_results[category]['passed'] += 1
                
        # Calcul du score par catégorie
        for category, data in category_results.items():
            data['pass_rate'] = data['passed'] / data['total'] if data['total'] > 0 else 0
            data['avg_score'] = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
            
        summary = {
            'total_validators': total_validators,
            'validators_passed': validators_passed,
            'validators_failed': validators_failed,
            'pass_rate': validators_passed / total_validators if total_validators > 0 else 0,
            'avg_raw_score': avg_raw_score,
            'avg_weighted_score': avg_weighted_score,
            'category_results': category_results,
            'validation_strength': self._calculate_validation_strength(category_results)
        }
        
        return summary
        
    def _calculate_validation_strength(self, category_results: Dict[str, Any]) -> str:
        """
        Calcule la force de validation basée sur les catégories.
        
        Args:
            category_results: Résultats par catégorie
            
        Returns:
            Force de validation ('strong', 'moderate', 'weak')
        """
        try:
            # Critères pour une validation forte
            critical_categories = ['trend', 'regime', 'volume']
            strong_validation = True
            
            for category in critical_categories:
                if category in category_results:
                    pass_rate = category_results[category]['pass_rate']
                    if pass_rate < 0.6:  # Moins de 60% de réussite dans cette catégorie
                        strong_validation = False
                        break
                        
            if strong_validation and len(category_results) >= 4:
                return 'strong'
            elif len([cat for cat, data in category_results.items() if data['pass_rate'] > 0.5]) >= 3:
                return 'moderate'
            else:
                return 'weak'
                
        except Exception as e:
            logger.error(f"Erreur calcul force validation: {e}")
            return 'weak'
            
    def _make_validation_decision(self, signal: Dict[str, Any], summary: Dict[str, Any], 
                                 validation_results: List[Dict[str, Any]]) -> Tuple[bool, float]:
        """
        Prend la décision finale de validation.
        
        Args:
            signal: Signal original
            summary: Résumé de validation
            validation_results: Résultats détaillés
            
        Returns:
            Tuple (is_valid, final_score)
        """
        try:
            # Critères de base
            meets_min_validators = summary['validators_passed'] >= self.min_validators_passed
            meets_min_score = summary['avg_weighted_score'] >= self.min_validation_score
            
            # Critères additionnels
            strong_validation = summary['validation_strength'] == 'strong'
            high_confidence_signal = signal['confidence'] > 0.7
            
            # Bonus pour signaux haute qualité
            final_score = summary['avg_weighted_score']
            
            if strong_validation:
                final_score *= 1.1  # Bonus 10% pour validation forte
                
            if high_confidence_signal:
                final_score *= 1.05  # Bonus 5% pour signal haute confiance
                
            # Pénalité pour trop d'échecs
            if summary['validators_failed'] > self.max_validators_failed:
                final_score *= 0.9  # Pénalité 10%
                
            # Limiter le score à 1.0
            final_score = min(1.0, final_score)
            
            # Décision finale
            is_valid = meets_min_validators and meets_min_score
            
            return is_valid, final_score
            
        except Exception as e:
            logger.error(f"Erreur décision validation: {e}")
            return False, 0.0
            
    def _build_validated_signal(self, signal: Dict[str, Any], summary: Dict[str, Any], 
                               validation_results: List[Dict[str, Any]], final_score: float) -> Dict[str, Any]:
        """
        Construit le signal validé final.
        
        Args:
            signal: Signal original
            summary: Résumé de validation
            validation_results: Résultats détaillés
            final_score: Score final
            
        Returns:
            Signal validé complet
        """
        validated_signal = signal.copy()
        
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
        
        # Construction du signal conforme au schéma StrategySignal
        validated_signal = {
            # Champs obligatoires pour StrategySignal
            'strategy': signal['strategy'],
            'symbol': signal['symbol'],
            'side': signal['side'],  # Déjà en format "BUY"/"SELL"
            'timestamp': signal['timestamp'],  # Format ISO string
            'price': price,  # Champ requis par le coordinator
            
            # Champs optionnels pour StrategySignal
            'confidence': min(1.0, signal.get('confidence', 0) * final_score),
            'strength': mapped_strength,
            'metadata': {
                # Métadonnées originales
                **signal.get('metadata', {}),
                
                # Métadonnées de validation ajoutées
                'timeframe': signal.get('timeframe'),
                'reason': signal.get('reason'),
                'validation_timestamp': datetime.utcnow().isoformat(),
                'validation_score': final_score,
                'raw_validation_score': summary['avg_raw_score'],
                'weighted_validation_score': summary['avg_weighted_score'],
                'validators_passed': summary['validators_passed'],
                'total_validators': summary['total_validators'],
                'validation_strength': summary['validation_strength'],
                'pass_rate': summary['pass_rate'],
                'aggregator_confidence': min(1.0, signal.get('confidence', 0) * final_score),
                'final_score': final_score * signal.get('confidence', 0),
                'category_scores': {
                    cat: data['avg_score'] 
                    for cat, data in summary['category_results'].items()
                },
                'validation_details': [
                    {
                        'validator': result['validator_name'],
                        'valid': result['is_valid'],
                        'score': result['score'],
                        'category': result['category']
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
                    
            # Valeur par défaut - récupérer depuis la DB
            if symbol and self.context_manager:
                try:
                    with self.context_manager.db_connection.cursor() as cursor:
                        cursor.execute("""
                            SELECT close FROM market_data 
                            WHERE symbol = %s 
                            ORDER BY time DESC 
                            LIMIT 1
                        """, (symbol,))
                        
                        result = cursor.fetchone()
                        if result:
                            return float(result[0])
                except Exception as e:
                    logger.warning(f"Erreur récupération prix DB pour {symbol}: {e}")
                    
            logger.warning(f"Prix non trouvé pour signal {symbol}, utilisation de 0.0")
            return 0.0
            
        except Exception as e:
            logger.error(f"Erreur extraction prix signal: {e}")
            return 0.0
        
    def _update_validation_stats(self, validation_results: List[Dict[str, Any]], final_score: float):
        """
        Met à jour les statistiques de validation.
        
        Args:
            validation_results: Résultats de validation
            final_score: Score final
        """
        try:
            # Mise à jour de la moyenne des scores
            current_avg = self.stats['avg_validation_score']
            total_processed = self.stats['signals_processed']
            
            self.stats['avg_validation_score'] = (
                (current_avg * (total_processed - 1) + final_score) / total_processed
            )
            
            # Mise à jour des performances par validator
            for result in validation_results:
                validator_name = result['validator_name']
                if validator_name not in self.stats['validator_performance']:
                    self.stats['validator_performance'][validator_name] = {
                        'total_runs': 0,
                        'successful_validations': 0,
                        'avg_score': 0.0
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
        """Retourne les statistiques de validation."""
        return self.stats.copy()
        
    def reset_stats(self):
        """Remet à zéro les statistiques."""
        self.stats = {
            'signals_processed': 0,
            'signals_validated': 0,
            'signals_rejected': 0,
            'validation_errors': 0,
            'avg_validation_score': 0.0,
            'validator_performance': {}
        }
        logger.info("Statistiques de validation remises à zéro")