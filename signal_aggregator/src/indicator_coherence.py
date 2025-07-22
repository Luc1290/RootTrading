"""
Module de vérification de cohérence entre indicateurs
S'assure que RSI et MACD sont alignés pour confirmer les signaux
"""
import logging
from typing import Dict, Tuple
import sys
import os

# Add path to shared modules BEFORE imports
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.src.config import MACD_HISTOGRAM_WEAK
from shared.src.volume_context_detector import volume_context_detector
try:
    from .shared.technical_utils import VolumeAnalyzer
except ImportError:
    # Fallback pour l'exécution dans le conteneur
    sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))
    from technical_utils import VolumeAnalyzer

logger = logging.getLogger(__name__)

class IndicatorCoherenceValidator:
    """
    Vérifie la cohérence entre différents indicateurs techniques
    """
    
    def __init__(self):
        # SEUILS STANDARDISÉS - Version harmonisée avec tous les modules
        self.rsi_buy_threshold = 40   # STANDARDISÉ: RSI minimum pour BUY (début pump)
        self.rsi_sell_threshold = 65  # STANDARDISÉ: RSI minimum pour SELL (fin pump)
        self.macd_threshold = MACD_HISTOGRAM_WEAK  # STANDARDISÉ: Momentum faible minimum
        
        # Poids des indicateurs pour le score de cohérence
        # BUY privilégie MACD et EMA (pour acheter en début de pump)
        # SELL privilégie RSI et position (pour vendre en fin de pump)
        self.indicator_weights = {
            'BUY': {
                'rsi': 0.30,       # Important pour débuts pump
                'macd': 0.35,      # Momentum important
                'ema_alignment': 0.25,  # Tendance émergente
                'volume': 0.10
            },
            'SELL': {
                'rsi': 0.45,       # Crucial pour fins pump
                'macd': 0.15,      # Moins important en fin pump
                'ema_alignment': 0.10,  # Moins important en fin pump
                'volume': 0.30     # Volume important pour confirmer essoufflement
            }
        }
    
    def validate_signal_coherence(self, signal_side: str, indicators: Dict) -> Tuple[bool, float, str]:
        """
        Valide la cohérence d'un signal avec les indicateurs
        
        Args:
            signal_side: 'BUY' ou 'SELL'
            indicators: Dict des indicateurs techniques
            
        Returns:
            (is_coherent, coherence_score, reason)
        """
        try:
            # NOUVEAU: Vérifier la tendance récente AVANT tout
            if signal_side == 'BUY':
                trend_check = self._check_recent_trend(indicators)
                if not trend_check['is_safe']:
                    return False, 0.0, trend_check['reason']
            
            coherence_score = 0.0
            reasons = []
            
            # Utiliser les poids spécifiques au signal
            weights = self.indicator_weights[signal_side]
            
            # Vérifier RSI
            rsi_score, rsi_reason = self._check_rsi_coherence(signal_side, indicators)
            coherence_score += rsi_score * weights['rsi']
            if rsi_reason:
                reasons.append(rsi_reason)
            
            # Vérifier MACD
            macd_score, macd_reason = self._check_macd_coherence(signal_side, indicators)
            coherence_score += macd_score * weights['macd']
            if macd_reason:
                reasons.append(macd_reason)
            
            # Vérifier alignement EMA
            ema_score, ema_reason = self._check_ema_alignment(signal_side, indicators)
            coherence_score += ema_score * weights['ema_alignment']
            if ema_reason:
                reasons.append(ema_reason)
            
            # Vérifier volume
            volume_score, volume_reason = self._check_volume_confirmation(signal_side, indicators)
            coherence_score += volume_score * weights['volume']
            if volume_reason:
                reasons.append(volume_reason)
            
            # Seuil plus bas pour SELL (0.45) pour permettre les ventes au sommet
            # Seuil normal pour BUY (0.55)
            threshold = 0.45 if signal_side == 'SELL' else 0.55
            is_coherent = coherence_score > threshold
            
            reason = " | ".join(reasons) if reasons else "Cohérence validée"
            
            if is_coherent:
                logger.info(f"✅ Cohérence validée (score: {coherence_score:.2f}): {reason}")
            else:
                logger.warning(f"❌ Cohérence faible (score: {coherence_score:.2f}): {reason}")
            
            return is_coherent, coherence_score, reason
            
        except Exception as e:
            logger.error(f"Erreur validation cohérence: {e}")
            return True, 0.5, "Erreur validation - signal autorisé par défaut"
    
    def _check_rsi_coherence(self, signal_side: str, indicators: Dict) -> Tuple[float, str]:
        """
        Vérifie la cohérence du RSI avec le signal
        
        Returns:
            (score, reason): score 0-1, reason si problème
        """
        try:
            rsi = indicators.get('rsi_14')
            if rsi is None:
                return 0.5, "RSI non disponible"
            
            rsi_val = float(rsi)
            
            if signal_side == 'BUY':
                # BUY = REBOND après oversold, PAS pendant la chute !
                if rsi_val < 20:
                    return 0.1, "RSI extrême, chute en cours"  # DANGER - ne PAS acheter
                elif rsi_val < 30:
                    return 0.2, "RSI oversold, attendre stabilisation"  # Attendre rebond
                elif rsi_val < 35:
                    return 0.5, "RSI remonte, surveiller"  # Début de rebond possible
                elif rsi_val < 40:
                    return 0.8, ""  # BON - rebond confirmé
                elif rsi_val < 50:
                    return 1.0, ""  # EXCELLENT - momentum haussier
                elif rsi_val < 60:
                    return 0.7, ""  # Acceptable mais prudence
                elif rsi_val < 70:
                    return 0.4, "RSI élevé, entrée tardive"
                else:
                    return 0.1, "RSI trop élevé, trop tard"
            
            else:  # SELL
                # SELL = fin pump (RSI haut nécessaire)
                if rsi_val > 80:
                    return 1.0, ""  # Excellent - surachat extrême (fin pump)
                elif rsi_val > 75:
                    return 0.9, ""  # Très bon - zone de vente optimale
                elif rsi_val > 70:
                    return 0.8, ""  # Bon - début fin pump
                elif rsi_val > 65:
                    return 0.7, ""  # Acceptable - pump qui monte
                elif rsi_val > 60:
                    return 0.5, ""  # Modéré - pump pas terminé
                elif rsi_val > 40:  # STANDARDISÉ: 40 pour cohérence avec tous les modules
                    return 0.3, "RSI trop bas, pump pas fini"
                else:
                    return 0.1, "RSI trop bas, pas de pump"
                    
        except Exception as e:
            logger.error(f"Erreur check RSI: {e}")
            return 0.5, "Erreur RSI"
    
    def _check_macd_coherence(self, signal_side: str, indicators: Dict) -> Tuple[float, str]:
        """
        Vérifie la cohérence du MACD avec le signal
        """
        try:
            macd_line = indicators.get('macd_line')
            macd_signal = indicators.get('macd_signal')
            macd_histogram = indicators.get('macd_histogram')
            
            if not all([macd_line is not None, macd_signal is not None, macd_histogram is not None]):
                return 0.5, "MACD incomplet"
            
            # Assertions pour mypy
            assert macd_line is not None
            assert macd_signal is not None
            assert macd_histogram is not None
            
            macd_line_val = float(macd_line)
            macd_signal_val = float(macd_signal)
            macd_histogram_val = float(macd_histogram)
            
            if signal_side == 'BUY':
                # Pour BUY: MACD line > signal line et histogram > 0
                if macd_line_val > macd_signal_val and macd_histogram_val > 0:
                    return 1.0, ""  # Parfait
                elif macd_line_val > macd_signal_val:
                    return 0.8, ""  # Bon - crossover bullish
                elif macd_histogram_val > 0:
                    return 0.6, ""  # Moyen - momentum positif
                elif abs(macd_histogram_val) < self.macd_threshold:
                    return 0.5, ""  # Neutre - potentiel retournement
                else:
                    return 0.3, ""  # MACD baissier mais acceptable
            
            else:  # SELL
                # Pour SELL: fin pump, MACD commence à s'essouffler
                if macd_line_val < macd_signal_val and macd_histogram_val < 0:
                    return 1.0, ""  # Parfait - momentum s'inverse (fin pump)
                elif macd_histogram_val < 0:
                    return 0.9, ""  # Très bon - momentum négatif (essoufflement)
                elif abs(macd_histogram_val) < self.macd_threshold:
                    return 0.8, ""  # Bon - momentum neutre (fin pump proche)
                elif macd_line_val < macd_signal_val:
                    return 0.7, ""  # Crossover bearish (début essoufflement)
                else:
                    # MACD encore haussier = pump pas terminé
                    return 0.4, "MACD haussier, pump pas terminé"
                    
        except Exception as e:
            logger.error(f"Erreur check MACD: {e}")
            return 0.5, "Erreur MACD"
    
    def _check_ema_alignment(self, signal_side: str, indicators: Dict) -> Tuple[float, str]:
        """
        Vérifie l'alignement des EMAs avec le signal
        """
        try:
            # MIGRATION BINANCE: Utilisation directe des nouvelles EMAs 7/26/99
            ema_7 = indicators.get('ema_7')
            ema_26 = indicators.get('ema_26')
            ema_99 = indicators.get('ema_99')
            
            if not all([ema_7 is not None, ema_26 is not None, ema_99 is not None]):
                return 0.5, "EMAs incomplètes"
            
            # Assertions pour mypy
            assert ema_7 is not None
            assert ema_26 is not None
            assert ema_99 is not None
            
            ema_7_val = float(ema_7)
            ema_26_val = float(ema_26)
            ema_99_val = float(ema_99)
            
            if signal_side == 'BUY':
                # Pour BUY: EMA7 > EMA26 > EMA99 (bullish alignment) - CONFIG BINANCE
                if ema_7_val > ema_26_val > ema_99_val:
                    return 1.0, ""  # Parfait
                elif ema_7_val > ema_26_val:
                    return 0.7, ""  # Bon - trend court terme haussier
                elif ema_7_val > ema_99_val:
                    return 0.5, ""  # Moyen
                else:
                    return 0.3, ""  # EMAs baissières mais acceptable pour début de pump
            
            else:  # SELL
                # Pour SELL: EMAs encore haussières mais price doit être au-dessus - CONFIG BINANCE
                if ema_7_val < ema_26_val < ema_99_val:
                    return 1.0, ""  # Parfait - tendance s'inverse (fin pump confirmée)
                elif ema_7_val < ema_26_val:
                    return 0.9, ""  # Très bon - début retournement
                elif ema_7_val > ema_26_val > ema_99_val:
                    # EMAs haussières = pump encore en cours, mais acceptable près du sommet
                    return 0.6, ""  # Acceptable si RSI haut
                else:
                    return 0.5, ""  # Configuration mixte
                    
        except Exception as e:
            logger.error(f"Erreur check EMA: {e}")
            return 0.5, "Erreur EMA"
    
    def _check_volume_confirmation(self, signal_side: str, indicators: Dict) -> Tuple[float, str]:
        """
        Vérifie la confirmation du volume avec adaptation contextuelle
        """
        try:
            volume_ratio = indicators.get('volume_ratio')
            volume_trend = indicators.get('volume_trend')
            
            if volume_ratio is None:
                return 0.5, "Volume non disponible"
            
            volume_ratio_val = float(volume_ratio)
            
            # Récupération des indicateurs pour contexte
            rsi = indicators.get('rsi')
            cci = indicators.get('cci')
            adx = indicators.get('adx')
            
            # Détection du contexte market et seuils adaptatifs
            contextual_threshold, context_name, contextual_score = volume_context_detector.get_contextual_volume_threshold(
                base_volume_ratio=volume_ratio_val,
                rsi=rsi,
                cci=cci,
                adx=adx,
                signal_type=signal_side
            )
            
            # Qualité du volume selon le contexte
            volume_quality = volume_context_detector.get_volume_quality_description(
                volume_ratio_val, context_name
            )
            
            # Analyse de la tendance volume pour bonus/malus
            volume_trend_bonus = 0.0
            volume_trend_info = ""
            if volume_trend is not None:
                trend_val = float(volume_trend)
                if signal_side == 'BUY' and trend_val > 0.1:  # Volume croissant pour BUY
                    volume_trend_bonus = 0.1
                    volume_trend_info = " (tendance volume ↗️)"
                elif signal_side == 'SELL' and trend_val > 0.1:  # Volume croissant pour SELL aussi bon
                    volume_trend_bonus = 0.1
                    volume_trend_info = " (tendance volume ↗️)"
                elif trend_val < -0.1:  # Volume décroissant = malus
                    volume_trend_bonus = -0.05
                    volume_trend_info = " (tendance volume ↘️)"
            
            if signal_side == 'BUY':
                # Pour BUY, utiliser les seuils avec utilitaire partagé
                base_boost = VolumeAnalyzer.calculate_volume_boost(volume_ratio_val)
                _, description = VolumeAnalyzer.analyze_volume_strength({'volume_ratio': volume_ratio_val})
                
                # Ajuster avec bonus tendance et seuil contextuel
                if volume_ratio_val >= contextual_threshold:
                    score = min(1.0, base_boost + volume_trend_bonus)
                    return score, f"{description} - Contexte: {context_name}{volume_trend_info}"
                else:
                    # En dessous du seuil contextuel
                    penalty_score = min(1.0, max(0.3, (volume_ratio_val / contextual_threshold) * 0.5 + volume_trend_bonus))
                    return penalty_score, f"Volume insuffisant pour {context_name} (seuil: {contextual_threshold:.2f}){volume_trend_info}"
            
            else:  # SELL
                # Pour SELL, utiliser utilitaire partagé avec logique adaptée
                _, description = VolumeAnalyzer.analyze_volume_strength({'volume_ratio': volume_ratio_val})
                
                # Logique SELL: volume élevé peut continuer pump, faible = essoufflement
                if volume_ratio_val > 2.5:
                    score = min(1.0, 0.7 + volume_trend_bonus)
                    return score, f"{description}, pump peut continuer - Contexte: {context_name}{volume_trend_info}"
                elif volume_ratio_val > 1.0:
                    base_boost = VolumeAnalyzer.calculate_volume_boost(volume_ratio_val)
                    score = min(1.0, base_boost + volume_trend_bonus)
                    return score, f"{description} - Contexte: {context_name}{volume_trend_info}"
                else:
                    score = min(1.0, 0.9 + volume_trend_bonus)
                    return score, f"{description} = essoufflement - Contexte: {context_name}{volume_trend_info}"
                
        except Exception as e:
            logger.error(f"Erreur check volume contextuel: {e}")
            return 0.5, "Erreur volume contextuel"
    
    def _check_recent_trend(self, indicators: Dict) -> Dict:
        """
        NOUVEAU: Vérifie la tendance récente pour éviter d'acheter pendant une chute
        CORRIGÉ: Utilise les vrais noms d'indicateurs disponibles
        
        Returns:
            {'is_safe': bool, 'reason': str}
        """
        try:
            # CORRECTION: Récupérer les indicateurs critiques avec les bons noms
            momentum = indicators.get('momentum_10')  # Disponible dans metadata
            rsi = indicators.get('rsi_14') or indicators.get('rsi')  # Essayer les deux
            volume_ratio = indicators.get('volume_ratio')  # Disponible dans metadata
            
            logger.info(f"🔍 Vérification tendance: momentum={momentum}, rsi={rsi}, volume={volume_ratio}")
            
            # CONDITIONS DE DANGER - Ne PAS acheter si:
            
            # 1. Volume très faible (signal principal de danger)
            if volume_ratio and volume_ratio < 0.8:  # Très strict sur le volume
                return {
                    'is_safe': False,
                    'reason': f"Volume trop faible ({volume_ratio:.2f}) - signal douteux"
                }
            
            # 2. Momentum négatif (chute active) - Seuil moins strict
            if momentum and momentum < -0.5:  # Réduit de -1.0 à -0.5
                return {
                    'is_safe': False,
                    'reason': f"Momentum négatif ({momentum:.2f}) - chute en cours"
                }
            
            # 3. RSI < 30 ET momentum négatif (oversold en chute)
            if rsi and momentum:
                if rsi < 30 and momentum < -0.2:  # Seuil momentum réduit
                    return {
                        'is_safe': False,
                        'reason': f"RSI oversold ({rsi:.1f}) avec momentum négatif - attendre rebond"
                    }
            
            # 4. Combinaison dangereuse: momentum très négatif + volume faible
            if momentum and volume_ratio:
                if momentum < -0.7 and volume_ratio < 1.0:
                    return {
                        'is_safe': False,
                        'reason': f"Chute ({momentum:.2f}) + volume faible ({volume_ratio:.2f})"
                    }
            
            # CONDITIONS DE SÉCURITÉ - OK pour acheter si:
            
            # Volume acceptable et momentum pas trop négatif
            if volume_ratio and volume_ratio >= 1.0 and (not momentum or momentum > -0.3):
                return {
                    'is_safe': True,
                    'reason': f"Volume OK ({volume_ratio:.2f}) et momentum acceptable"
                }
            
            # RSI oversold mais momentum qui s'améliore
            if rsi and rsi < 35 and momentum and momentum > -0.3:
                return {
                    'is_safe': True,
                    'reason': f"RSI oversold ({rsi:.1f}) mais momentum stabilisé"
                }
            
            # Par défaut, être PRUDENT (changement)
            return {
                'is_safe': False,
                'reason': f"Conditions insuffisantes (mom={momentum}, vol={volume_ratio})"
            }
            
        except Exception as e:
            logger.error(f"Erreur vérification tendance récente: {e}")
            # En cas d'erreur, être très prudent
            return {
                'is_safe': False,
                'reason': f"Erreur vérification tendance: {e}"
            }
    
    def get_coherence_requirements(self, signal_side: str) -> Dict:
        """
        Retourne les exigences de cohérence pour un type de signal
        """
        if signal_side == 'BUY':
            return {
                'rsi': 'RSI < 40 (idéalement < 25)',  # STANDARDISÉ
                'macd': f'MACD line > signal line, histogram > {MACD_HISTOGRAM_WEAK}',  # STANDARDISÉ
                'ema': 'EMA7 > EMA26 > EMA99',  # MIGRATION BINANCE
                'volume': 'Volume contextuel adaptatif (0.6-2.0 selon RSI/CCI/ADX)'  # NOUVEAU: Contextuel
            }
        else:  # SELL
            return {
                'rsi': 'RSI > 65 (idéalement > 75)',  # STANDARDISÉ
                'macd': f'MACD line < signal line, histogram < -{MACD_HISTOGRAM_WEAK}',  # STANDARDISÉ
                'ema': 'EMA7 < EMA26 < EMA99',  # MIGRATION BINANCE
                'volume': 'Volume contextuel adaptatif (1.0-2.5 selon contexte)'  # NOUVEAU: Contextuel
            }