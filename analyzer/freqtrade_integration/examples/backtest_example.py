"""
Exemple de backtesting d'une stratégie ROOT avec Freqtrade.

Ce script démontre comment:
1. Charger une stratégie ROOT existante
2. La convertir en stratégie Freqtrade
3. Exécuter un backtest avec Freqtrade
4. Analyser les résultats

Prérequis:
- pip install freqtrade
- freqtrade setup pour télécharger données historiques
"""

import logging
import os
import sys
from pathlib import Path

from analyzer.freqtrade_integration import RootToFreqtradeAdapter
from analyzer.strategies.EMA_Cross_Strategy import EMA_Cross_Strategy

# Ajouter les paths nécessaires
analyzer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, analyzer_root)


# Configurer logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def backtest_root_strategy_simple():
    """
    Exemple simple: Convertir stratégie ROOT et l'exporter.

    Note: Pour exécuter le backtest, vous devez utiliser
    la CLI Freqtrade avec le fichier exporté.
    """
    logger.info("=== Exemple Backtesting Simple ===\n")

    # 1. Sélectionner stratégie ROOT à tester
    logger.info("1. Chargement stratégie ROOT: EMA_Cross_Strategy")
    root_strategy = EMA_Cross_Strategy

    # 2. Convertir vers Freqtrade
    logger.info("2. Conversion ROOT → Freqtrade")
    adapter = RootToFreqtradeAdapter(root_strategy)
    freqtrade_strategy = adapter.convert()

    logger.info(f"   ✓ Stratégie convertie: {freqtrade_strategy.__name__}")

    # 3. Exporter vers fichier
    output_dir = Path(__file__).parent / "strategies_freqtrade"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{root_strategy.__name__}_freqtrade.py"
    adapter.export_to_file(str(output_file))

    logger.info(f"   ✓ Exportée vers: {output_file}")

    # 4. Instructions pour backtesting
    logger.info("\n3. Pour exécuter le backtest avec Freqtrade CLI:")
    logger.info("   " + "─" * 60)
    logger.info(
        f"""
   # Télécharger données historiques (si pas déjà fait)
   freqtrade download-data \\
       --exchange binance \\
       --pairs BTC/USDT ETH/USDT BNB/USDT \\
       --timeframe 5m \\
       --days 30

   # Exécuter backtest
   freqtrade backtesting \\
       --strategy {freqtrade_strategy.__name__} \\
       --strategy-path {output_dir.absolute()} \\
       --timeframe 5m \\
       --timerange 20240101-20240131

   # Voir résultats détaillés
   freqtrade backtesting-analysis
    """
    )
    logger.info("   " + "─" * 60)

    return output_file


def backtest_with_config():
    """
    Exemple avancé: Générer configuration Freqtrade complète.
    """
    logger.info("\n=== Exemple Backtesting Avancé ===\n")

    # Configuration Freqtrade
    config = {
        "strategy": "EMA_Cross_Strategy_Freqtrade",
        "strategy_path": "./strategies_freqtrade",
        "exchange": {
            "name": "binance",
            "pair_whitelist": [
                "BTC/USDT",
                "ETH/USDT",
                "BNB/USDT",
                "SOL/USDT",
                "ADA/USDT",
            ],
            "pair_blacklist": [],
        },
        "timeframe": "5m",
        "stake_currency": "USDT",
        "stake_amount": 100,
        "dry_run": True,
        "max_open_trades": 3,
        "minimal_roi": {"0": 0.10, "30": 0.05, "60": 0.02, "120": 0.01},
        "stoploss": -0.05,
        "trailing_stop": True,
        "trailing_stop_positive": 0.01,
        "trailing_stop_positive_offset": 0.02,
        "trailing_only_offset_is_reached": True,
    }

    # Exporter config
    output_dir = Path(__file__).parent / "freqtrade_configs"
    output_dir.mkdir(exist_ok=True)

    config_file = output_dir / "backtest_config.json"

    import json

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info(f"✓ Configuration générée: {config_file}")

    logger.info("\nPour utiliser cette config:")
    logger.info(
        f"""
   freqtrade backtesting \\
       --config {config_file.absolute()} \\
       --timerange 20240101-20240131
    """
    )

    return config_file


def batch_backtest_all_strategies():
    """
    Exemple batch: Convertir toutes les stratégies ROOT et générer script de backtest.
    """
    logger.info("\n=== Exemple Backtesting Batch ===\n")

    from analyzer.freqtrade_integration import BatchConverter

    # Convertir toutes les stratégies
    converter = BatchConverter()
    stats = converter.convert_all_root_to_freqtrade()

    logger.info(f"\nConversion terminée:")
    logger.info(f"  Succès: {stats['success']}")
    logger.info(f"  Échecs: {stats['failed']}")

    # Générer script bash pour backtester toutes les stratégies
    output_dir = Path(__file__).parent / "batch_backtest"
    output_dir.mkdir(exist_ok=True)

    script_file = output_dir / "backtest_all.sh"

    with open(script_file, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# Script de backtesting batch pour toutes les stratégies ROOT\n")
        f.write("# Généré automatiquement\n\n")

        strategies_dir = (
            Path(analyzer_root)
            / "analyzer/freqtrade_integration/converted_strategies/freqtrade"
        )

        f.write(f'STRATEGY_PATH="{strategies_dir.absolute()}"\n')
        f.write('TIMERANGE="20240101-20240131"\n')
        f.write('TIMEFRAME="5m"\n\n')

        # Pour chaque stratégie convertie
        if strategies_dir.exists():
            for strategy_file in strategies_dir.glob("*_freqtrade.py"):
                strategy_name = strategy_file.stem

                f.write(f'echo "Backtesting {strategy_name}..."\n')
                f.write(f"freqtrade backtesting \\\n")
                f.write(f"    --strategy {strategy_name} \\\n")
                f.write(f"    --strategy-path $STRATEGY_PATH \\\n")
                f.write(f"    --timeframe $TIMEFRAME \\\n")
                f.write(f"    --timerange $TIMERANGE\n\n")

    logger.info(f"✓ Script batch généré: {script_file}")
    logger.info(f"\nPour exécuter: chmod +x {script_file} && {script_file}")

    return script_file


def analyze_backtest_results():
    """
    Exemple: Analyser résultats de backtest Freqtrade.
    """
    logger.info("\n=== Analyse Résultats Backtest ===\n")

    logger.info("Après avoir exécuté un backtest, analysez les résultats avec:")
    logger.info(
        """
   # Voir résumé
   freqtrade backtesting-show

   # Analyse détaillée des trades
   freqtrade backtesting-analysis

   # Générer plots graphiques
   freqtrade plot-dataframe \\
       --strategy EMA_Cross_Strategy_Freqtrade \\
       --indicators1 ema_12,ema_26,ema_50 \\
       --indicators2 rsi,macd_line

   # Exporter résultats en JSON
   freqtrade backtesting \\
       --strategy EMA_Cross_Strategy_Freqtrade \\
       --export trades
    """
    )

    logger.info("\nFichiers de résultats typiques:")
    logger.info("  - user_data/backtest_results/backtest-result-*.json")
    logger.info("  - user_data/backtest_results/strategy_*.json")


def compare_strategies():
    """
    Exemple: Comparer performance de plusieurs stratégies ROOT.
    """
    logger.info("\n=== Comparaison Stratégies ===\n")

    strategies_to_compare = [
        "EMA_Cross_Strategy",
        "MACD_Crossover_Strategy",
        "RSI_Cross_Strategy",
        "Bollinger_Touch_Strategy",
    ]

    logger.info("Stratégies à comparer:")
    for strat in strategies_to_compare:
        logger.info(f"  - {strat}")

    logger.info("\nPour comparer avec Freqtrade:")
    logger.info(
        """
   # Backtester chaque stratégie
   for strategy in EMA_Cross MACD_Crossover RSI_Cross Bollinger_Touch; do
       freqtrade backtesting \\
           --strategy ${strategy}_Strategy_Freqtrade \\
           --timerange 20240101-20240131
   done

   # Comparer résultats
   freqtrade backtesting-show --backtest-dir user_data/backtest_results/

   # Métriques clés à comparer:
   # - Total Return (%)
   # - Win Rate (%)
   # - Max Drawdown (%)
   # - Sharpe Ratio
   # - Profit Factor
   # - Avg Trade Duration
    """
    )


def main():
    """Point d'entrée principal."""
    print("\n" + "=" * 70)
    print("EXEMPLES BACKTESTING FREQTRADE × ROOT")
    print("=" * 70 + "\n")

    try:
        # Exemple 1: Backtesting simple
        backtest_root_strategy_simple()

        # Exemple 2: Configuration avancée
        backtest_with_config()

        # Exemple 3: Batch conversion
        batch_backtest_all_strategies()

        # Exemple 4: Analyse résultats
        analyze_backtest_results()

        # Exemple 5: Comparaison stratégies
        compare_strategies()

        print("\n" + "=" * 70)
        print("✓ Exemples générés avec succès!")
        print("=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)


if __name__ == "__main__":
    main()
