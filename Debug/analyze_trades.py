# -*- coding: utf-8 -*-
import requests
import json
import sys
from collections import defaultdict

def get_strategy_config(strategy_name: str, base_url="http://localhost:8000"):
    url = f"{base_url}/strategy_configs?name={strategy_name}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        configs = response.json()
        if configs:
            return configs[0]  # Retourne la première configuration trouvée
        return None
    except requests.exceptions.RequestException as e:
        print(f"  Erreur lors de la récupération de la configuration de stratégie {strategy_name}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"  Erreur de décodage JSON pour la configuration de stratégie {strategy_name}. Réponse: {response.text}")
        return None

def analyze_trades(base_url="http://localhost:8000"):
    print("Démarrage de l'analyse des trades...")
    all_trades = []
    page = 1
    page_size = 50  # Taille de page par défaut de l'API

    while True:
        url = f"{base_url}/trades?page={page}&page_size={page_size}"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Lève une exception pour les codes d'état HTTP d'erreur
            data = response.json()
            
            trades_on_page = data.get("trades", [])
            total_count = data.get("total_count", 0)

            all_trades.extend(trades_on_page)
            print(f"  Récupération de la page {page}/{ (total_count + page_size - 1) // page_size } ({len(trades_on_page)} trades)...")

            if len(all_trades) >= total_count:
                break
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"  Erreur lors de la récupération des trades: {e}")
            print("Assurez-vous que le service 'portfolio' est bien démarré et accessible sur http://localhost:8000.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"  Erreur de décodage JSON pour la page {page}. Réponse: {response.text}")
            sys.exit(1)

    if not all_trades:
        print("Aucun trade trouvé pour l'analyse.")
        return

    print(f"  {len(all_trades)} trades récupérés au total.")

    # Initialisation des statistiques
    total_profit_loss = 0.0
    winning_trades = []
    losing_trades = []
    break_even_trades = []
    
    profit_by_strategy = defaultdict(float)
    profit_by_symbol = defaultdict(float)

    for trade in all_trades:
        try:
            profit_loss = float(trade.get("profit_loss", 0.0))
            profit_loss_percent = float(trade.get("profit_loss_percent", 0.0))
            
            total_profit_loss += profit_loss
            
            trade_info = {
                "id": trade.get("id"),
                "symbol": trade.get("symbol"),
                "strategy": trade.get("strategy"),
                "profit_loss": profit_loss,
                "profit_loss_percent": profit_loss_percent
            }

            if profit_loss > 0:
                winning_trades.append(trade_info)
            elif profit_loss < 0:
                losing_trades.append(trade_info)
            else:
                break_even_trades.append(trade_info)
                
            # Agrégation par stratégie et symbole
            profit_by_strategy[trade.get("strategy")] += profit_loss
            profit_by_symbol[trade.get("symbol")] += profit_loss

        except (ValueError, TypeError) as e:
            print(f"  Ignoré un trade avec des données invalides: {trade} - Erreur: {e}")
            continue

    num_winning = len(winning_trades)
    num_losing = len(losing_trades)
    num_break_even = len(break_even_trades)
    total_completed_trades = num_winning + num_losing + num_break_even

    win_rate = (num_winning / total_completed_trades * 100) if total_completed_trades > 0 else 0

    avg_winning_profit = sum(t["profit_loss"] for t in winning_trades) / num_winning if num_winning > 0 else 0
    avg_losing_loss = sum(t["profit_loss"] for t in losing_trades) / num_losing if num_losing > 0 else 0

    # Trier les trades pour identifier les plus gros gains/pertes
    winning_trades_sorted = sorted(winning_trades, key=lambda x: x["profit_loss"], reverse=True)
    losing_trades_sorted = sorted(losing_trades, key=lambda x: x["profit_loss"])

    print("\n--- Rapport d'Analyse des Trades ---")
    print(f"Nombre total de trades complétés: {total_completed_trades}")
    print(f"Trades gagnants: {num_winning} ({win_rate:.2f}%)")
    print(f"Trades perdants: {num_losing}")
    print(f"Trades à l'équilibre: {num_break_even}")
    print(f"Profit/Perte total: {total_profit_loss:.2f}")
    print(f"Profit moyen par trade gagnant: {avg_winning_profit:.2f}")
    print(f"Perte moyenne par trade perdant: {avg_losing_loss:.2f}")

    print("\n--- Top 5 des Trades Gagnants ---")
    for i, trade in enumerate(winning_trades_sorted[:5]):
        print(f"{i+1}. ID: {trade['id']}, Symbole: {trade['symbol']}, Stratégie: {trade['strategy']}, Profit: {trade['profit_loss']:.2f} ({trade['profit_loss_percent']:.2f}%)")

    print("\n--- Top 5 des Trades Perdants ---")
    for i, trade in enumerate(losing_trades_sorted[:5]):
        print(f"{i+1}. ID: {trade['id']}, Symbole: {trade['symbol']}, Stratégie: {trade['strategy']}, Perte: {trade['profit_loss']:.2f} ({trade['profit_loss_percent']:.2f}%)")
        
    print("\n--- Profit/Perte par Stratégie ---")
    for strategy, profit in sorted(profit_by_strategy.items(), key=lambda item: item[1], reverse=True):
        print(f"  {strategy}: {profit:.2f}")

    print("\n--- Profit/Perte par Symbole ---")
    for symbol, profit in sorted(profit_by_symbol.items(), key=lambda item: item[1], reverse=True):
        print(f"  {symbol}: {profit:.2f}")

    if num_losing > 0 and avg_losing_loss < avg_winning_profit:
        print("\n--- Conclusion ---")
        print("Votre analyse confirme que malgré un bon taux de trades gagnants,")
        print("les pertes moyennes par trade sont supérieures aux gains moyens.")
        print("Cela indique un problème de gestion du risque où les trades perdants sont trop importants.")
        print("Il est crucial de revoir votre stratégie de stop-loss ou de taille de position pour limiter l'ampleur des pertes.")

    # Récupération et affichage de la configuration de la stratégie Aggregated_1
    print("\n--- Configuration de la Stratégie 'Aggregated_1' ---")
    aggregated_strategy_config = get_strategy_config("Aggregated_1", base_url)
    if aggregated_strategy_config:
        print(f"Nom: {aggregated_strategy_config.get('name')}")
        print(f"Mode: {aggregated_strategy_config.get('mode')}")
        print(f"Symboles: {aggregated_strategy_config.get('symbols')}")
        print(f"Paramètres: {json.dumps(aggregated_strategy_config.get('params'), indent=2)}")
        print(f"Trades simultanés max: {aggregated_strategy_config.get('max_simultaneous_trades')}")
        print(f"Activée: {aggregated_strategy_config.get('enabled')}")
    else:
        print("Configuration de la stratégie 'Aggregated_1' non trouvée ou erreur lors de la récupération.")


if __name__ == "__main__":
    analyze_trades()