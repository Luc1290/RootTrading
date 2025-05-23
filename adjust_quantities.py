#!/usr/bin/env python3
"""
Script pour ajuster automatiquement les quantités de trading dans .env
en fonction des soldes disponibles et des contraintes Binance.
"""
import os
import requests
import re
from typing import Dict, Tuple

# Configuration de l'API
TRADER_HOST = os.getenv("TRADER_HOST", "localhost")
TRADER_PORT = os.getenv("TRADER_PORT", "5002")
TRADER_URL = f"http://{TRADER_HOST}:{TRADER_PORT}"

# Paires de trading et leurs devises de cotation
TRADING_PAIRS = {
    "BTCUSDC": ("BTC", "USDC"),
    "ETHUSDC": ("ETH", "USDC"),
    "ETHBTC": ("ETH", "BTC"),
}

def get_current_balances() -> Dict[str, float]:
    """Récupère les soldes actuels depuis l'API."""
    try:
        response = requests.get(f"{TRADER_URL}/balance")
        if response.status_code == 200:
            data = response.json()
            balances = {}
            for balance in data.get('balances', []):
                balances[balance['asset']] = balance['free']
            return balances
        else:
            print(f"Erreur API: {response.status_code}")
            return {}
    except Exception as e:
        print(f"Erreur lors de la récupération des soldes: {e}")
        return {}

def get_current_price(symbol: str) -> float:
    """Récupère le prix actuel d'un symbole."""
    try:
        response = requests.get(f"{TRADER_URL}/status")
        if response.status_code == 200:
            data = response.json()
            prices = data.get('last_prices', {})
            return prices.get(symbol, 0)
        else:
            return 0
    except Exception as e:
        print(f"Erreur lors de la récupération du prix: {e}")
        return 0

def get_symbol_constraints(symbol: str) -> Dict[str, float]:
    """Récupère les contraintes d'un symbole."""
    try:
        response = requests.get(f"{TRADER_URL}/constraints/{symbol}")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        print(f"Erreur lors de la récupération des contraintes: {e}")
        return {}

def calculate_optimal_quantity(symbol: str, base_asset: str, quote_asset: str, 
                             balances: Dict[str, float], allocation_percent: float = 20) -> float:
    """
    Calcule la quantité optimale pour un symbole donné.
    
    Args:
        symbol: Le symbole de trading (ex: BTCUSDC)
        base_asset: L'actif de base (ex: BTC)
        quote_asset: L'actif de cotation (ex: USDC)
        balances: Les soldes disponibles
        allocation_percent: Pourcentage du solde à allouer par trade
    
    Returns:
        La quantité optimale à trader
    """
    # Récupérer le prix actuel
    price = get_current_price(symbol)
    if price <= 0:
        print(f"Prix invalide pour {symbol}")
        return 0
    
    # Récupérer les contraintes
    constraints = get_symbol_constraints(symbol)
    min_qty = constraints.get('min_qty', 0.001)
    step_size = constraints.get('step_size', 0.00001)
    min_notional = constraints.get('min_notional', 10)
    
    # Calculer la quantité basée sur le pourcentage du solde
    quote_balance = balances.get(quote_asset, 0)
    max_spend = quote_balance * (allocation_percent / 100)
    
    # Calculer la quantité correspondante
    quantity = max_spend / price
    
    # S'assurer que la quantité respecte les contraintes
    # 1. Vérifier la quantité minimale
    if quantity < min_qty:
        quantity = min_qty
    
    # 2. Vérifier le notional minimal
    notional = quantity * price
    if notional < min_notional:
        quantity = min_notional / price
    
    # 3. Arrondir au step size
    if step_size > 0:
        quantity = round(quantity / step_size) * step_size
    
    # 4. Vérifier que nous avons assez de fonds
    total_cost = quantity * price * 1.001  # Inclure les frais
    if total_cost > quote_balance:
        # Réduire la quantité pour qu'elle corresponde au solde disponible
        quantity = (quote_balance * 0.99) / price  # 99% pour garder une marge
        # Re-arrondir au step size
        if step_size > 0:
            quantity = int(quantity / step_size) * step_size
    
    return quantity

def update_env_file(new_quantities: Dict[str, float]):
    """Met à jour le fichier .env avec les nouvelles quantités."""
    env_path = "/mnt/e/RootTrading/RootTrading/.env"
    
    # Lire le fichier .env
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Remplacer les quantités
    replacements = {
        "BTCUSDC": "TRADE_QUANTITY_BTC",
        "ETHUSDC": "TRADE_QUANTITY_ETH",
        "ETHBTC": "TRADE_QUANTITY_ETHBTC",
    }
    
    for symbol, quantity in new_quantities.items():
        if symbol in replacements:
            env_var = replacements[symbol]
            # Rechercher et remplacer la ligne
            pattern = f"^{env_var}=.*$"
            replacement = f"{env_var}={quantity:.8f}"
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Écrire le fichier mis à jour
    with open(env_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Fichier .env mis à jour avec les nouvelles quantités")

def main():
    """Fonction principale."""
    print("🔍 Analyse des soldes et ajustement des quantités de trading...")
    
    # Récupérer les soldes actuels
    balances = get_current_balances()
    if not balances:
        print("❌ Impossible de récupérer les soldes")
        return
    
    print("\n📊 Soldes actuels:")
    for asset, balance in balances.items():
        if balance > 0:
            print(f"  {asset}: {balance:.8f}")
    
    # Calculer les nouvelles quantités
    new_quantities = {}
    print("\n📈 Calcul des quantités optimales:")
    
    for symbol, (base_asset, quote_asset) in TRADING_PAIRS.items():
        quantity = calculate_optimal_quantity(symbol, base_asset, quote_asset, balances)
        if quantity > 0:
            new_quantities[symbol] = quantity
            price = get_current_price(symbol)
            cost = quantity * price * 1.001
            print(f"  {symbol}: {quantity:.8f} {base_asset} (coût: {cost:.2f} {quote_asset})")
    
    # Demander confirmation
    print("\n⚠️  Les quantités ci-dessus seront écrites dans le fichier .env")
    response = input("Voulez-vous continuer? (y/N): ")
    
    if response.lower() == 'y':
        update_env_file(new_quantities)
        print("\n✅ Quantités mises à jour avec succès!")
        print("⚠️  N'oubliez pas de redémarrer le service trader pour appliquer les changements:")
        print("   docker-compose restart trader")
    else:
        print("❌ Mise à jour annulée")

if __name__ == "__main__":
    main()