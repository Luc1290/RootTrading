"""
Script de test pour vérifier les clés API Binance.
Ce script tente de se connecter à l'API Binance et de récupérer les informations du compte.
"""
import os
import hmac
import hashlib
import time
import requests
import json
from typing import Dict, Any

def test_binance_keys(api_key: str, api_secret: str) -> Dict[str, Any]:
    """
    Teste la validité des clés API Binance.
    
    Args:
        api_key: Clé API Binance
        api_secret: Clé secrète API Binance
        
    Returns:
        Résultat du test
    """
    base_url = "https://api.binance.com"
    endpoint = "/api/v3/account"
    
    # Préparer les paramètres
    timestamp = int(time.time() * 1000)
    params = {
        'timestamp': timestamp,
        'recvWindow': 10000
    }
    
    # Créer la signature
    query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(
        api_secret.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Ajouter la signature aux paramètres
    params['signature'] = signature
    
    # Configurer les headers
    headers = {
        "X-MBX-APIKEY": api_key
    }
    
    # Faire la requête
    try:
        response = requests.get(
            f"{base_url}{endpoint}",
            params=params,
            headers=headers
        )
        
        # Vérifier le statut
        if response.status_code == 200:
            result = {
                "success": True,
                "status_code": response.status_code,
                "message": "Connexion à l'API Binance réussie",
                "data": response.json()
            }
        else:
            result = {
                "success": False,
                "status_code": response.status_code,
                "message": f"Erreur: {response.text}",
                "data": None
            }
        
        return result
    
    except Exception as e:
        return {
            "success": False,
            "status_code": None,
            "message": f"Exception: {str(e)}",
            "data": None
        }

if __name__ == "__main__":
    # Récupérer les clés API depuis l'environnement ou les saisir manuellement
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_SECRET_KEY", "")
    
    if not api_key or not api_secret:
        print("Clés API non trouvées dans les variables d'environnement.")
        api_key = input("Entrez votre clé API Binance: ")
        api_secret = input("Entrez votre clé secrète API Binance: ")
    
    # Tester les clés
    print(f"Test des clés API Binance...\nClé API: {api_key[:5]}{'*' * 15}")
    result = test_binance_keys(api_key, api_secret)
    
    # Afficher le résultat
    print("\nRésultat du test:")
    print(f"Succès: {result['success']}")
    print(f"Code HTTP: {result['status_code']}")
    print(f"Message: {result['message']}")
    
    if result['success']:
        # Afficher quelques informations du compte
        account_data = result['data']
        print("\nInformations du compte:")
        print(f"Statut du compte: {account_data.get('accountType', 'Inconnu')}")
        print(f"Peut trader: {account_data.get('canTrade', False)}")
        print(f"Peut retirer: {account_data.get('canWithdraw', False)}")
        print(f"Peut déposer: {account_data.get('canDeposit', False)}")
        
        # Afficher quelques balances
        print("\nQuelques balances:")
        balances = [b for b in account_data.get('balances', []) 
                    if float(b.get('free', 0)) > 0 or float(b.get('locked', 0)) > 0]
        
        for balance in balances[:5]:  # Afficher les 5 premières balances non nulles
            asset = balance.get('asset', '')
            free = float(balance.get('free', 0))
            locked = float(balance.get('locked', 0))
            print(f"{asset}: {free} (libre) + {locked} (bloqué) = {free + locked} (total)")
        
        if len(balances) > 5:
            print(f"... et {len(balances) - 5} autres actifs")
    
    print("\nTest terminé.")