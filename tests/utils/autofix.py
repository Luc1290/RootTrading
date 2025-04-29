#!/usr/bin/env python3
"""
Script d'auto-correction des problèmes courants dans le code RootTrading.
Ce script cherche et corrige automatiquement certains problèmes détectés
par pylint, flake8 et mypy.
"""

import os
import re
import sys
import argparse
from typing import List, Dict, Tuple, Set, Optional


def fix_exception_handling(file_path: str) -> int:
    """
    Corrige les blocs try/except trop génériques dans un fichier.
    
    Args:
        file_path: Chemin vers le fichier à corriger
        
    Returns:
        Nombre de corrections effectuées
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Motif pour détecter les try/except Exception trop génériques
    pattern = r'try:.*?except\s+Exception\s+as\s+e:.*?(?:logger\.error|print)\s*\(.*?\)'
    
    # Compter les occurrences avant correction
    matches = re.findall(pattern, content, re.DOTALL)
    count = len(matches)
    
    if count == 0:
        return 0
    
    # Correction basée sur le contexte
    if "kafka" in file_path.lower():
        # Correction pour les fichiers liés à Kafka
        content = re.sub(
            r'try:(\s+.*?)except\s+Exception\s+as\s+e:(\s+.*?logger\.error\s*\(.*?\))',
            r'try:\1except BufferError as e:\2\n        self.producer.flush()\n        # Réessayer après flush\n    except KafkaException as e:\2\n    except (ConnectionError, TimeoutError) as e:\n        logger.warning(f"Problème de connexion: {str(e)}")\n        self.reconnect()',
            content, 
            flags=re.DOTALL
        )
    
    elif "redis" in file_path.lower():
        # Correction pour les fichiers liés à Redis
        content = re.sub(
            r'try:(\s+.*?)except\s+Exception\s+as\s+e:(\s+.*?logger\.error\s*\(.*?\))',
            r'try:\1except (ConnectionError, TimeoutError) as e:\n        logger.warning(f"Perte de connexion Redis: {str(e)}")\n        self.reconnect()\n    except RedisError as e:\2',
            content, 
            flags=re.DOTALL
        )
    
    else:
        # Correction générique
        content = re.sub(
            r'try:(\s+.*?)except\s+Exception\s+as\s+e:(\s+.*?logger\.error\s*\(.*?\))',
            r'try:\1except (ValueError, TypeError) as e:\2\n    except (ConnectionError, TimeoutError) as e:\n        logger.warning(f"Problème de connexion: {str(e)}")\n    except Exception as e:\n        logger.critical(f"Erreur inattendue: {str(e)}")\n        raise',
            content, 
            flags=re.DOTALL
        )
    
    # Écrire le contenu corrigé
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return count


def add_type_annotations(file_path: str) -> int:
    """
    Ajoute des annotations de type basiques aux fonctions qui en manquent.
    
    Args:
        file_path: Chemin vers le fichier à corriger
        
    Returns:
        Nombre de corrections effectuées
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Détecter les imports existants
    has_typing_import = re.search(r'from\s+typing\s+import', content) is not None
    
    # Ajouter l'import typing si nécessaire
    if not has_typing_import:
        import_pattern = r'import\s+.*?\n\n'
        if re.search(import_pattern, content):
            content = re.sub(
                import_pattern,
                lambda m: m.group(0) + 'from typing import Any, Dict, List, Optional, Tuple, Union\n\n',
                content,
                count=1
            )
        else:
            content = 'from typing import Any, Dict, List, Optional, Tuple, Union\n\n' + content
    
    # Motif pour trouver les définitions de fonction sans annotations de type
    func_pattern = r'def\s+(\w+)\s*\(([^)]*)\)(?!\s*->\s*[\w\[\],\s]+):'
    
    # Compter les occurrences
    matches = re.findall(func_pattern, content)
    count = len(matches)
    
    if count == 0:
        return 0
    
    # Correction
    def add_types(match):
        func_name = match.group(1)
        params = match.group(2).strip()
        
        # Si pas de paramètres
        if not params:
            return f'def {func_name}() -> None:'
        
        # Ajouter des types aux paramètres
        typed_params = []
        for param in params.split(','):
            param = param.strip()
            if not param:
                continue
                
            if '=' in param:
                param_name, default = param.split('=', 1)
                param_name = param_name.strip()
                if param_name == 'self':
                    typed_params.append(param)
                else:
                    # Deviner le type en fonction du nom et de la valeur par défaut
                    if 'None' in default:
                        typed_params.append(f'{param_name}: Optional[Any] = {default.strip()}')
                    elif '"' in default or "'" in default:
                        typed_params.append(f'{param_name}: str = {default.strip()}')
                    elif default.strip().isdigit():
                        typed_params.append(f'{param_name}: int = {default.strip()}')
                    elif default.strip() in ['True', 'False']:
                        typed_params.append(f'{param_name}: bool = {default.strip()}')
                    elif '.' in default and default.replace('.', '').isdigit():
                        typed_params.append(f'{param_name}: float = {default.strip()}')
                    elif default.strip() in ['[]', '{}', 'dict()', 'list()']:
                        if default.strip() in ['[]', 'list()']:
                            typed_params.append(f'{param_name}: List[Any] = {default.strip()}')
                        else:
                            typed_params.append(f'{param_name}: Dict[str, Any] = {default.strip()}')
                    else:
                        typed_params.append(f'{param_name}: Any = {default.strip()}')
            else:
                param_name = param.strip()
                if param_name == 'self':
                    typed_params.append(param_name)
                else:
                    # Deviner le type en fonction du nom
                    if param_name in ['id', 'count', 'index', 'port']:
                        typed_params.append(f'{param_name}: int')
                    elif param_name in ['name', 'key', 'symbol', 'topic', 'channel']:
                        typed_params.append(f'{param_name}: str')
                    elif param_name in ['is_valid', 'enabled', 'active']:
                        typed_params.append(f'{param_name}: bool')
                    elif param_name.endswith('s') and not param_name in ['status', 'process']:
                        typed_params.append(f'{param_name}: List[Any]')
                    elif param_name.startswith('dict_') or param_name.endswith('_dict'):
                        typed_params.append(f'{param_name}: Dict[str, Any]')
                    elif param_name in ['callback', 'func', 'handler']:
                        typed_params.append(f'{param_name}: Callable')
                    else:
                        typed_params.append(f'{param_name}: Any')
        
        # Déterminer le type de retour
        return_type = 'Any'
        if func_name.startswith('get_') or func_name.startswith('fetch_'):
            return_type = 'Any'
        elif func_name.startswith('is_') or func_name.startswith('has_'):
            return_type = 'bool'
        elif func_name.startswith('count_'):
            return_type = 'int'
        elif func_name in ['__init__', 'close', 'disconnect', 'stop', 'unsubscribe']:
            return_type = 'None'
        
        return f'def {func_name}({", ".join(typed_params)}) -> {return_type}:'
    
    # Appliquer les corrections
    content = re.sub(func_pattern, add_types, content)
    
    # Écrire le contenu corrigé
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return count


def fix_missing_imports(file_path: str) -> int:
    """
    Ajoute les imports manquants pour les exceptions et types.
    
    Args:
        file_path: Chemin vers le fichier à corriger
        
    Returns:
        Nombre de corrections effectuées
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    imports_to_add = set()
    count = 0
    
    # Vérifier les exceptions utilisées
    if "RedisError" in content and "from redis.exceptions import RedisError" not in content:
        imports_to_add.add("from redis.exceptions import RedisError")
        count += 1
    
    if "KafkaException" in content and "from confluent_kafka import KafkaException" not in content:
        imports_to_add.add("from confluent_kafka import KafkaException")
        count += 1
    
    if "ConnectionError" in content and "import socket" not in content and "ConnectionError" not in content:
        if "requests" in content:
            imports_to_add.add("from requests.exceptions import ConnectionError")
        else:
            imports_to_add.add("from socket import error as ConnectionError")
        count += 1
    
    if "TimeoutError" in content and "TimeoutError" not in content:
        if "socket" in content:
            imports_to_add.add("from socket import timeout as TimeoutError")
        else:
            imports_to_add.add("from concurrent.futures import TimeoutError")
        count += 1
    
    # Ajouter les imports manquants
    if imports_to_add:
        # Trouver l'endroit où ajouter les imports
        import_section_end = 0
        for match in re.finditer(r'import.*?\n', content):
            end = match.end()
            if end > import_section_end:
                import_section_end = end
        
        for match in re.finditer(r'from\s+.*?\s+import.*?\n', content):
            end = match.end()
            if end > import_section_end:
                import_section_end = end
        
        if import_section_end > 0:
            # Ajouter après la section d'imports existante
            new_content = content[:import_section_end] + "\n" + "\n".join(imports_to_add) + "\n" + content[import_section_end:]
        else:
            # Ajouter au début du fichier, après les docstrings
            docstring_end = 0
            if content.startswith('"""') or content.startswith("'''"):
                docstring_end = content.find('"""', 3) + 3
                if docstring_end <= 2:  # Pas trouvé
                    docstring_end = content.find("'''", 3) + 3
                if docstring_end <= 2:  # Pas trouvé
                    docstring_end = 0
            
            new_content = content[:docstring_end] + "\n" + "\n".join(imports_to_add) + "\n\n" + content[docstring_end:]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    return count


def fix_long_lines(file_path: str, max_length: int = 120) -> int:
    """
    Divise les lignes trop longues.
    
    Args:
        file_path: Chemin vers le fichier à corriger
        max_length: Longueur maximale des lignes
        
    Returns:
        Nombre de corrections effectuées
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    count = 0
    fixed_lines = []
    
    for line in lines:
        if len(line.rstrip()) > max_length:
            # Pour les chaînes de format f-strings
            if 'f"' in line or "f'" in line:
                matches = re.findall(r'f([\'"])(.*?)\1', line)
                if matches:
                    quote, content = matches[0]
                    parts = re.split(r'(\{.*?\})', content)
                    current_line = f'f{quote}'
                    for part in parts:
                        if len(current_line + part + quote) > max_length:
                            fixed_lines.append(current_line + quote + '\n')
                            current_line = f'f{quote}' + part
                        else:
                            current_line += part
                    fixed_lines.append(current_line + quote + '\n')
                    count += 1
                    continue
            
            # Pour les listes d'arguments de fonction
            if '(' in line and ')' in line and ',' in line:
                open_idx = line.find('(')
                close_idx = line.rfind(')')
                if open_idx < close_idx:
                    prefix = line[:open_idx+1]
                    args = line[open_idx+1:close_idx]
                    suffix = line[close_idx:]
                    
                    # Diviser les arguments
                    arg_list = []
                    current_arg = ""
                    in_string = False
                    string_char = None
                    
                    for char in args:
                        if char in ['"', "'"]:
                            if not in_string:
                                in_string = True
                                string_char = char
                            elif char == string_char:
                                in_string = False
                                string_char = None
                        
                        current_arg += char
                        
                        if char == ',' and not in_string:
                            arg_list.append(current_arg)
                            current_arg = ""
                    
                    if current_arg:
                        arg_list.append(current_arg)
                    
                    if len(arg_list) > 1:
                        indent = ' ' * (len(prefix) - len(prefix.lstrip()))
                        fixed_lines.append(prefix + '\n')
                        for i, arg in enumerate(arg_list):
                            is_last = i == len(arg_list) - 1
                            arg = arg.strip()
                            if not is_last and not arg.endswith(','):
                                arg += ','
                            fixed_lines.append(f"{indent}    {arg}\n")
                        fixed_lines.append(f"{indent}{suffix}")
                        count += 1
                        continue
            
            # Pour les lignes d'assignation
            if ' = ' in line:
                parts = line.split(' = ', 1)
                if len(parts) == 2:
                    var_name, value = parts
                    fixed_lines.append(f"{var_name} = (\n")
                    indent = ' ' * (len(var_name) + 4)
                    fixed_lines.append(f"{indent}{value.strip()}\n")
                    fixed_lines.append(f"{' ' * (len(var_name))}\n)")
                    count += 1
                    continue
        
        # Si aucune correction n'a été appliquée
        fixed_lines.append(line)
    
    if count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
    
    return count


def add_parameter_validation(file_path: str) -> int:
    """
    Ajoute une validation basique des paramètres aux fonctions.
    
    Args:
        file_path: Chemin vers le fichier à corriger
        
    Returns:
        Nombre de corrections effectuées
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    count = 0
    
    # Trouver les fonctions avec des paramètres annotés
    pattern = r'def\s+(\w+)\s*\((.*?)\)\s*->\s*.*?:'
    
    def add_validations(match):
        nonlocal count
        func_name = match.group(1)
        params = match.group(2).strip()
        
        # Ne pas modifier __init__, __enter__, etc.
        if func_name.startswith('__'):
            return match.group(0)
        
        # Ne pas ajouter de validation pour les fonctions sans paramètres ou juste self
        if not params or params == 'self':
            return match.group(0)
        
        # Extraire les paramètres annotés
        param_list = []
        in_default = False
        current_param = ""
        
        for char in params:
            if char == '=' and not in_default:
                in_default = True
            elif char == ',' and not in_default:
                param_list.append(current_param.strip())
                current_param = ""
                in_default = False
            else:
                current_param += char
        
        if current_param:
            param_list.append(current_param.strip())
        
        # Générer les validations
        validations = []
        for param in param_list:
            if param == 'self':
                continue
                
            parts = param.split(':')
            if len(parts) < 2:
                continue
                
            param_name = parts[0].strip()
            param_type = parts[1].split('=')[0].strip()
            
            # Ignorer les paramètres déjà avec valeur par défaut None
            if '= None' in param:
                continue
            
            if 'str' in param_type:
                validations.append(f'if {param_name} is not None and not isinstance({param_name}, str):\n            raise TypeError(f"{param_name} doit être une chaîne, pas {{type({param_name}).__name__}}")')
            elif 'int' in param_type:
                validations.append(f'if {param_name} is not None and not isinstance({param_name}, int):\n            raise TypeError(f"{param_name} doit être un entier, pas {{type({param_name}).__name__}}")')
            elif 'float' in param_type:
                validations.append(f'if {param_name} is not None and not isinstance({param_name}, (int, float)):\n            raise TypeError(f"{param_name} doit être un nombre, pas {{type({param_name}).__name__}}")')
            elif 'bool' in param_type:
                validations.append(f'if {param_name} is not None and not isinstance({param_name}, bool):\n            raise TypeError(f"{param_name} doit être un booléen, pas {{type({param_name}).__name__}}")')
            elif 'List' in param_type:
                validations.append(f'if {param_name} is not None and not isinstance({param_name}, list):\n            raise TypeError(f"{param_name} doit être une liste, pas {{type({param_name}).__name__}}")')
            elif 'Dict' in param_type:
                validations.append(f'if {param_name} is not None and not isinstance({param_name}, dict):\n            raise TypeError(f"{param_name} doit être un dictionnaire, pas {{type({param_name}).__name__}}")')
        
        if validations:
            count += 1
            return match.group(0) + "\n        # Validation des paramètres\n        " + "\n        ".join(validations)
        else:
            return match.group(0)
    
    content = re.sub(pattern, add_validations, content, flags=re.DOTALL)
    
    if count > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    return count


def process_file(file_path: str) -> Dict[str, int]:
    """
    Applique toutes les corrections à un fichier.
    
    Args:
        file_path: Chemin vers le fichier à corriger
        
    Returns:
        Dictionnaire des corrections appliquées
    """
    results = {
        'exception_handling': 0,
        'type_annotations': 0,
        'missing_imports': 0,
        'long_lines': 0,
        'parameter_validation': 0
    }
    
    # Vérifier que le fichier existe et est un fichier Python
    if not os.path.isfile(file_path) or not file_path.endswith('.py'):
        return results
    
    print(f"Traitement du fichier: {file_path}")
    
    results['exception_handling'] = fix_exception_handling(file_path)
    results['type_annotations'] = add_type_annotations(file_path)
    results['missing_imports'] = fix_missing_imports(file_path)
    results['long_lines'] = fix_long_lines(file_path)
    results['parameter_validation'] = add_parameter_validation(file_path)
    
    return results


def process_directory(directory: str) -> Dict[str, int]:
    """
    Applique toutes les corrections aux fichiers Python d'un répertoire.
    
    Args:
        directory: Chemin vers le répertoire à traiter
        
    Returns:
        Dictionnaire des corrections appliquées
    """
    results = {
        'exception_handling': 0,
        'type_annotations': 0,
        'missing_imports': 0,
        'long_lines': 0,
        'parameter_validation': 0,
        'files_processed': 0
    }
    
    # Parcourir récursivement le répertoire
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                file_results = process_file(file_path)
                
                # Agréger les résultats
                for key, value in file_results.items():
                    results[key] += value
                
                results['files_processed'] += 1
    
    return results


def main() -> None:
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(description='Corrige automatiquement les problèmes courants dans le code Python.')
    parser.add_argument('path', help='Chemin vers le fichier ou répertoire à traiter')
    args = parser.parse_args()
    
    path = args.path
    
    # Vérifier que le chemin existe
    if not os.path.exists(path):
        print(f"Le chemin {path} n'existe pas.")
        sys.exit(1)
    
    # Traiter un fichier ou un répertoire
    if os.path.isfile(path):
        results = process_file(path)
        print("\nRésultats pour le fichier:", path)
    else:
        results = process_directory(path)
        print("\nRésultats pour le répertoire:", path)
    
    # Afficher les résultats
    print(f"Fichiers traités: {results.get('files_processed', 1)}")
    print(f"Corrections de gestion d'exceptions: {results['exception_handling']}")
    print(f"Annotations de types ajoutées: {results['type_annotations']}")
    print(f"Imports manquants ajoutés: {results['missing_imports']}")
    print(f"Lignes longues corrigées: {results['long_lines']}")
    print(f"Validations de paramètres ajoutées: {results['parameter_validation']}")
    
    total_fixes = sum(v for k, v in results.items() if k != 'files_processed')
    print(f"\nTotal des corrections: {total_fixes}")


if __name__ == "__main__":
    main()