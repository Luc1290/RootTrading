#!/usr/bin/env python3
"""
Script pour corriger les problèmes de syntaxe courants dans les blocs try/except.
"""
import re
import sys

def fix_try_except_blocks(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Motif 1: Erreur de parenthèse non fermée dans logger.error
    pattern1 = r'(logger\.error\(f".*?\{str\(e\))[^"\)]*?(\n\s*except)'
    content = re.sub(pattern1, r'\1"})\2', content, flags=re.DOTALL)
    
    # Motif 2: Problème avec des blocs except consécutifs mal indentés
    pattern2 = r'(\s*except \(ValueError, TypeError\) as e:.*?str\(e\))\n(\s*)except'
    content = re.sub(pattern2, r'\1")\n\2    self.running = False\n\2    raise\n\2except', content, flags=re.DOTALL)
    
    # Motif 3: Problème de "raise" incorrectement placé
    pattern3 = r'(logger\.critical\(f".*?\{str\(e\)})\n(\s*)raise"'
    content = re.sub(pattern3, r'\1")\n\2raise', content, flags=re.DOTALL)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Correction des blocs try/except terminée dans {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_syntax.py <file_path>")
        sys.exit(1)
    
    fix_try_except_blocks(sys.argv[1])