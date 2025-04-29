import sys
import os
import yaml
import re

def validate_docker_compose():
    """Valide le fichier docker-compose.yml."""
    try:
        print("Validation du fichier docker-compose.yml...")
        
        # Vérifier que le fichier existe
        if not os.path.exists('docker-compose.yml'):
            print("❌ Le fichier docker-compose.yml n'existe pas!")
            return False
        
        # Charger le fichier
        with open('docker-compose.yml', 'r') as f:
            docker_compose = yaml.safe_load(f)
        
        # Vérifier la structure de base
        if 'services' not in docker_compose:
            print("❌ La section 'services' est manquante dans docker-compose.yml!")
            return False
        
        # Vérifier les services essentiels
        essential_services = ['redis', 'kafka', 'db', 'gateway', 'analyzer', 'trader', 'portfolio']
        missing_services = [svc for svc in essential_services if svc not in docker_compose['services']]
        
        if missing_services:
            print(f"❌ Services essentiels manquants: {', '.join(missing_services)}")
        else:
            print("✅ Tous les services essentiels sont présents.")
        
        # Vérifier les volumes
        if 'volumes' not in docker_compose:
            print("⚠️ La section 'volumes' est manquante. Assurez-vous de définir les volumes persistants.")
        else:
            print(f"✅ Volumes définis: {', '.join(docker_compose['volumes'].keys())}")
        
        # Vérifier les dépendances
        dependency_issues = []
        
        for service_name, service in docker_compose['services'].items():
            if service_name not in ['redis', 'kafka', 'db', 'kafka-init']:
                # Vérifier que les services métier dépendent des services d'infrastructure
                if 'depends_on' not in service:
                    dependency_issues.append(f"{service_name} n'a pas de dépendances définies")
                else:
                    depends_on = service['depends_on']
                    
                    # Vérifier le format des dépendances (simple ou avec conditions)
                    if isinstance(depends_on, list):
                        missing_deps = [d for d in ['redis', 'kafka'] if d not in depends_on]
                        if missing_deps:
                            dependency_issues.append(f"{service_name} devrait dépendre de {', '.join(missing_deps)}")
                    elif isinstance(depends_on, dict):
                        missing_deps = [d for d in ['redis', 'kafka'] if d not in depends_on]
                        if missing_deps:
                            dependency_issues.append(f"{service_name} devrait dépendre de {', '.join(missing_deps)}")
        
        if dependency_issues:
            print("⚠️ Problèmes de dépendance détectés:")
            for issue in dependency_issues:
                print(f"  - {issue}")
        else:
            print("✅ Les dépendances entre services semblent correctes.")
        
        return True
    
    except yaml.YAMLError as e:
        print(f"❌ Erreur de syntaxe YAML dans docker-compose.yml: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la validation de docker-compose.yml: {str(e)}")
        return False

def validate_dockerfiles():
    """Valide les Dockerfiles des différents services."""
    try:
        print("\nValidation des Dockerfiles...")
        
        # Services à vérifier
        services = ['gateway', 'analyzer', 'trader', 'portfolio', 'coordinator', 'dispatcher', 'logger', 'pnl_tracker', 'risk_manager', 'scheduler']
        
        valid_files = []
        missing_files = []
        issues = []
        
        for service in services:
            dockerfile_path = f"{service}/Dockerfile"
            
            if not os.path.exists(dockerfile_path):
                missing_files.append(service)
                continue
            
            valid_files.append(service)
            
            # Lecture du contenu du Dockerfile
            with open(dockerfile_path, 'r') as f:
                content = f.read()
            
            # Vérifications de base
            checks = {
                "Base image": re.search(r'FROM\s+python', content, re.IGNORECASE) is not None,
                "WORKDIR": re.search(r'WORKDIR', content, re.IGNORECASE) is not None,
                "COPY requirements": re.search(r'COPY.*requirements', content, re.IGNORECASE) is not None,
                "pip install": re.search(r'pip\s+install', content, re.IGNORECASE) is not None,
                "COPY src": re.search(r'COPY', content, re.IGNORECASE) is not None,
                "CMD": re.search(r'CMD', content, re.IGNORECASE) is not None
            }
            
            failed_checks = [check for check, passed in checks.items() if not passed]
            
            if failed_checks:
                issues.append(f"{service}: manque {', '.join(failed_checks)}")
        
        if missing_files:
            print(f"⚠️ Dockerfiles manquants pour: {', '.join(missing_files)}")
        
        if valid_files:
            print(f"✅ Dockerfiles trouvés pour: {', '.join(valid_files)}")
        
        if issues:
            print("\n⚠️ Problèmes détectés dans les Dockerfiles:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Tous les Dockerfiles validés contiennent les instructions essentielles.")
        
        return len(missing_files) == 0 and len(issues) == 0
    
    except Exception as e:
        print(f"❌ Erreur lors de la validation des Dockerfiles: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Validation des fichiers Docker ===")
    
    compose_valid = validate_docker_compose()
    dockerfiles_valid = validate_dockerfiles()
    
    if compose_valid and dockerfiles_valid:
        print("\n✅ Tous les fichiers Docker semblent corrects!")
    else:
        issues = []
        if not compose_valid:
            issues.append("docker-compose.yml")
        if not dockerfiles_valid:
            issues.append("Dockerfiles")
            
        print(f"\n⚠️ Des problèmes ont été détectés dans: {', '.join(issues)}")
        print("Corrigez ces problèmes avant de lancer les conteneurs.")