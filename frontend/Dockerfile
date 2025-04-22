FROM node:18-alpine as build

WORKDIR /app

# Copier les fichiers de configuration
COPY frontend/package.json frontend/package-lock.json ./

# Installer les dépendances
RUN npm ci

# Copier le code source
COPY frontend/ ./

# Construire l'application
RUN npm run build

# Étape de production avec Nginx
FROM nginx:alpine

# Copier la configuration Nginx personnalisée si nécessaire
# COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copier les fichiers construits depuis l'étape précédente
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 3000

# Configurer Nginx pour écouter sur le port 3000
RUN sed -i 's/listen\s*80;/listen 3000;/g' /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]