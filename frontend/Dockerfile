FROM node:18-alpine AS build

WORKDIR /app

# Adjust the path to point to the frontend directory
COPY frontend/package.json ./

# Install dependencies
RUN npm install

# Copy the rest of the frontend files
COPY frontend/ ./

# Build the application
RUN npm run build

# Etape de production avec Nginx
FROM nginx:alpine

# Copy the built files from the previous stage
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 3000

# Configure Nginx to listen on port 3000
RUN sed -i 's/listen\s*80;/listen 3000;/g' /etc/nginx/conf.d/default.conf

CMD ["nginx", "-g", "daemon off;"]