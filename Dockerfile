# ---------- Build stage ----------
FROM node:18-alpine AS build

WORKDIR /app

# install deps
COPY package.json package-lock.json* yarn.lock* ./
COPY wllama-wllama-2.3.5.tgz ./

# gunakan npm atau yarn sesuai proyek
RUN npm ci --silent

# copy source and build
COPY . .
RUN npm run build

# ---------- Production stage ----------
FROM nginx:stable-alpine AS production

# remove default nginx content
RUN rm -rf /usr/share/nginx/html/*

# copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# optional: expose port
EXPOSE 80

# healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
