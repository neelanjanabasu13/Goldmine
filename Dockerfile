FROM node:22-slim

WORKDIR /app

# Goldmine currently uses Bun's lockfile rather than an npm package-lock, so
# install declared dependencies before copying the application source.
COPY package.json bun.lock ./
RUN npm install --no-audit --no-fund

COPY . .
RUN npm run build

ENV NODE_ENV=production
EXPOSE 8080

CMD ["npm", "run", "start"]
