# Node 23.6+ runs TypeScript directly, so there is no build step and no compiler in
# the image — just prod dependencies and the source.
FROM node:24-alpine

WORKDIR /app
ENV NODE_ENV=production HOST=0.0.0.0 PORT=8787

COPY package*.json ./
RUN npm install --omit=dev --no-audit --no-fund && npm cache clean --force

COPY src ./src
COPY system-prompt.md ./

USER node
EXPOSE 8787
CMD ["node", "src/index.ts"]
