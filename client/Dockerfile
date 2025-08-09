FROM node:lts-alpine

WORKDIR /app

ARG VITE_GOOGLE_OAUTH2
ARG VITE_GOOGLE_API_KEY
ARG VITE_SERVER_BASE_URL

ENV VITE_GOOGLE_OAUTH2=$VITE_GOOGLE_OAUTH2
ENV VITE_GOOGLE_API_KEY=$VITE_GOOGLE_API_KEY
ENV VITE_SERVER_BASE_URL=$VITE_SERVER_BASE_URL

COPY package.json package-lock.json vite.config.js ./

RUN npm install -f 

RUN npm i -g serve

COPY . .

RUN npm run build

EXPOSE 3000

CMD ["serve", "dist", "--single", "-l", "3000"]