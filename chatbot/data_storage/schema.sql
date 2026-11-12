-- Description: create tables, types for the auth schema in the chlit db

CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS bot;
CREATE SCHEMA IF NOT EXISTS data;

-- enum for user_role in table users
CREATE TYPE USER_ROLE AS ENUM ('admin', 'user', 'mis-user');
CREATE TYPE DATA_TYPE AS ENUM ('website', 'pdf', 'image');

-- table to save users
CREATE TABLE IF NOT EXISTS auth.users (
    "id" UUID PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" JSONB NOT NULL,
    "createdAt" TEXT
);


CREATE TABLE IF NOT EXISTS bot.threads (
    "id" UUID PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" UUID,
    "userIdentifier" TEXT,
    "tags" TEXT[],
    "metadata" JSONB,
    FOREIGN KEY ("userId") REFERENCES auth.users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bot.steps (
    "id" UUID PRIMARY KEY,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    "threadId" UUID NOT NULL,
    "parentId" UUID,
    "streaming" BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError" BOOLEAN,
    "metadata" JSONB,
    "tags" TEXT[],
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "command" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" JSONB,
    "showInput" TEXT,
    "language" TEXT,
    "indent" INT,
    "defaultOpen" BOOLEAN,
    "modes" JSONB,
    "autoCollapse" BOOLEAN,
    FOREIGN KEY ("threadId") REFERENCES bot.threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bot.elements (
    "id" UUID PRIMARY KEY,
    "threadId" UUID,
    "type" TEXT,
    "url" TEXT,
    "chainlitKey" TEXT,
    "name" TEXT NOT NULL,
    "display" TEXT,
    "objectKey" TEXT,
    "size" TEXT,
    "page" INT,
    "language" TEXT,
    "forId" UUID,
    "mime" TEXT,
    "props" JSONB,
    FOREIGN KEY ("threadId") REFERENCES bot.threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bot.feedbacks (
    "id" UUID PRIMARY KEY,
    "forId" UUID NOT NULL,
    "threadId" UUID NOT NULL,
    "value" INT NOT NULL,
    "comment" TEXT,
    FOREIGN KEY ("threadId") REFERENCES bot.threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS data.website (
    "id" SERIAL PRIMARY KEY,
    "url" TEXT NOT NULL UNIQUE,
    "type" DATA_TYPE NOT NULL,
    "markdown" TEXT NOT NULL,
    "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data.auth_users (
    "id" SERIAL PRIMARY KEY,
    "username" TEXT NOT NULL UNIQUE,
    "password" TEXT NOT NULL,
    "role" USER_ROLE NOT NULL DEFAULT 'user',
    "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
