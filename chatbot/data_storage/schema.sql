-- Description: create tables, types for the auth schema in the chlit db

CREATE SCHEMA IF NOT EXISTS auth;
CREATE SCHEMA IF NOT EXISTS bot;

-- enum for user_role in table users
CREATE TYPE USER_ROLE AS ENUM ('admin', 'user', 'mis-user');

-- table to save users
CREATE TABLE IF NOT EXISTS auth.users (
    "id" SERIAL PRIMARY KEY,
    "metadata" JSONB NOT NULL,
    "first_name" TEXT NOT NULL,
    "last_name" TEXT NOT NULL,
    "password_hash" VARCHAR(255) NOT NULL,
    "email" VARCHAR(255) UNIQUE CHECK(email ~ '^[A-Za-z]+\.[A-Za-z0-9]+@(?=[^@]*uni)(?=[^@]*a)[A-Za-z0-9.-]+\.de$'),
    "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    "user_uuid" UUID UNIQUE NOT NULL, -- added for personal references, not as easy to guess as id
    "user_name" TEXT NOT NULL,
    "permissions" USER_ROLE NOT NULL DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS auth.verification_codes (
    "id" SERIAL PRIMARY KEY,
    "user_id" INTEGER REFERENCES auth.users("id") ON DELETE CASCADE,
    "reset_code" UUID UNIQUE NOT NULL,
    "additional_data" JSONB DEFAULT NULL, -- to store optional changes in users
    "used" BOOLEAN DEFAULT FALSE NOT NULL,
    "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS bot.threads (
    "id" UUID PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" UUID,
    "userIdentifier" TEXT,
    "tags" TEXT[],
    "metadata" JSONB,
    FOREIGN KEY ("userId") REFERENCES auth.users("user_uuid") ON DELETE CASCADE
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
