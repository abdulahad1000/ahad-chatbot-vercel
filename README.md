# Ahad-chatbot (Vercel)

A chatbot that remembers what you told it in past conversations.

**Live:** https://ahad-chatbot-vercel.vercel.app/


## How it works

Every message is saved to a Postgres database along with its embedding. When you
send a new message, it finds related messages from *other* sessions and ranks
them by how recent and how relevant they are.


## Files

```
index.html          the page
api/chat.py         backend function
requirements.txt    dependencies
vercel.json         build config
```

## Setup

Import the repo into Vercel and add three environment variables:

```
DATABASE_URL
GEMINI_API_KEY
GROQ_API_KEY
```

## Try it

Say "I'm allergic to shellfish", reload the page, then ask what to avoid at a
seafood restaurant. It remembers.

## Stack

Python, Postgres, Gemini, Groq, Vercel.
