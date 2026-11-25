# Deploy to Render (free tier)

1. Push this project to GitHub (the `.env` file is git-ignored, so your key stays private).
2. Go to https://dashboard.render.com and sign in with GitHub.
3. Click **New +** -> **Blueprint**, pick the `agentic-rag` repository, and click **Apply**.
   Render reads `render.yaml` and builds the Dockerfile.
4. When it asks for values, enter:
   - `GEMINI_API_KEY`: your key from aistudio.google.com/apikey
   - `APP_PASSWORD`: a password you choose (anyone opening the site must enter it)
5. Wait for the build to finish, then open the `.onrender.com` link. Log in with any username and that password.

## What to expect on the free tier

- The service **sleeps after ~15 minutes** without visitors; the first request afterwards takes about a minute.
- The disk is **temporary**: indexed repos disappear on restart, so paste your link again after a restart.
- All visitors share **your** Gemini free quota. `APP_PASSWORD` and `RATE_LIMIT_PER_MIN` protect it.
- The free instance has about 512 MB of memory. Keep `MAX_CHUNKS` low (250 is a safe start).

## Run it locally with Docker instead

```powershell
Copy-Item .env.example .env      # then put your key in .env
docker compose up --build
```

Open http://localhost:8000.

## Tip for your resume

Put the live link and a demo GIF in the README. Share the password privately with recruiters
instead of writing it in the README.
