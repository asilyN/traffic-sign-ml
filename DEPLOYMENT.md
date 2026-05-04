# Deployment Guide

This project is a monorepo with a Next.js frontend and Python backend. Follow the instructions below to deploy both components.

## Frontend Deployment (Vercel)

The Next.js frontend is configured to deploy on Vercel automatically.

### Prerequisites

- A Vercel account (free tier available)
- GitHub account with this repository connected

### Steps

1. **Connect to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Sign in with GitHub
   - Click "Import Project"
   - Select this repository
   - Select `client` as the "Root Directory"
   - Click "Deploy"

2. **Configure Environment Variables**
   - In your Vercel project settings, go to Settings → Environment Variables
   - Add `NEXT_PUBLIC_API_URL` pointing to your backend API (see Backend Deployment section)
   - Example: `https://your-backend-url.railway.app`

3. **Deploy**
   - Merges to `main` (or your default branch) will automatically deploy

## Backend Deployment (Railway, Render, or Fly.io)

The Python backend needs to be deployed separately as Vercel doesn't support Python.

### Option 1: Railway (Recommended)

1. **Prerequisites**
   - Railway account at [railway.app](https://railway.app)
   - GitHub account connected

2. **Deploy**
   - Create a new project on Railway
   - Select "GitHub Repo"
   - Choose this repository
   - Railway will auto-detect the Python app from `server/`
   - Configure environment variables:
     - `FLASK_ENV=production` (or similar for your framework)
     - Any API keys or database URLs

3. **Get Your API URL**
   - Railway will assign a domain (e.g., `https://your-app.railway.app`)
   - Add this to your frontend's `NEXT_PUBLIC_API_URL`

### Option 2: Render

1. **Prerequisites**
   - Render account at [render.com](https://render.com)
   - GitHub account connected

2. **Deploy**
   - Create a new "Web Service"
   - Connect your GitHub repository
   - Set root directory to `server/`
   - Environment: Python 3.11
   - Build command: `pip install -r requirements.txt`
   - Start command: `python run.py`
   - Configure environment variables as needed

3. **Get Your API URL**
   - Render assigns a domain automatically
   - Add this to your frontend's `NEXT_PUBLIC_API_URL`

### Option 3: Fly.io

1. **Prerequisites**
   - Fly.io account at [fly.io](https://fly.io)
   - Install flyctl CLI

2. **Deploy**
   ```bash
   cd server
   fly launch
   ```

   - Follow prompts to create app
   - Deploy with: `fly deploy`

## Configuration Files

### `vercel.json` (Frontend)

- Configures Next.js build for Vercel
- Defines environment variables needed by frontend
- Sets up API rewrites

### `server/railway.json`

- Configuration for Railway deployment
- Specifies Python 3.11 runtime

### `.vercelignore`

- Excludes unnecessary files from Vercel deployment

## Environment Variables

### Frontend (.env.local or Vercel)

```
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
```

### Backend (set in deployment platform)

```
FLASK_ENV=production
PORT=8000
# Add any other variables your API needs
```

## Monitoring & Debugging

- **Vercel Logs**: View in Vercel dashboard → Deployments → Function Logs
- **Railway/Render Logs**: Check their respective dashboards for application logs
- **Local Testing**: Run `npm run dev` from root to test both frontend and backend locally

## CORS Configuration

Make sure your backend is configured to accept requests from your Vercel frontend domain:

```python
# In your Flask/FastAPI app
ALLOWED_ORIGINS = [
    "https://your-vercel-domain.vercel.app",
    "http://localhost:3000",  # Local development
]
```

## Continuous Deployment

Both platforms support automatic deployments:

- **Vercel**: Deploys on every push to main branch
- **Railway/Render**: Configure webhook from GitHub for auto-deployment

## Domain Setup (Optional)

1. **Custom Domain on Vercel**
   - Vercel dashboard → Settings → Domains
   - Add your custom domain
   - Update DNS records

2. **Custom Domain on Railway/Render**
   - Follow platform-specific instructions
   - Point DNS to their servers

## Troubleshooting

### API calls return 404

- Check `NEXT_PUBLIC_API_URL` is correctly set
- Verify backend URL is accessible
- Check CORS configuration on backend

### Build fails on Vercel

- Ensure `client/package.json` has all dependencies
- Check Node version compatibility
- View build logs in Vercel dashboard

### Backend won't start on Railway/Render

- Verify `requirements.txt` has all dependencies
- Check start command matches your framework
- Review startup logs in platform dashboard

## Rollback

- **Vercel**: Go to Deployments tab, click "..." on previous deployment, select "Redeploy"
- **Railway/Render**: Similar rollback options in their dashboards
