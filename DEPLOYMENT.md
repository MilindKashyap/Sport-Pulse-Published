# Deployment Guide for Render

This guide will help you deploy SportPulse on Render.

## Prerequisites

1. A GitHub account
2. A Render account (sign up at https://render.com)
3. Your code pushed to a GitHub repository

## Deployment Steps

### 1. Push to GitHub

Make sure all your code is committed and pushed to a GitHub repository:

```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Deploy on Render

#### Option A: Using render.yaml (Recommended)

1. Go to your Render Dashboard
2. Click "New +" and select "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` file
5. Click "Apply" to deploy

#### Option B: Manual Setup

1. Go to your Render Dashboard
2. Click "New +" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: sportpulse (or your preferred name)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT app:app`
5. Click "Create Web Service"

### 3. Important Notes

- The `multiTimeline.csv` file must be included in your repository (it's not in .gitignore)
- Render will automatically set the `PORT` environment variable
- The free tier may have cold starts - your app may take a few seconds to respond after inactivity
- Make sure all dependencies in `requirements.txt` are compatible

### 4. Environment Variables (if needed)

If you need to add environment variables later:
1. Go to your service settings in Render
2. Navigate to "Environment"
3. Add any required environment variables

## Troubleshooting

- **Build fails**: Check that all dependencies in `requirements.txt` are valid
- **App crashes**: Check the logs in Render dashboard for error messages
- **502 Bad Gateway**: Ensure gunicorn is properly configured and the app starts correctly
- **Import errors**: Make sure `models/__init__.py` exists and all model files are present

## Post-Deployment

After deployment, your app will be available at:
`https://sportpulse.onrender.com` (or your custom domain)

Note: On the free tier, your app will spin down after 15 minutes of inactivity and may take 30-60 seconds to spin back up.

