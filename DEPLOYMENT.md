# 🚀 Deployment Guide for The Eye

## Overview
This guide will help you deploy **The Eye** exoplanet detection application:
- **Frontend (React/Vite)**: GitHub Pages
- **Backend (Flask)**: Render (recommended free hosting)

---

## 📦 Part 1: Deploy Backend to Render

### Step 1: Prepare Backend for Deployment

1. Create a `requirements.txt` in `/TheEyeWebUI/server`:
```bash
cd TheEyeWebUI/server
pip freeze > requirements.txt
```

2. Create a `render.yaml` file in the project root:
```yaml
services:
  - type: web
    name: theeye-backend
    runtime: python
    buildCommand: pip install -r TheEyeWebUI/server/requirements.txt
    startCommand: cd TheEyeWebUI/server && gunicorn main:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.13.0
      - key: PORT
        value: 10000
```

3. Install gunicorn (production server):
```bash
pip install gunicorn
pip freeze > requirements.txt
```

### Step 2: Deploy to Render

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository (`RihzyStudent/TheEye`)
4. Configure:
   - **Name**: `theeye-backend`
   - **Branch**: `FrontEnd` or `main`
   - **Root Directory**: Leave blank
   - **Build Command**: `pip install -r TheEyeWebUI/server/requirements.txt`
   - **Start Command**: `cd TheEyeWebUI/server && gunicorn main:app --bind 0.0.0.0:$PORT`
   - **Instance Type**: Free
5. Click **"Create Web Service"**
6. Wait 5-10 minutes for deployment
7. **Copy your backend URL** (e.g., `https://theeye-backend.onrender.com`)

### Alternative: Railway

1. Go to [railway.app](https://railway.app) and sign up
2. Click **"New Project"** → **"Deploy from GitHub"**
3. Select your repository
4. Railway will auto-detect Flask and deploy
5. Copy the generated URL

---

## 🌐 Part 2: Deploy Frontend to GitHub Pages

### Step 1: Configure GitHub Repository

1. Go to your GitHub repository settings
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select **"GitHub Actions"**

### Step 2: Add Backend URL Secret

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **"New repository secret"**
3. Name: `VITE_API_URL`
4. Value: Your Render backend URL (e.g., `https://theeye-backend.onrender.com`)
5. Click **"Add secret"**

### Step 3: Enable GitHub Pages

1. The GitHub Action (`.github/workflows/deploy.yml`) will automatically:
   - Build the frontend
   - Deploy to GitHub Pages
2. Push your code to trigger deployment:
```bash
git add .
git commit -m "🚀 Configure for deployment"
git push origin FrontEnd
```

3. Check **Actions** tab to monitor deployment progress
4. Your site will be live at: `https://rihzystudent.github.io/TheEye/`

---

## 🔧 Part 3: Update CORS in Backend

After deploying, update CORS in `TheEyeWebUI/server/main.py`:

```python
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:5173",  # Local development
            "https://rihzystudent.github.io"  # Production
        ]
    }
})
```

Commit and push:
```bash
git add TheEyeWebUI/server/main.py
git commit -m "🌐 Update CORS for production"
git push origin FrontEnd
```

---

## ✅ Part 4: Test Deployment

1. Visit: `https://rihzystudent.github.io/TheEye/`
2. Try:
   - **FITS Analysis** with Target ID: `KIC 11446443`
   - **CSV Upload** with test data
   - **Manual Entry** with sample parameters

### Troubleshooting

#### Frontend shows "Failed to fetch"
- Check that backend URL in GitHub Secrets is correct
- Verify backend is running on Render (check logs)
- Check browser console for CORS errors

#### Backend crashes on Render
- Check Render logs for errors
- Verify `requirements.txt` includes all dependencies
- Ensure `lightkurve` and `transitleastsquares` are installed

#### GitHub Pages shows 404
- Wait 2-3 minutes after deployment
- Check that GitHub Actions completed successfully
- Verify GitHub Pages is enabled in repository settings

---

## 📊 Monitoring

### Backend Logs (Render)
- Go to your Render dashboard
- Click on your service
- Click **"Logs"** to see real-time output

### Frontend Logs
- Open browser DevTools (F12)
- Check Console tab for errors
- Check Network tab for failed requests

---

## 💰 Cost

- **GitHub Pages**: Free
- **Render Free Tier**: 
  - Free forever
  - 750 hours/month
  - Spins down after 15 min of inactivity
  - First request may be slow (30s cold start)

---

## 🔄 Making Updates

### Update Frontend
```bash
git add TheEyeWebUI/the-eye/
git commit -m "✨ Update frontend"
git push origin FrontEnd
```
GitHub Actions will auto-deploy!

### Update Backend
```bash
git add TheEyeWebUI/server/
git commit -m "🔧 Update backend"
git push origin FrontEnd
```
Render will auto-deploy!

---

## 🎉 You're Done!

Your exoplanet detection app is now live! Share the link:
`https://rihzystudent.github.io/TheEye/`

