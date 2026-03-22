# BetHoopsXG (formerly PrizePicks Pro)

BetHoopsXG is a fully automated, AI-powered web application that projects NBA player performance and automatically evaluates whether to go OVER or UNDER on PrizePicks betting lines. 

The application utilizes live data from `nba_api` to assemble historical box scores and match-ups. It derives rolling averages (like the player's performance over the last 5 games) and feeds them into a `scikit-learn` Random Forest Regressor model to generate accurate, leak-free predictions.

## Features
- **Multi-Stat Predictions:** Accurately forecasts player **Points**, **Rebounds**, and **Assists**.
- **Serverless Automation:** Runs completely hands-free using GitHub Actions. A daily cron job downloads new box scores, trains fresh ML models, and commits static JSON predictions directly to the repository.
- **Modern React Frontend:** A beautiful, responsive, dark-mode dashboard built with Vite and React, styled with glassmorphism Vanilla CSS.
- **Smart Recommendations:** Automatically compares the ML projection against the live PrizePicks lines to flag recommended OVER or UNDER bets (ignoring props that are too close to call).

## Architecture

1. **`buildmodel.py` & `predict.py`**: The core ML pipeline. Fetches gamelogs, computes rolling features, and trains the Scikit-Learn Random Forest model.
2. **`prizepickslines.py`**: Scrapes live player props and stat lines from the PrizePicks API.
3. **`generate_daily.py`**: The overarching script ran by GitHub Actions to generate static JSON files containing all predictions.
4. **`ui/`**: The Vite React frontend that instantly reads the static JSON data and presents the dashboard.

## Deployment & Hosting
This project operates entirely on a 100% free serverless stack:
- **Backend Model Training**: GitHub Actions (`.github/workflows/daily-predictions.yml`) executes the Python ML pipeline every morning.
- **Frontend Hosting**: The React application is built and hosted statically on GitHub Pages via the `gh-pages` deployment script.
