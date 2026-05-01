# Traffic Sign Recognition ML

Traffic sign recognition app with:
- `client/`: Next.js frontend
- `server/`: Flask API + ML inference service

This README is focused on one goal: **get the project running locally from scratch**.

## Prerequisites

- Node.js 18+ and npm
- Python 3.10+ (recommended)

## 1) Install dependencies

From project root:

```bash
npm install
cd client && npm install
cd ../server && pip install -r requirements.txt
cd ..
```

## 2) Configure backend environment

Create `server/.env`:

```bash
FLASK_ENV=development
FRONTEND_ORIGINS=http://localhost:3000
```

Optional backend env vars:
- `MAX_UPLOAD_MB` (default: `4`)
- `FRONTEND_ORIGINS` can be a comma-separated list

## 3) Prepare and train a model (required first run)

The backend expects this file:
- `server/app/ml/models/simple_classifier.joblib`

If it does not exist, train it:

```bash
python server/app/ml/scripts/clean-splits.py --root server/app/ml/dataset
python server/app/ml/scripts/train-simple-classifier.py --root server/app/ml/dataset
```

Expected outputs:
- `server/app/ml/dataset/models/simple_classifier.joblib`
- `server/app/ml/dataset/reports/simple_train_metrics.txt`

If your trained model is in `server/app/ml/dataset/models/`, copy it to `server/app/ml/models/`:

```bash
New-Item -ItemType Directory -Path server/app/ml/models -Force
Copy-Item server/app/ml/dataset/models/simple_classifier.joblib server/app/ml/models/simple_classifier.joblib -Force
```

## 4) Run the app

From project root:

```bash
npm run dev
```

Starts:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:5000`

## 5) Verify backend is ready

Health:

```bash
curl http://localhost:5000/api/health
```

Model readiness:

```bash
curl http://localhost:5000/api/v1/ready
```

If readiness returns `503`, the model file is missing or invalid.

## Useful scripts

From project root:
- `npm run dev` - run frontend + backend together
- `npm run dev:backend` - run backend only
- `npm run lint` - lint frontend files
- `npm run lint:fix` - auto-fix lint issues
- `npm run lint:python` - lint backend Python code
- `npm run type-check` - TypeScript type check

From `client/`:
- `npm run dev`
- `npm run build`
- `npm run start`
