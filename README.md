# Traffic Sign Recognition ML

An intelligent traffic sign recognition system using machine learning. Upload an image, and the app automatically detects and classifies traffic signs with high accuracy.

**Frontend:** Next.js  
**Backend:** Flask + ML Inference  
**Model:** PyTorch CNN → Keras/TensorFlow classifier

---

## ✨ Features

- 📸 **Flexible Image Input**
  - Upload images (PNG, JPG, WEBP)
  - Drag-and-drop support
  - Live camera feed for real-time capture
  - Front/rear camera switching on mobile

- 🎯 **Real-time Detection**
  - Live detection from camera feed (800ms intervals)
  - Fast image upload detection
  - Browse files without resetting between detections
  - Multiple prediction confidence scores

- 🔍 **Advanced UI/UX**
  - Image preview with detection results
  - Detection history with localStorage
  - Tab-based interface (Camera/Upload modes)
  - Live bounding box visualization
  - Responsive design for mobile and desktop

- 🧠 **Machine Learning**
  - 4-block CNN with residual connection trained in PyTorch
  - Saved in Keras/H5 format for inference
  - High accuracy predictions with confidence scores (top-1, top-3, macro-F1)
  - Detailed instruction prompts for detected signs
  - Batch prediction support
  
- 📱 **Cross-Platform**
  - Progressive Web App ready
  - Mobile camera integration
  - Responsive Tailwind CSS design
  - Touch-friendly UI components
  
- 🚀 **Production Ready**
  - TypeScript for type safety
  - Docker-ready architecture
  - Git hooks and code linting
  - Performance optimized builds

---

## 🛠️ Built With

- **Frontend:** React, Next.js 16, TypeScript, Tailwind CSS
- **Backend:** Flask, Python 3.10+
- **ML:** PyTorch, TensorFlow/Keras, torchvision, scikit-learn
- **DevOps:** npm, pip, Docker-ready structure
- **Tools:** ESLint, Prettier, Husky (git hooks)s)

---

## 📋 Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.10+ (recommended)
- **Git** (for cloning the repo)
- ~500MB disk space (for dependencies and model)

---

## 🚀 Quick Start

### 1️⃣ Clone & Install Dependencies

From your terminal:

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-sign-ml.git
cd traffic-sign-ml

# Install frontend dependencies
cd client && npm install && cd ..

# Install backend dependencies
cd server && pip install -r requirements.txt && cd ..
```

### 2️⃣ Configure Backend Environment

Create `server/.env`:

```env
FLASK_ENV=development
FRONTEND_ORIGINS=http://localhost:3000
```

**Optional environment variables:**
- `MAX_UPLOAD_MB` (default: `4`) - Maximum file upload size
- `FRONTEND_ORIGINS` - Comma-separated list of allowed origins

### 3️⃣ Prepare Machine Learning Model

The backend requires:
- `server/app/ml/dataset/models/simple_classifier.h5`

**First time setup (train the model):**


```bash
# From project root

# Run train2.py
python server\app\ml\notebook\train2.py

```

**Output files after training:**
 
| File | Description |
|------|-------------|
| `dataset/models/simple_classifier.keras` | Main model — recommended for inference |
| `dataset/models/simple_classifier.h5` | Legacy HDF5 format — universally compatible |
| `dataset/models/simple_classifier_meta.json` | Label map + normalisation stats |
| `dataset/reports/simple_train_metrics.txt` | Per-epoch log, top-1 / top-3 / macro-F1 scores |
| `dataset/reports/confusion_matrix.png` | Validation confusion matrix |
| `dataset/reports/sample_predictions.png` | 10-image prediction grid with correctness borders |

### 4️⃣ Start the Application

```bash
# From project root

# Run frontend
cd client && npm run dev

# Run backend
cd server && python run.py
```

This starts:
- **Frontend:** `http://localhost:3000`
- **Backend:** `http://localhost:5000`

### 5️⃣ Verify Backend is Running

Check backend health:

```bash
curl http://localhost:5000/api/health
```

Check if model is ready:

```bash
curl http://localhost:5000/api/v1/ready
```

⚠️ If you get a `503` response, the model file is missing or invalid. Re-run step 3.

---

## 📁 Project Structure

```
traffic-sign-ml/
│
├── package.json                    # ← Root monorepo scripts (npm run dev starts everything)
├── docker-compose.yml              # Container orchestration
├── README.md
├── DEPLOYMENT.md
│
├── client/                         # ── FRONTEND (Next.js 16 + TypeScript) ──────────────
│   ├── public/
│   │   └── labels.json             # ← Traffic sign class labels (48 classes)
│   └── src/
│       ├── app/
│       │   ├── page.tsx            # ← Landing page
│       │   ├── detect/
│       │   │   └── page.tsx        # ← Main detection interface
│       │   ├── layout.tsx
│       │   └── globals.css
│       ├── components/
│       │   ├── image-input/        # ← Core input system (camera + upload)
│       │   │   ├── Image-input.tsx         # Main controller component
│       │   │   ├── UploadView.tsx          # File upload with drag-and-drop
│       │   │   ├── CameraView.tsx          # Live camera feed + detection overlay
│       │   │   ├── TabSelector.tsx         # Upload / Camera tab switcher
│       │   │   ├── DetectionListItem.tsx
│       │   │   ├── CameraPermissionError.tsx
│       │   │   └── utils/
│       │   │       ├── fileUtils.ts        # File validation & conversion
│       │   │       ├── detectionUtils.ts   # ML detection & frame capture
│       │   │       ├── canvasUtils.ts      # Bounding box drawing
│       │   │       └── cameraUtils.ts      # MediaStream API wrapper
│       │   ├── detection-result/   # ← Predictions & confidence scores display
│       │   │   └── DetectionResult.tsx
│       │   ├── detection-history/  # ← Cached detection history
│       │   │   └── DetectionHistory.tsx
│       │   ├── detection-page/
│       │   │   └── DetectionPage.tsx       # Page orchestrator
│       │   ├── landing-page/
│       │   │   └── LandingPage.tsx
│       │   ├── navbar/
│       │   │   └── NavigationBar.tsx
│       │   └── ui/                 # shadcn/ui component library (60+ components)
│       ├── hooks/
│       │   ├── useCamera.ts        # Camera stream management hook
│       │   └── use-mobile.ts
│       ├── lib/
│       │   ├── api.ts              # ← All backend API calls live here
│       │   ├── utils.ts
│       │   └── landing-page.ts
│       └── types/
│           └── prediction.ts       # TypeScript interfaces for predictions
│
└── server/                         # ── BACKEND (Flask + Python) ────────────────────────
    ├── run.py                      # ← Flask app entry point
    ├── requirements.txt            # Python dependencies
    ├── .env                        # Environment variables (not committed)
    ├── Dockerfile
    ├── yolov8n.pt                  # ← YOLOv8 nano pretrained weights
    └── app/
        ├── __init__.py
        ├── api/
        │   ├── __init__.py
        │   └── routes.py           # ← API endpoint definitions
        │                           #   GET  /api/health
        │                           #   GET  /api/v1/ready
        │                           #   POST /api/v1/predict
        │                           #   GET  /api/v1/instruction/<sign_name>
        └── ml/
            ├── predict_service.py      # ← Primary ML inference (scikit-learn classifier)
            ├── yolo_predict_service.py # ← YOLO inference (bounding box detection)
            ├── labels.json             # Sign label definitions
            ├── dataset/
            │   ├── Train_Augmented_Balanced/  # Training images organized by class (1–48)
            │   ├── Test/                      # Test images
            │   ├── Meta/                      # Class metadata images
            │   ├── splits/                    # train/val/test split manifests
            │   ├── scripts/                   # Analysis and inspection scripts
            │   ├── models/                    # ← ⚠️  REQUIRED: trained model lives here
            │   │   └── simple_classifier.h5   # Generated by training script (train2.py)
            │   └── reports/                   # Metrics, confusion matrix, sample predictions
            ├── scripts/
            │   ├── clean-splits.py                 # Step 1: prepare dataset splits
            │   ├── train-simple-classifier.py      # ← Step 2: train the model
            │   ├── train-cnn-classifier.py         # Alternative CNN trainer
            │   └── predict-simple-classifier.py    # Standalone prediction script
            └── notebook/                      # Jupyter notebooks and training experiments
                └── train2.py                  # Final file used to train model

```

---

## 📚 Key Components

### Frontend Components

#### Image Input System (`src/components/image-input/`)
- **ImageInput.tsx** - Main controller component managing both camera and upload modes
- **UploadView.tsx** - File upload interface with drag-and-drop and file browsing
- **CameraView.tsx** - Live camera feed with real-time detection overlays
- **TabSelector.tsx** - Switch between Camera and Upload modes
- **Utils:**
  - `fileUtils.ts` - File validation and conversion
  - `detectionUtils.ts` - ML detection and frame capture
  - `canvasUtils.ts` - Drawing bounding boxes on canvas
  - `cameraUtils.ts` - MediaStream API wrapper

#### Detection System (`src/components/detection-*`)
- **DetectionPage.tsx** - Main detection page orchestrator
- **DetectionResult.tsx** - Display predictions and confidence scores
- **DetectionHistory.tsx** - Cached detection history with persistence

#### Pages (`src/app/`)
- **page.tsx** - Landing page with introduction
- **detect/page.tsx** - Detection interface page

### Backend Components

#### API Routes (`server/app/api/`)
- Health check endpoint
- Image prediction endpoint
- Model status endpoint

#### ML System (`server/app/ml/`)
- **Inference** - Real-time sign classification
- **Training Scripts** - Dataset preparation and model training
- **Models** - Pre-trained scikit-learn classifier
- **Dataset** - Organized training/test/validation splits

---

## 🔄 Data Flow

```
User Upload / Camera
          ↓
    Frontend (Next.js)
          ↓
   ImageInput Component
          ↓
  File Validation & Preview
          ↓
    Backend API (Flask)
          ↓
    YOLOv8 (.pt)
    Locates sign in image
    Returns bounding box
          ↓
  Crop region of interest
          ↓
  Classifier (.h5)
  Identifies what sign it is
  Returns label + confidence
          ↓
  Predictions with Confidence
          ↓
  DetectionResult Component
          ↓
  Display Results + History
```

---

## 🎨 UI Component Library

The project includes 60+ pre-built UI components from shadcn/ui, including:
- Buttons, inputs, cards, dialogs
- Tables, forms, navigation components
- Alerts, badges, tooltips, and more

Located in `client/src/components/ui/`

---

## 🔌 API Endpoints

### Backend (Flask)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Server health check |
| `/api/v1/ready` | GET | Model readiness status |
| `/api/v1/predict` | POST | Predict traffic sign from image |
| `/api/v1/instruction/<sign_name>` | GET | Get safety instruction for sign |

---

### From Project Root

| Command | Purpose |
|---------|---------|
| `npm run lint` | Lint frontend TypeScript/React files |
| `npm run lint:fix` | Auto-fix linting issues |
| `npm run lint:python` | Lint backend Python code (pylint) |
| `npm run type-check` | TypeScript type checking |
| `npm run format` | Format code with Prettier |

### From `client/` Directory

| Command | Purpose |
|---------|---------|
| `npm run dev` | Start frontend on port 3000 |
| `npm run build` | Production build |
| `npm run start` | Start production server |

### From `server/` Directory

| Command | Purpose |
|---------|---------|
| `python run.py` | Start backend on port 5000 |
---

## 💡 Usage

1. Open `http://localhost:3000` in your browser
2. Upload a traffic sign image
3. The app sends it to the backend for analysis
4. View real-time predictions and confidence scores

---

## 📝 Git Conventions

This project uses:
- **Conventional Commits** (via commitlint)
- **Husky hooks** - Prevents invalid commits
- **ESLint + Prettier** - Code formatting consistency

---

## 🐛 Troubleshooting

**Model not loading?**
- Ensure `server/app/ml/models/simple_classifier.joblib` exists
- Run the training script from step 3

**Frontend can't connect to backend?**
- Verify `FRONTEND_ORIGINS` in `server/.env` includes `http://localhost:3000`
- Check that backend is running: `curl http://localhost:5000/api/health`

**Port already in use?**
- Frontend default: 3000 | Backend default: 5000
- Modify in respective config files if needed

**Browse Files not working after upload?**
- This has been fixed in the latest version
- The file input is now always available in the DOM
- You can now browse new files directly from the image preview without clicking Reset

**Camera permission denied?**
- Allow camera access when browser prompts
- Check browser camera permissions in settings
- Some browsers require HTTPS for camera access (except localhost)

**Detection is slow?**
- Check backend is running: `curl http://localhost:5000/api/health`
- Verify network latency between frontend and backend
- Consider optimizing image size before upload

---

## 👤 Authors

**Siaotong, Krystal Jane**  
📧 krystaljane.siaotong-23@cpu.edu.ph  
🔗 [GitHub](https://github.com/xiaokjxiao)

**Tuden, Nelissa**  
📧 nelissa.tuden-23@cpu.edu.ph  
🔗 [GitHub](https://github.com/asilyN)
