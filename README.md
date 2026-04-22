# Traffic Sign Recognition ML

A machine learning project for traffic sign recognition with a Next.js frontend and Flask backend API.

## Tech Stack

- **Frontend**: Next.js 16, React 19, TypeScript, Tailwind CSS
- **Backend**: Flask, Python 3
- **ML Libraries**: OpenCV, scikit-learn, NumPy, Pandas, Pillow

## Prerequisites

- Node.js 18+ and npm
- Python 3.8+
- Git

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd traffic-sign-ML
```

### 2. Install Root Dependencies

```bash
npm install
```

### 3. Install Frontend Dependencies

```bash
cd client
npm install
cd ..
```

### 4. Install Backend Dependencies

```bash
cd server
pip install -r requirements.txt
cd ..
```

## Running the Project

### Option 1: Run Both Frontend and Backend Concurrently (Recommended)

```bash
npm run dev
```

This command will start both:

- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:5000

### Option 2: Run Frontend Only

```bash
cd client
npm run dev
```

Access the frontend at http://localhost:3000

### Option 3: Run Backend Only

```bash
cd server
python run.py
```

The API will be available at http://localhost:5000

### Option 4: Production Build

Frontend:

```bash
cd client
npm run build
npm start
```

## Available Scripts

### Root Scripts

- `npm run dev` - Run frontend and backend concurrently (development mode)
- `npm run dev:backend` - Run backend server only
- `npm run lint` - Lint TypeScript/TSX files
- `npm run lint:fix` - Fix linting issues
- `npm run lint:python` - Lint Python files
- `npm run format` - Format code with Prettier
- `npm run type-check` - Type check TypeScript

### Frontend Scripts (from `client/` directory)

- `npm run dev` - Start Next.js development server
- `npm run build` - Build for production
- `npm run start` - Start production server

## Project Structure

```
.
├── client/              # Next.js frontend application
│   ├── src/
│   │   ├── app/        # Next.js app directory
│   │   ├── components/ # React components
│   │   ├── hooks/      # Custom React hooks
│   │   ├── services/   # API services
│   │   └── types/      # TypeScript types
│   └── package.json
├── server/             # Flask backend application
│   ├── app/
│   │   ├── api/       # API routes
│   │   └── ml/        # ML models
│   ├── requirements.txt
│   └── run.py
├── public/            # Static assets
└── package.json
```

## Environment Setup

Create a `.env.local` file in the `server/` directory if needed for environment variables:

```bash
FLASK_ENV=development
```

## Development Tips

- Frontend auto-reloads on file changes
- Backend will reload if debug mode is enabled in environment
- Use `npm run lint:fix` to auto-fix code style issues
- Run `npm run type-check` before committing to catch TypeScript errors

## Additional Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
