import os

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    from app import create_app

    app = create_app()
    debug = os.getenv("FLASK_ENV") == "development"
    port = int(os.getenv("PORT", 5000))
    app.run(debug=debug, host="0.0.0.0", port=port)
