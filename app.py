from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

db = SQLAlchemy()

def create_app():
   app = Flask(__name__)

   # Configure secret key and database URI from environment variables (recommended)
   app.secret_key = os.getenv("SECRET_KEY", "Deepak@Hemant")  # Default used if env var not set
   app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///site.db")
   app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

   db.init_app(app)
   
   migrate = Migrate(app, db)  # Initialize migrate inside the app factory

   from routes import register_routes
   register_routes(app)
   
   return app


if __name__ == '__main__':
   app = create_app()
   app.run(host='0.0.0.0', port=5000, debug=True)
