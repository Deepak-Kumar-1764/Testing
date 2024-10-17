from flask import Flask, request, abort
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

db = SQLAlchemy()

ALLOWED_HOSTS = {'127.0.0.1', 'localhost'}  # Define allowed hosts

def create_app():
    app = Flask(__name__)

    # Configure secret key and database URI from environment variables (recommended)
    app.secret_key = os.getenv("SECRET_KEY", "Deepak@Hemant")  # Default used if env var not set
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///site.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    migrate = Migrate(app, db)  # Initialize migrate inside the app factory

    # Check for allowed hosts before every request
    @app.before_request
    def limit_remote_addr():
        # Extract the 'Host' header from the request
        host = request.host.split(':')[0]  # This extracts the domain/IP, ignoring the port

        # If the host is not in ALLOWED_HOSTS, abort the request with 403 Forbidden
        if host not in ALLOWED_HOSTS:
            abort(403)  # Forbidden

    from routes import register_routes
    register_routes(app)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
