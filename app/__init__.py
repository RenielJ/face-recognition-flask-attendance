from flask import Flask
# from flask_login import LoginManager
from .auth.routes import User
def create_app():
    app = Flask(__name__)

    # Set a secret key to enable session handling
    app.config['SECRET_KEY'] = 'FaceSenseTempTrack'

    # login_manager = LoginManager()
    # login_manager.init_app(app)
    
    from .auth import auth_bp
    from .face_temp import face_temp_bp

    # # Define the user loader callback
    # @login_manager.user_loader
    # def load_user(user_id):
    #     # Replace this with your actual user loading code
    #     return User.query.get(int(user_id))


    app.register_blueprint(auth_bp, url_prefix = '/auth')
    app.register_blueprint(face_temp_bp, url_prefix = '/faceSense&tempTrack')

    return app

    