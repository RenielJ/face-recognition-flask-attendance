from flask import Flask

def create_app():
    app = Flask(__name__)

    # Set a secret key to enable session handling
    app.config['SECRET_KEY'] = 'FaceSenseTempTrack'

    
    from .auth import auth_bp
    from  .face_temp   import face_temp_bp
    

    app.register_blueprint(auth_bp, url_prefix = '/auth')
    app.register_blueprint(face_temp_bp, url_prefix = '/faceSense&tempTrack')

    return app

    