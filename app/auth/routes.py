from flask import render_template, request, redirect, url_for, session
from . import auth_bp

# Simulated user database
users = {'username' : 'password'} # Replace with your actual user data

@auth_bp.route('/', methods=['GET', 'POST'])
def login():

    error_message = None #Default value for error_message

    if request.method == 'POST':
        username=request.form['username']
        password=request.form['password']
        remember = request.form.get('remember', False)  # Use False as the default value

        print("Remember checkbox value:", remember)  # Add this line for debugging

        if users.get(username) == password:
            session['username'] = username
            if remember:
                session.permanent = True  # Mark the session as permanent (cookies will be stored)
            return redirect(url_for('auth.home'))
        else:
            error_message = "Incorrect username or password"
        
    return render_template('login.html', error_message=error_message, title="Login System")


@auth_bp.route('/logout', methods=['GET'])
def logout():
    session.clear()  # Clear the session data
    response = redirect(url_for('auth.login'))  # Redirect to the login page
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


@auth_bp.route('/home')
def home():
    if 'username' in session:
        username = session['username']
        return render_template('home.html', username=username)
    else:
        return redirect(url_for('auth.login'))


