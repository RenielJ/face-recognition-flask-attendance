from flask import render_template, request, redirect, url_for, session
from . import auth_bp

# Simulated user database
users = {'test@test.com' : '12345'} # Replace with your actual user data

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


@auth_bp.route('/research-dedicate')
def dedication():
    return render_template('dedication.html')

@auth_bp.route('/james-reniel-bambao')
def r_one():
    return render_template('r_one.html')

@auth_bp.route('/rommel-baybayon')
def r_two():
    return render_template('r_two.html')

@auth_bp.route('/annie-rose-cocamas')
def r_three():
    return render_template('r_three.html')

@auth_bp.route('/gail-montallana')
def r_four():
    return render_template('r_four.html')

@auth_bp.route('/fred-luis-macatigos')
def r_five():
    return render_template('r_five.html')

@auth_bp.route('/janine-sinangote')
def r_six():
    return render_template('r_six.html')

@auth_bp.route('/get-started!')
def get_started():
    return render_template("get_started.html")

@auth_bp.route('/temperature-monitoring-tutorial!')
def temp():
    return render_template("temp.html")

