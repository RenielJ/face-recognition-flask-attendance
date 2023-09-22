from flask import render_template
from app import create_app
from datetime import timedelta

# def app for initialiation of server
app = create_app()

# Define the custom error handlers
@app.errorhandler(404) # 404 is a error url,
def page_not_found(e):
    return render_template('404.html', title="404 - Error"), 404

@app.errorhandler(405) 
def method_not_allowed(e):
    return render_template('405.html', title="405 - Error"), 405

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html', title="500 - Error"), 500

# Configure session to last for 7 days (or any desired duration)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

