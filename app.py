from flask import Flask, redirect, url_for
from flask_login import LoginManager
from auth.routes import auth_blueprint, users
from views.routes import views_blueprint

app = Flask(__name__)
app.secret_key = "change_me"

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return users.get("admin") if user_id == "1" else None

app.register_blueprint(auth_blueprint)
app.register_blueprint(views_blueprint)

@app.route("/")
def home():
    return redirect(url_for("views.dashboard"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

