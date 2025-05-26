from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, UserMixin

auth_blueprint = Blueprint("auth", __name__, template_folder="../templates")

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

users = {
    "admin": User(id="1", username="admin", password="admin123")
}

@auth_blueprint.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = users.get(username)
        if user and user.password == password:
            login_user(user)
            return redirect(url_for("views.dashboard"))
        flash("Identifiants invalides", "danger")
    return render_template("login.html")

@auth_blueprint.route("/logout")
def logout():
    logout_user()
    flash("Déconnexion réussie", "info")
    return redirect(url_for("auth.login"))
