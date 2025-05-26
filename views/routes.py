from flask import Blueprint, render_template
from flask_login import login_required

views_blueprint = Blueprint("views", __name__, template_folder="../templates")

@views_blueprint.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard_streamlit.html")
