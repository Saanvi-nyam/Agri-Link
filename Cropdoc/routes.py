from . import app
from flask import render_template

@app.route("/", methods=["GET", "POST"])
def crop_doc():
    return render_template("cropdoc.html")

