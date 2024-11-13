from . import app
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import json
import matplotlib.pyplot as plt
from flask import render_template, request

@app.route("/", methods=["GET", "POST"])
def crop_doc():
    if request.method == "POST":
        file = request.files["cropdocFile"]
        print(file.filepath)
    return render_template("cropdoc.html")

