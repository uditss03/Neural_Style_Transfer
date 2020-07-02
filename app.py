import os
from flask import Flask, render_template, request


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    if not os.path.isdir(target):
        os.mkdir(target)
    for f in os.listdir(target):
        os.remove(os.path.join(target,f))
    content_img = request.files.get("content_file")
    style_img = request.files.get("style_file")

    content_name = content_img.filename
    style_name = style_img.filename

    content_path = "/".join([target, content_name])
    style_path = "/".join([target, style_name])
    
    content_img.save(content_path)
    style_img.save(style_path)

    images_names = os.listdir('./static')

    return render_template("upload.html", content_name=content_name, style_name=style_name)



if __name__ == "__main__":
    app.run(debug=True)
