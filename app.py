from flask import Flask,render_template
from newsscraper import get_news,get_titles
app = Flask("__WorldTime__")

@app.route("/")
def root():
    channels = get_news()
    titles = get_titles()
    return render_template("index.html",channels = channels,titles = titles)

app.run(host='0.0.0.0')