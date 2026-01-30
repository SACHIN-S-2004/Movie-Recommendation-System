from flask import Flask, request, jsonify, render_template
from model import recommend_by_movie, recommend_by_genre

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend/movie", methods=["POST"])
def movie_recommendation():
    movie_name = request.json.get("movie_name", "")
    return jsonify(recommend_by_movie(movie_name))

@app.route("/recommend/genre", methods=["POST"])
def genre_recommendation():
    genre = request.json.get("genre", "")
    return jsonify(recommend_by_genre(genre))

if __name__ == "__main__":
    app.run(debug=True)
