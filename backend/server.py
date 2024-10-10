from flask import Flask, request, render_template
from recommendationSystem import getRecommendations

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/recommend", methods=["GET"])
def recommend():
    location = request.args.get("location", default="NULL", type=str)
    description = request.args.get("description", default="NULL", type=str)

    if location == "NULL" or description == "NULL":
        return "Invalid parameters"

    return getRecommendations(location, description)


if __name__ == "__main__":
    app.run(debug=True)
