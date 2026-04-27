from flask import Flask, jsonify

app = Flask(__name__)

# Estado global simple
estado = {"power": "OFF"}

@app.route("/on", methods=["GET"])
def encender():
    estado["power"] = "ON"
    return jsonify({"status": "ok", "power": estado["power"]})

@app.route("/off", methods=["GET"])
def apagar():
    estado["power"] = "OFF"
    return jsonify({"status": "ok", "power": estado["power"]})

@app.route("/status", methods=["GET"])
def status():
    return jsonify(estado)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=28080, debug=True)