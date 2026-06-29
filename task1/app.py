from flask import Flask,render_template,request,jsonify
from chatbot import chatbot_response

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get",methods=["POST"])
def get_response():

    data=request.get_json()

    message=data["message"]

    response=chatbot_response(message)

    return jsonify({"response":response})

if __name__=="__main__":
    app.run(debug=True)