from flask import Flask, render_template, request,jsonify
from preprocess_data import Question


app = Flask(__name__)
que = Question()



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():

    userText = request.args.get('msg')
    print_answer,*_ = que.question_process(userText,flag='msg')
    print('print_answer',print_answer)
    return print_answer

@app.route("/chart")
def get_chart():

    userText = request.args.get('data')
    print('userText',userText)

    print_answer,chart,template_id,cql = que.question_process(userText,flag='chart')
    return jsonify(print_answer=print_answer,
                   chart=chart,
                   template_id=template_id,
                   cql=cql)

def dealquestion(question):
    # 查询知识图谱
    return que.question_process(question)

if __name__ == "__main__":
    app.run()
