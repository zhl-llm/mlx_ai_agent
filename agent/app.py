from flask import Flask, render_template, request
from main import run_agent

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.form['query']
    result = run_agent(user_query)
    return result

if __name__ == '__main__':
    app.run(debug=True)

