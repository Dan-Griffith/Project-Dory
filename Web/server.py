from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report.html')
def report():
    return render_template('report.html')

@app.route('/run', methods=['POST'])
def run():
    f = request.files['file']
    print('Run AI')
    print(f.filename)
    return index()

if __name__ == '__main__':
    app.run(debug=True)
