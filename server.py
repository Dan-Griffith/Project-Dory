from flask import Flask, render_template, request


app = Flask(__name__)

#presentation
@app.route('/')
def index():
    return render_template('index.html')

#report document
@app.route('/report.html')
def report():
    return render_template('report.html')



#run model
@app.route('/run', methods=['POST'])
def run():
    f = request.files['file']
    print('model running...')
    print(f.filename)

    #send filename to model
    

    #get predicted label from model
    predicted_label = 0

    if predicted_label == 1:
        return render_template('report.html')
    elif predicted_label == 2:
        return render_template('report.html')
    elif predicted_label == 2:
        return render_template('report.html')
    else:
        return index()

if __name__ == '__main__':
    app.run(debug=True)
