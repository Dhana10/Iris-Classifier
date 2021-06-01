from flask import Flask , render_template , request
import joblib
app=Flask(__name__)
model=joblib.load('weights.sav')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction' , methods=['POST'])
def funtion():
    sln=float(request.form['sl'])
    swn=float(request.form['sw'])
    pln=float(request.form['pl'])
    pwn=float(request.form['pw'])
    output=model.predict([[sln,swn,pln,pwn]])
    name=['setosa','virginica','vericolor']

    return render_template('index.html',pred=name[output[0]])






if __name__ == "__main__":
    app.run(debug=True)