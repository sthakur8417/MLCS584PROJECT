
from flask import Flask, render_template, request, jsonify
import prediction
from flask import Flask, render_template, flash, request, redirect,url_for
#from wtforms import Form, TextAreaField, validators, StringField, SubmitField,widgets, SelectMultipleField
#from flask_wtf import FlaskForm


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("home.html")

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

# @app.route('/about')
# def about():
#     pass
#     #return render_template("about.html")


@app.route('/', methods=['POST'])
def predict():
    response = None
    if request.method == 'POST' :
        try:
            ssc_p = request.form['ssc_p'] 
            ssc_board = request.form['ssc_board_dd1']
            hsc_p = request.form['hsc_p'] 
            hsc_board = request.form['hsc_board_dd1']
            hsc_stream = request.form['hsc_stream_dd1'] 
            degree_p = request.form['degree_p']
            degree_stream = request.form['degree_stream_dd1'] 
            etest_per = request.form['etest_per']
            work_exp = request.form['work_exp_dd1'] 
            specialisation = request.form['specialisation_dd1']
            mba_per = request.form['mba_per'] 
            salary = request.form['salary']

            list1=[]
            list1.append(ssc_p)
            list1.append(ssc_board)
            list1.append(hsc_p)
            list1.append(hsc_board)
            list1.append(hsc_stream)
            list1.append(degree_p)
            list1.append(degree_stream)
            list1.append(etest_per)
            list1.append(work_exp)
            list1.append(specialisation)
            list1.append(mba_per)
            list1.append(salary)
            global new_response
            response = prediction.predict(list1)
            print(response)
            message=response
        except Exception as e:
            return respond(e)
    return render_template('home.html', message=message)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='stactic/' + filename), code=301)




def respond(err, res=None):
    return_res =  {
        'status_code': 400 if err else 200,
        'body': err.message if err else res,
    }
    return jsonify(return_res)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
