from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


file1 = open('model.pkl','rb')
model1 = pickle.load(file1)
file1.close()

file = open('rf_model.pkl','rb')
model = pickle.load(file)
file.close()

@app.route('/')
def index():
    return render_template ('test.html')

@app.route('/',methods=['POST']) 
def result(): 
    if request.method == 'POST': 
        int_feature = [float(x) for x in request.form.values()]
        final_feature = np.array(int_feature).reshape(1,8)
        prediction = model.predict(final_feature)

        output = round(prediction[0],1)
        return render_template("test.html", prediction = "Rating of the app is{}".format(output) )
        
        # return render_template("test.html", prediction1 = int_feature) 

@app.route('/test')
def index1():
    return render_template ('test.html')

# def ValuePredictor(to_predict_list): 
#     to_predict = np.array(to_predict_list).reshape(1, 8) 
#     loaded_model = pickle.load(open("model.pkl", "rb")) 
#     result = loaded_model.predict(to_predict) 
#     return result[0] 



@app.route('/test1')
def test1():
    return render_template ('index_output.html')

@app.route('/dict_data',methods=['POST'])
def predict():

    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        # data = pd.DataFrame(to_predict_list)
        
        for k,v in to_predict_list.items():
            to_predict_list[k] = [float(v)]
        data = pd.DataFrame(to_predict_list)
        # col = ['Category','Reviews','Size','Installs','Type','Price','Content Rating','Genres']
        # to_predict_list = list(to_predict_list.values()) 
        # data = pd.DataFrame(to_predict_list,columns=col)
        # to_predict_list = list(map(int, to_predict_list)) 
        # result = ValuePredictor(to_predict_list) 
        op = model1.predict(data)
        result = op[0]          
        # return render_template("index_output.html", result = "{:.1f}".format(result)  )
        
        return render_template("index_output.html", result = to_predict_list )


@app.route('/predictrf',methods=['POST'])
def predictrf():
    if request.method == 'POST':
        data = request.get_json()
        data = [np.array([float(x) for x in data.values()])]
        pred = model.predict(data)

        return jsonify(pred[0])

@app.route('/predictxg',methods=['POST'])
def predictxg():
    if request.method == 'POST':
        data = request.get_json()
        # for k,v in data.items():
        #     data[k] = [v]
            
        # data = pd.DataFrame(data)
        # data = [np.array([float(x) for x in data.values()])]
        # col = ['Category', 'Reviews', 'Size', 'Installs', 'Type', 'Price', 'Content Rating', 'Genres']
        

        # pred = model1.predict(data)

        return jsonify(data.values())

if __name__ == "__main__":
    app.run(debug=True)
