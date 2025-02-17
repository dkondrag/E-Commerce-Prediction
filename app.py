from flask import Flask, render_template, request
import joblib
import numpy as np

# Load all models
models = {
    'Linear Regression': joblib.load("lmodel.pkl"),
    'Random Forest': joblib.load("rmodel.pkl"),
    'Gradient Boosting': joblib.load("GBRmodel.pkl"),
    'KNN Model': joblib.load("knnmodel.pkl"),
    'Decision Tree': joblib.load("DTmodel.pkl")
}

app = Flask(__name__, template_folder='temp')

@app.route("/")
def home():
    # Pass the model names to the template to create a dropdown menu
    model_names = list(models.keys())
    return render_template('index.html', model_names=model_names)

def ValuePredictor(to_predict_list, selected_model):
    model = models[selected_model]
    to_predict = np.array(to_predict_list).reshape(1, -1)
    result = model.predict(to_predict)
    return max(0, result[0])

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        selected_model = request.form['model_selection']
        show_prediction = 'show_prediction' in request.form
        show_graphs = 'show_graphs' in request.form

        to_predict_list = request.form.to_dict()
        to_predict_list.pop('model_selection') 
        to_predict_list.pop('show_prediction', None) 
        to_predict_list.pop('show_graphs', None)
        to_predict_list = list(to_predict_list.values())
        to_predict_list = [float(item) for item in to_predict_list]

        result = ValuePredictor(to_predict_list, selected_model)
        prediction_text = f"Predicted Total Price using {selected_model}: {result}" if show_prediction else ""
        
        # Correctly map the selected model to its corresponding graph file prefix
        graph_filename_prefix = {
            'Linear Regression': 'lr',
            'Random Forest': 'rf',
            'Gradient Boosting': 'gbr',
            'KNN Model': 'knn',
            'Decision Tree': 'dt'
        }[selected_model]  # Directly use the selected model as the key

        # Construct the filenames for the graph images
        graph1 = f"{graph_filename_prefix}_graph1.png" if show_graphs else ""
        graph2 = f"{graph_filename_prefix}_graph2.png" if show_graphs else ""

        return render_template("result.html",
                               prediction_text=prediction_text,
                               graph1=graph1,
                               graph2=graph2,
                               show_prediction=show_prediction,
                               show_graphs=show_graphs)

if __name__ == "__main__":
    app.run(debug=True)

