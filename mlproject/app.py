import gradio 
import pickle 
import pandas as pd 
import numpy as np 
global vectorizer, model 
vec_file, mod_file = open('vectorizer.pkl', 'rb'), open('model.pkl', 'rb')
filtered_df = pd.read_csv('filtered_df.csv')

vectorizer = pickle.load(vec_file)
model = pickle.load(mod_file)

vec_file.close()
mod_file.close()
def recommend(description): 
    vector = vectorizer.transform(np.array([description]))
    category = model.predict(vector)
    return filtered_df[filtered_df.Category.isin(category)].to_html()

app = gradio.Interface(fn = recommend, inputs = ['text'], outputs=gradio.HTML())
app.launch(share = True)