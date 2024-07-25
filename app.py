import streamlit as st
import joblib
import pandas as pd

# from pathlib import Path
# st.text(Path.cwd()) 

# Load the preprocessor and best model
preprocessor = joblib.load('preprocessor.pkl')
best_model = joblib.load('best_model.pkl')

# Load your dataset
df = pd.read_csv('data.csv')

# Drop 'spec_rating' and 'price' columns
X = df.drop(columns=['spec_rating', 'price'])

# Define the unique values for each feature
unique_values = {}
for col in X.columns:
    unique_values[col] = X[col].unique().tolist()

def predict_laptop_price(brand, name, cpu, processor, ram, ram_type, storage, storage_type, gpu, display_size, resolution_width, resolution_height, os, warranty):
    input_data = pd.DataFrame({
        'brand': [brand],
        'name': [name],
        'processor': [processor],
        'CPU': [cpu],
        'Ram': [ram],
        'Ram_type': [ram_type],
        'ROM': [storage],
        'ROM_type': [storage_type],
        'GPU': [gpu],
        'display_size': [display_size],
        'resolution_width': [resolution_width],
        'resolution_height': [resolution_height],
        'OS': [os],
        'warranty': [warranty]
    })

    input_data_transformed = preprocessor.transform(input_data)
    price_prediction = best_model.predict(input_data_transformed)[0]

    return price_prediction

# Set up the Streamlit app
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")
st.title("💻 Laptop Price Predictor")
st.subheader("Enter laptop details to get predicted price")


# Define the input fields
with st.sidebar:
    st.header("Input Features")
    brand = st.selectbox("Select Brand", unique_values['brand'])
    if brand:
        laptops_brand = df[df['brand'] == brand]['name'].tolist()
        name = st.sidebar.selectbox("Select Laptop Name", laptops_brand)
    processor = st.selectbox("Select Processor", unique_values['processor'])
    cpu = st.selectbox("Select CPU", unique_values['CPU'])
    ram = st.selectbox("Select RAM", unique_values['Ram'])
    ram_type = st.selectbox("Select RAM Type", unique_values['Ram_type'])
    storage = st.selectbox("Select Storage", unique_values['ROM'])
    storage_type = st.selectbox("Select Storage Type", unique_values['ROM_type'])
    gpu = st.selectbox("Select GPU", unique_values['GPU'])
    display_size = st.selectbox("Select Display Size", unique_values['display_size'])
    resolution_width = st.number_input("Enter Resolution Width", min_value=float(min(unique_values['resolution_width'])), step=1.0)
    resolution_height = st.number_input("Enter Resolution Height", min_value=float(min(unique_values['resolution_height'])), step=1.0)
    os = st.selectbox("Select Operating System", unique_values['OS'])
    warranty = st.selectbox("Select Warranty", unique_values['warranty'])

# Predict the laptop price
if st.button("Predict Laptop Price"):
    results = predict_laptop_price(
        brand, name, cpu, processor, ram, ram_type, storage, storage_type, gpu, display_size, resolution_width, resolution_height, os, warranty
    )
    st.write(f"Predicted Laptop Price: {results*3.34:.0f}rs")

## About ME:
st.header("About Us:")
st.write("Hi there, We Are ML Masters!👋")
st.write("CS is all the glamour nowadays and everyone wants to enter the field in one way or another. A laptop is a necessity to acheive that success. However, it's hard to figure out the right price for the laptop. We went through that problem and have created a predictor to get accurate prices for laptops of vaious brands and models")

# Social Media Links
st.header("Let's Connect:")
st.markdown("Abheesh")
st.markdown(
    "[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abheesh-kumar-194b8014a/) "
    "[![Github](https://img.shields.io/badge/Github-%23FF0000.svg?logo=Github&logoColor=Black)](https://github.com/AbheeshKumar)"
)
st.markdown("Sandesh Kukreja")
st.markdown(
    "[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sandesh-kukreja/) "
    "[![Github](https://img.shields.io/badge/Github-%23FF0000.svg?logo=Github&logoColor=Black)](https://github.com/AbheeshKumar)"
)
st.markdown("Neeraj Kumar")
st.markdown(
    "[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/neeraj-kumar-63401325a/) "
    "[![Github](https://img.shields.io/badge/Github-%23FF0000.svg?logo=Github&logoColor=Black)](https://github.com/Neerajk43/)"
)
