import streamlit as st
import pickle
import numpy as np

# Title
st.title("Financial Inclusion Prediction")

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))


# Input fields for user data
st.sidebar.header("Input Features")
input_data ={}

input_data['country'] = st.sidebar.selectbox('Country', ['Kenya', 'Rwanda', 'Tanzania', 'Uganda']    )         

input_data['year'] = st.sidebar.selectbox('Year', [2016, 2017, 2018])

input_data['location_type'] = st.sidebar.selectbox('Location type', ['Rural', 'Urban'])
                            
input_data['cellphone_access']= st.sidebar.checkbox('Cellphone access')                          

input_data['household_size'] =st.sidebar.number_input("Household size", min_value=1)

input_data["age"] = st.sidebar.number_input("age", min_value=16)

input_data["marital_status"] =  st.sidebar.selectbox('Marital status', ['Married/Living together','Single/Never Married', 'Widowed',  'Divorced/Seperated'])

input_data["education_level"] =  st.sidebar.selectbox('Marital status', ["Primary education", "No formal education", "Secondary education", "Tertiary education", "Vocational/Specialised training", "Other/Dont know/RTA"])

input_data["job_type"] = st.sidebar.selectbox("Job type", ["Self employed", "Informally employed", "Farming and Fishing", "Remittance Dependent", "Other Income", "Formally employed Private", "No Income", "Formally employed Government", "Government Dependent", "Dont Know/Refuse to answer"])
def encode(data_point):
    encoding_maps = {
        'country': {'Kenya': 0, 'Rwanda': 1, 'Tanzania': 2, 'Uganda': 3},
        'location_type': {'Rural': 0, 'Urban': 1},
        'cellphone_access': {False: 0, True: 1},
        'marital_status': {
            'Married/Living together': 0,
            'Single/Never Married': 1,
            'Widowed': 2,
            'Divorced/Seperated': 3
        },
        'education_level': {
            "Primary education": 0,
            "No formal education": 1,
            "Secondary education": 2,
            "Tertiary education": 3,
            "Vocational/Specialised training": 4,
            "Other/Dont know/RTA": 5
        },
        'job_type': {
            "Self employed": 0,
            "Informally employed": 1,
            "Farming and Fishing": 2,
            "Remittance Dependent": 3,
            "Other Income": 4,
            "Formally employed Private": 5,
            "No Income": 6,
            "Formally employed Government": 7,
            "Government Dependent": 8,
            "Dont Know/Refuse to answer": 9
        }
    }
    encoded_data = {}
    for key, value in data_point.items():
        if key in encoding_maps:
            if value in encoding_maps[key]:
                encoded_data[key] = encoding_maps[key][value]
            else:
                # Handle cases where the value is not in the encoding map
                print(f"Warning: Value '{value}' for key '{key}' not found in encoding map. Skipping.")
                # Example: Set to None or use a default value
                # encoded_data[key] = None
        else:
            # Keep original value if no encoding map exists
            encoded_data[key] = value
    return encoded_data

if st.button("Predict bank account ownership"):
    input_data = encode(dict(input_data))
    input_data = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    st.write(f"Does this person own a bank account prediction: {'Yes' if prediction == 1 else 'No'}")



