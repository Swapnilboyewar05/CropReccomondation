import numpy as np
from PIL import Image
import streamlit as st
import pickle
import numpy
decision_model = pickle.load(open('DecisiontreeModel.pkl','rb'))
logistic_model = pickle.load(open('LogisticRegressionModel.pkl','rb'))
random_model = pickle.load(open('RandomForestModel.pkl','rb'))
st.write("CROP PREDICTION PROJECT")
st.title("CROP RECOMMONDATION SYSTEM")
image = Image.open('crop.jpg')
st.image(image,'crop')
activites =['Decision Tree',"Logistic Regression","Random Forest"]
option = st.sidebar.selectbox("Which model you want to use",activites)
st.subheader(option)
Nitrogen = st.text_input("Nitrogen","Type here",key="1")
Potassium = st.text_input("Potassium",'Type here',key="2")
phosphorus = st.text_input("phosphorus","Type here",key="3")
Temparture = st.text_input("Temparture","Type here",key="4")
Humidity = st.text_input("Humidity","Type here",key="5")
pH = st.text_input("ph","Type here",key="6")
Rainfall = st.text_input("Rainfall","Type here",key="7")
def predict_crop(Nitrogen,Potassium,phosphorus,Temparture,Humidity,pH,Rainfall):
    input = np.array([[Nitrogen,Potassium,phosphorus,Temparture,Humidity,pH,Rainfall]]).astype(np.float64)
    if option == "Decision Tree":
        prediction = decision_model.predict(input)
    elif option == "Logistic Regression":
        prediction = logistic_model.predict(input)
    else :
        prediction = random_model.predict(input)
    return prediction[0]
if st.button("Predict your crop"):
            output=predict_crop(Nitrogen,Potassium,phosphorus,Temparture,Humidity,pH,Rainfall)
            res = "“"+ output.capitalize() + "”"
            st.success('The most suitable crop for your field is {}'.format(res))

