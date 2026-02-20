import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Earthquake Building Damage Prediction",
    layout="centered"
)

st.markdown(
    """
<style>
/* ================= Background ================= */
.stApp {
    background:
        linear-gradient(
            rgba(255, 200, 210, 0.6),
            rgba(255, 200, 210, 0.6)
        ),
        url("https://upload.wikimedia.org/wikipedia/commons/b/b4/Flag_of_Turkey.svg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ================= All text black ================= */
html, body, label, span, p, div,
h1, h2, h3, h4, h5, h6 {
    color: black !important;
}

/* ================= Main white blocks ================= */
section[data-testid="stVerticalBlock"] {
    background-color: white !important;
    padding: 20px;
    border-radius: 12px;
}

/* ================= Inputs (general) ================= */
input, textarea {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ccc !important;
}

/* ================= SELECTBOX (BaseWeb – FIXED) ================= */
div[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 10px;
}

div[data-baseweb="select"] > div {
    background-color: white !important;
    color: black !important;
    
}

/* when clicked / focused */
div[data-baseweb="select"] > div:focus,
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="select"][aria-expanded="true"] > div {
    background-color: white !important;
    color: black !important;
    border: 1px solid #800020 !important;
}

/* dropdown list */
ul[role="listbox"] {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ccc !important;
}

/* dropdown options */
li[role="option"] {
    background-color: white !important;
    color: black !important;
    border: 1px solid #ccc !important;
}

li[role="option"]:hover {
    background-color: #f2f2f2 !important;
    color: black !important;
    border: 1px solid #ccc !important;
}

/* ================= Buttons ================= */
button {
    background-color: white !important;
    color: black !important;
    border: 1px solid #aaa !important;
}

button:hover {
    background-color: #f2f2f2 !important;
    border: 1px solid #ccc !important;
}

/* number input +/- */
div[data-testid="stNumberInput"] button {
    background-color: white !important;
    color: black !important;
    border: 1px solid #aaa !important;
}

/* ================= Metrics ================= */
div[data-testid="metric-container"] {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
    padding: 15px;
    border: 1px solid #ddd;
}
/* ================= Expander (FULL FIX) ================= */

/* whole expander box */
details {
    background-color: white !important;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #ddd !important;
}

/* expander title (Show model comparison results) */
summary {
    background-color: white !important;
    color: black !important;
    padding: 10px;
    border-radius: 10px;
    font-weight: 600;
}

/* remove default arrow background effect */
summary:hover {
    background-color: #f2f2f2 !important;
    color: black !important;
}

/* expander content */
details > div {
    background-color: white !important;
    color: black !important;
}
    """,
    unsafe_allow_html=True
)

col_flag, col_info = st.columns([1, 4])

with col_flag:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/b/b4/Flag_of_Turkey.svg",
        width=80
    )

with col_info:
    st.markdown(
        """
        **Predicting Building Collapse After an Earthquake**  
        **Made By:**  
        Mohammad Hosseini  
        Mohammad Mahan Haghi  
        Kourosh Ameri Far  
        Seyed Mohammadparsa Azimi
        """
    )

st.markdown("---")


st.markdown("## 🏗️ Earthquake Building Damage Prediction")
st.write("Predict structural damage based on building and site characteristics")

with open("models/model_forest_classifier.pickle", "rb") as f:
    clf_model = pickle.load(f)

with open("models/model_tree_regressor.pickle", "rb") as f:
    reg_model = pickle.load(f)

data = pd.read_csv("building_damage.csv")
data = data.drop("Unnamed: 0", axis=1)

occ_type_display = {
    "Residential": ["RES1", "RES3", "RES4"],
    "Commercial": ["COM1", "COM2", "COM3", "COM4", "COM7", "COM8"],
    "Industrial": ["IND1", "IND2", "IND3"],
    "Agricultural": ["AGR1"],
    "Educational": ["EDU1"],
    "Religious": ["REL1"],
    "Governmental": ["GOV1"]
}

encoders = {}
for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

struct_type_display = {
    "Unreinforced Masonry (URM)": "URM",
    "Steel Moment Frame (S1)": "S1",
    "Reinforced Concrete Moment Frame (C4)": "C4",
    "Wooden Frame (W1)": "W1",
    "Precast Concrete (PC1)": "PC1",
    "Reinforced Concrete Shear Wall (C1)": "C1",
}

st.subheader("🔢 Input Features")

struct_display_choice = st.selectbox(
    "Structural Type",
    list(struct_type_display.keys())
)
struct_typ = struct_type_display[struct_display_choice]

occ_choice = st.selectbox(
    "Occupancy Type",
    list(occ_type_display.keys())
)
occ_type_code = occ_type_display[occ_choice][0]

year_built = st.number_input("Year Built", 1985, 2017, 2000)
no_stories = st.number_input("Number of Stories", 0, 30, 0)
magnitude = st.number_input("Earthquake Magnitude", value=5.0)
distance = st.number_input("Distance from Epicenter (km)", value=3.0)

X = np.array([[
    encoders["struct_typ"].transform([struct_typ])[0],
    encoders["occ_type"].transform([occ_type_code])[0],
    year_built,
    no_stories,
    magnitude,
    distance
]])

if st.button("🚀 Predict Damage"):
    meandamage_pred = reg_model.predict(X)[0]
    damage_class_pred = clf_model.predict(X)[0]

    damage_map = {
        0: "🟢 Safe",
        1: "🟠 High Risk",
        2: "🔴 Collapsed"
    }

    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Damage Index", round(float(meandamage_pred), 4))
    with col2:
        st.metric("Damage Class", damage_map[int(damage_class_pred)])

    st.success("Prediction completed successfully ✅")

st.markdown("---")
st.subheader("📈 Model Performance Summary")

with st.expander("Show model comparison results", expanded=True):
    st.image(
        "model_result.jpg",
        caption="Classification & Regression Model Performance",
        use_container_width=True
    )

st.markdown("---")