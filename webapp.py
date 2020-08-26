# importing libraries
import numpy as np
import streamlit as st
from PIL import Image
from pred_fromDNN import predict_and_draw

# ignore file upload warnings
st.set_option("deprecation.showfileUploaderEncoding", False)

st.beta_set_page_config(page_title = "ALPR", layout='wide')
# @st.cache
# def cached_model():
    # pass


# model = cached_model()
def main():


    # Side Bar
    st.sidebar.title("Alter Parameters")

    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Show instructions", "Run the app","View Webapp Code" ,"View Prediction Code"])

    # extract prob-thres and NMS_thresh from slider
    confidence_threshold, overlap_threshold = thresh_slider_ui()

    if app_mode == "Show instructions":
        # Title Bar
        st.markdown("<h2 style='text-align: center; color: black;'>Detect License Plates With YoloV4 and Easy OCR</h2>",
                    unsafe_allow_html=True)

        st.markdown("<h3 style='text-align: center; color: black;'>To run inference or see the codes select the Option in Sidebar.</h3>",
                    unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: black;'>You can tune the parameters of Inference using the sliders in Sidebar</h3>",
                    unsafe_allow_html=True)
        st.sidebar.success('To continue select "Run the app".')

    elif app_mode == "Run the app":
        st.markdown("<h2 style='text-align: center; color: black;'>Upload Image File to detect Plate</h2>",
                    unsafe_allow_html=True)
        run_pred(confidence_threshold, overlap_threshold )

    elif app_mode == "View Prediction Code":
        st.markdown("<h2 style='text-align: center; color: black;'>Code for running the Inferences</h2>",
                    unsafe_allow_html=True)
        st.code(get_file_content_as_string("pred_fromDNN.py"))

    elif app_mode == "View Webapp Code":
        st.markdown("<h2 style='text-align: center; color: black;'>Streamlit Code for Web Application</h2>",
                    unsafe_allow_html=True)
        st.code(get_file_content_as_string("webapp.py"))

    st.sidebar.text("Currently in " + app_mode)


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    # url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    # response = urllib.request.urlopen(url)
    with open(path, 'r') as code:
        return code.read()


# This sidebar UI lets the user select parameters for the YOLO object detector.
def thresh_slider_ui():
    st.sidebar.markdown("# Model Thresholds")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.1, 0.05)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.05)
    return confidence_threshold, overlap_threshold


def run_pred(confidence_threshold, overlap_threshold ):
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Before", use_column_width=True)
        st.markdown("<h4 style='text-align: center; color: black;'>Detecting Plates</h4>", unsafe_allow_html=True)

        # calling the prediction function
        result_image = predict_and_draw(image, conf_thres = confidence_threshold,  nms_thresh = overlap_threshold)

        st.image(result_image, caption="Result", use_column_width=True)

if __name__ == "__main__":
    main()