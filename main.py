import streamlit as st
from PIL import Image
from PIL import ImageDraw, ImageFont
import matplotlib.colors as mcolors
from enum import Enum
from src.models import ModelName, model_path_dict
from src.models.factory import ModelFactory, run_downloads
from src.attacks.tog import TOG, transform_to_prediction_tensor, transform_to_image, clean_env_after_one_run
from src.attacks import TOGAttacks
from pillow_heif import register_heif_opener, register_avif_opener
from ultralytics.yolo.data.augment import LetterBox
import numpy as np
import logging
import cv2

EPS = 8 / 255.  # Hyperparameter: epsilon in L-inf norm
EPS_ITER = 2 / 255.  # Hyperparameter: attack learning rate
N_ITER = 10  # Hyperparameter: number of attack iterations


def write_title():
    new_title = '<p style="font-size: 42px;">Welcome to my Adversarial Attack and Object Detection App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)
    read_me = st.markdown("""
    This project was built using Streamlit and Keras 
    to demonstrate YOLO Object detection (Darknet) and TOG adversarial attack.
    
    The COCO-based YOLO object Detection project can detect 80 types of object (classes). The full list of the classes can be found 
    [here](https://github.com/KaranJagtiani/YOLO-Coco-Dataset-Custom-Classes-Extractor/blob/main/classes.txt)
    
    The ABO-based YOLO object Detection project can detect 4 types of object on the seas. The types are Powerboat, Sailboat, Ship, and Stationary. 
    
    Detail of TOG attack can be found 
    [here](https://github.com/git-disl/TOG)""")


class ImageStContainer:
    def __init__(self,
                 title,
                 image):
        self.title = title
        self.image = image

    def streamlit_show(self,
                       st_container):
        with st_container:
            st.title(self.title)
            st.image(self.image)


def set_initial_session_state():
    # Store the initial value of widgets in session state
    if "surrogate_model" not in st.session_state:
        st.session_state.surrogate_model = ModelName.YOLOv8ABO.value
    if "target_model" not in st.session_state:
        st.session_state.target_model = ModelName.YOLOv8ABO.value


def set_sidebar_select_box():
    st.sidebar.selectbox(
        "Choose Surrogate Model",
        ModelName.get_all_model_names_in_str(),
        index=0,
        key="surrogate_model"
    )
    st.sidebar.selectbox(
        "Choose Target Model",
        ModelName.get_all_model_names_in_str(),
        index=0,
        key="target_model"
    )


def main():
    clean_env_after_one_run()
    run_downloads()
    set_initial_session_state()
    set_sidebar_select_box()
    write_title()
    uploaded_file = st.file_uploader(
        "Choose an image to upload: {}".format(" ".join(["png", "jpg", "heic", "avif", "heif", "jpeg"])),
        type=["png", "jpg", "heic", "avif", "heif", "jpeg"])

    # uploading section
    show_uploaded_image = st.empty()

    if not uploaded_file or uploaded_file is None:
        show_uploaded_image.info(
            "Please choose an image to upload: {}".format(" ".join(["png", "jpg", "heic", "avif", "heif", "jpeg"])))
        return
    # garbage clean up
    import gc
    for name in dir():
        if name == 'tab_container':
            del name
    gc.collect()
    if 'heif' in uploaded_file.name.lower() or 'heic' in uploaded_file.name.lower():
        register_heif_opener()
    elif 'avif' in uploaded_file.name.lower():
        register_avif_opener()

    # PIL LOAD
    image_processing_bar = st.progress(1)
    logging.info("start detection! surrogate: {}, target: {}".format(
        st.session_state.surrogate_model,
        st.session_state.target_model
    ))
    surrogate_mn = ModelName.from_str(st.session_state.surrogate_model)

    detector = ModelFactory.from_model_name(st.session_state.target_model)
    tog = TOG.from_weights_file_yolo(model_path_dict[surrogate_mn])

    pil_image = Image.open(uploaded_file).convert('RGB')
    pil_np = LetterBox(scaleFill=True)(image=np.array(pil_image))
    tab_names = ['Original Image']
    images = [pil_image]

    resized_pil = Image.fromarray(pil_np)
    tab_names.append('Resized Image')
    images.append(resized_pil)

    # Prepare image for adversarial attack
    total_process = 5
    i = 1
    benign_result = detector.predict(pil_image)[0]

    benign_pil = benign_result.plot(im=cv2.cvtColor(np.array(resized_pil), cv2.COLOR_BGR2RGB))
    tab_names.append('Benign (No Attack)')
    images.append(benign_pil)

    image_processing_bar.progress(int(100. / total_process) * i)
    i += 1

    ori_tensor = transform_to_prediction_tensor(resized_pil)
    batch = {'img': ori_tensor,
             'im_file': [''],
             # 'ori_shape': [ori_tensor.shape[2:]],
             # 'resized_shape': [ori_tensor.shape[2:]],
             # 'ratio_pad': [[[1.0, 1.0], [0.0, 0.0]]]
             }
    if batch['img'].shape[1] != 3:
        raise Exception('shape is wrong! {}'.format(batch['img'].shape))
    fabrication_tensor = tog(batch, None, mode=TOGAttacks.fabrication)[0].to('cpu')
    fabrication_result = detector.predict(transform_to_image(fabrication_tensor))[0]
    fabrication_toplot = cv2.cvtColor(np.array(transform_to_image(fabrication_tensor)), cv2.COLOR_BGR2RGB)
    fabrication_pil = Image.fromarray(fabrication_result.plot(im=fabrication_toplot), mode='RGB')
    tab_names.append('Fabrication')
    images.append(fabrication_pil)
    image_processing_bar.progress(int(100. / total_process) * i)
    i += 1
    # vanishing_tensor = tog(batch, None, mode=TOGAttacks.vanishing)[0]
    # vanishing_result = detector.predict(transform_to_image(vanishing_tensor))[0]
    # vanishing_pil = vanishing_result.plot(im=resized_pil)
    # tab_names.append('Vanishing')
    # images.append(vanishing_pil)
    # i += 1
    # image_processing_bar.progress(int(100. / total_process) * i)

    # result
    image_processing_bar.progress(100)
    image_processing_bar.empty()

    # Tab plotting
    tab_containers = st.tabs(tab_names)
    for i, tab_name in enumerate(tab_names):
        tab_container = tab_containers[i]
        ImageStContainer(tab_names[i], images[i]).streamlit_show(tab_container)
        # img_dicts[tab_name].streamlit_show(tab_container)
    uploaded_file.close()  # may need to move down


if __name__ == '__main__':
    # CMD ["jupyter-notebook", "--ip=0.0.0.0", "--port=8000", "--allow-root"]
    try:
        main()
    except Exception as e:  # in some cases there are unknown tensorflow issue currently
        logging.error(e)
        st.experimental_rerun()
