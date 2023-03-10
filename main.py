import streamlit as st
from dataset_utils.preprocessing import letterbox_image_padded
from models.yolov3 import YOLOv3_Darknet53, YOLOv3_Darknet53_ABO, YOLOv3
from PIL import Image
from tog.attacks import *
import matplotlib.colors as mcolors
from PIL import ImageDraw, ImageFont
from enum import Enum
from pillow_heif import register_heif_opener, register_avif_opener
import logging

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


class AdversarialExample:
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


class ModelName(Enum):
    COCO = "COCO"
    ABO = "ABO"


class DetectionModelFactory:
    @classmethod
    def from_model_name(cls, model_name: str) -> YOLOv3:
        if model_name == ModelName.COCO.name:
            return YOLOv3_Darknet53(weights="model_weights/YOLOv3_Darknet53.h5")
        elif model_name == ModelName.ABO.name:
            return YOLOv3_Darknet53_ABO(weights='model_weights/YOLOv3_Darknet53_ABO.h5')


def find_font_size(text, font, image, target_width_ratio):
    tested_font_size = 100
    tested_font = ImageFont.truetype(font, tested_font_size)
    observed_width, observed_height = get_text_size(text, image, tested_font)
    estimated_font_size = tested_font_size / (observed_width / image.width) * target_width_ratio
    return round(estimated_font_size)


def get_text_size(text, image, font):
    im = Image.new('RGB', (image.width, image.height))
    draw = ImageDraw.Draw(im)
    return draw.textsize(text, font)


def draw_images_with_detections(detections_dict):
    colors = list(mcolors.CSS4_COLORS.values())
    images_dict = {}
    width_ratio = 0.2  # Portion of the image the text width should be (between 0 and 1)
    font_family = "arial.ttf"

    for pid, title in enumerate(detections_dict.keys()):
        input_img, detections, model_img_size, classes = detections_dict[title]
        #         print((input_img.reshape(x_query.shape[1:])*255).astype(np.uint8))
        input_img = Image.fromarray((input_img.reshape(input_img.shape[1:]) * 255).astype(np.uint8))
        img_draw_context = ImageDraw.Draw(input_img)

        for box in detections:
            xmin = max(int(box[-4] * input_img.size[0] / model_img_size[1]), 0)
            ymin = max(int(box[-3] * input_img.size[1] / model_img_size[1]), 0)
            xmax = min(int(box[-2] * input_img.size[0] / model_img_size[1]), input_img.size[0])
            ymax = min(int(box[-1] * input_img.size[1] / model_img_size[1]), input_img.size[1])
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            font_size = find_font_size(label, font_family, input_img, width_ratio)
            font = ImageFont.truetype(font_family, font_size)
            img_draw_context.rectangle(xy=[xmin, ymin, xmax, ymax], outline=color, width=4)
            img_draw_context.text(xy=[xmin, ymin], text=label, font=font)
        images_dict[title] = AdversarialExample(title=title, image=input_img)
    return images_dict


def set_initial_session_state():
    # Store the initial value of widgets in session state
    if "detection_model" not in st.session_state:
        st.session_state.detection_model = ModelName.COCO.name


def set_sidebar_select_box():
    st.sidebar.selectbox(
        "Choose Detection Model",
        [ModelName.COCO.name, ModelName.ABO.name],
        index=0,
        key="detection_model"
    )


def main():
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
    detector = DetectionModelFactory.from_model_name(st.session_state.detection_model)

    pil_image = Image.open(uploaded_file)

    # Prepare image for adversarial attack
    total_process = 14

    x_query, x_meta = letterbox_image_padded(pil_image, size=detector.model_img_size)
    image_processing_bar.progress(int(100. / total_process) * 1)
    # initial detection
    detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)

    # apply various attacks
    image_processing_bar.progress(int(100. / total_process) * 2)
    x_adv_fabrication = tog_fabrication(victim=detector, x_query=x_query, n_iter=N_ITER, eps=EPS, eps_iter=EPS_ITER)
    image_processing_bar.progress(int(100. / total_process) * 3)
    x_adv_mislabeling_ml = tog_mislabeling(victim=detector, x_query=x_query, target='ml', n_iter=N_ITER, eps=EPS,
                                           eps_iter=EPS_ITER)
    image_processing_bar.progress(int(100. / total_process) * 4)
    x_adv_mislabeling_ll = tog_mislabeling(victim=detector, x_query=x_query, target='ll', n_iter=N_ITER, eps=EPS,
                                           eps_iter=EPS_ITER)
    image_processing_bar.progress(int(100. / total_process) * 5)
    x_adv_vanishing = tog_vanishing(victim=detector, x_query=x_query, n_iter=N_ITER, eps=EPS, eps_iter=EPS_ITER)
    image_processing_bar.progress(int(100. / total_process) * 6)
    x_adv_untargeted = tog_untargeted(victim=detector, x_query=x_query, n_iter=N_ITER, eps=EPS, eps_iter=EPS_ITER)
    image_processing_bar.progress(int(100. / total_process) * 7)

    # apply detections again after attack
    detections_adv_fabrication = detector.detect(x_adv_fabrication, conf_threshold=detector.confidence_thresh_default)
    image_processing_bar.progress(int(100. / total_process) * 8)
    detections_adv_mislabeling_ml = detector.detect(x_adv_mislabeling_ml,
                                                    conf_threshold=detector.confidence_thresh_default)
    image_processing_bar.progress(int(100. / total_process) * 9)
    detections_adv_mislabeling_ll = detector.detect(x_adv_mislabeling_ll,
                                                    conf_threshold=detector.confidence_thresh_default)
    image_processing_bar.progress(int(100. / total_process) * 10)
    detections_adv_untargeted = detector.detect(x_adv_untargeted, conf_threshold=detector.confidence_thresh_default)
    image_processing_bar.progress(int(100. / total_process) * 11)
    detections_adv_vanishing = detector.detect(x_adv_vanishing, conf_threshold=detector.confidence_thresh_default)
    image_processing_bar.progress(int(100. / total_process) * 12)

    # result
    tab_names = ['Original Image',
                 'Benign (No Attack)',
                 'TOG-fabrication',
                 'TOG-mislabeling (ML)',
                 'TOG-mislabeling (LL)',
                 'TOG-vanishing',
                 'TOG-untargeted']

    # Draw rectangles on images
    adversarial_detections_dicts = {
        tab_names[1]: (x_query, detections_query, detector.model_img_size, detector.classes),
        tab_names[2]: (x_adv_fabrication, detections_adv_fabrication, detector.model_img_size, detector.classes),
        tab_names[3]: (x_adv_mislabeling_ml, detections_adv_mislabeling_ml, detector.model_img_size, detector.classes),
        tab_names[4]: (x_adv_mislabeling_ll, detections_adv_mislabeling_ll, detector.model_img_size, detector.classes),
        tab_names[5]: (x_adv_vanishing, detections_adv_vanishing, detector.model_img_size, detector.classes),
        tab_names[6]: (x_adv_untargeted, detections_adv_untargeted, detector.model_img_size, detector.classes)}

    drawn_images_dict = draw_images_with_detections(adversarial_detections_dicts)
    drawn_images_dict[tab_names[0]] = AdversarialExample(title=tab_names[0],
                                                         image=pil_image)
    image_processing_bar.progress(100)
    image_processing_bar.empty()

    # Tab plotting
    tab_containers = st.tabs(tab_names)
    for i, tab_name in enumerate(tab_names):
        tab_container = tab_containers[i]
        drawn_images_dict[tab_name].streamlit_show(tab_container)
    uploaded_file.close()  # may need to move down


if __name__ == '__main__':
    # CMD ["jupyter-notebook", "--ip=0.0.0.0", "--port=8000", "--allow-root"]
    try:
        main()
    except Exception as e:  # in some cases there are unknown tensorflow issue currently
        logging.error(e)
        st.experimental_rerun()
