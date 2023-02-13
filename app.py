import streamlit as st
import base64
import torch
from PIL import Image
from models.crnn import CRNN
from models.transformer import Seq2SeqTransformer
from utils.utils import Converter
from utils.utils import Process
from utils.utils import Translation

#Caching the model for faster loading
#@st.cache(suppress_st_warning=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOCAB_PATH = './weights/vocabs.pkl'
vocab_src_path = './weights/vocab_src.pkl'
vocab_tgt_path = './weights/vocab_tgt.pkl'
WEIGHTS_PATH = './weights/crnn_pretrained.pt' 
TRANSFORMER_WEIGHTS = './weights/transformer_pretrained.pt'

EMB_SIZE = 256
NHEAD = 4 
FFN_HID_DIM = 256
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROP_OUT = 0.1

st.set_page_config(
    page_title="Nom Regconition",
    layout="centered"
    )

@st.cache(allow_output_mutation=True)
def load_model():
    converter = Converter(VOCAB_PATH)
    translation = Translation(vocab_src_path, vocab_tgt_path, DEVICE)
    vocab_size = converter.vocab_size()
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = translation.len_vocab()

    # Load crnn model
    crnn = CRNN(vocab_size)
    crnn.load_state_dict(torch.load(WEIGHTS_PATH,map_location = DEVICE))
    crnn = crnn.to(DEVICE)
    # Load transformer model
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM,DROP_OUT)

    transformer.load_state_dict(torch.load(TRANSFORMER_WEIGHTS,map_location = DEVICE)) 
    transformer = transformer.to(DEVICE)

    return crnn, transformer, converter, translation

def get_prediction(model,converter,image):
    process = Process()

    with torch.no_grad():
        model.eval()
        img = process(image)
        img = img.unsqueeze(0)
        logits = model(img.to(DEVICE))
        
    pred_text = converter.decode(logits.cpu())

    return pred_text

def main():
    crnn, transformer, converter, translation = load_model()
    col1, col2 = st.columns([6,5])

    with col1:
        st.image('./assets/nom_header.png')
    with col2:
        st.header(':u5408::u5408: Old Vietnamese Handwritten(Nom) Recognition :u6709::u6709:')

    tab1, tab2 = st.tabs(["Nom Handwritten Recognition", " Nom Translation"])

    with tab1:
        st.header("**:blue[Regconition nom characters from an image]**")
        st.text("Please upload images file with right format call 'Patch'.")
        st.text('Patch is image regions containing text, each of these regions will be cropped')
        st.text('from the original image (like the image above).')
        st.image('./assets/604.jpg')
        st.text("Only access 'jpg, png and jpeg' file format.")
        st.caption(":red[**Note**]: This model only predict a _sequence_ of nom characters not a page.")
        file_uploaded = st.file_uploader('Upload your image here', type = ['jpg', 'png','jpeg'], accept_multiple_files= True)
        
        if file_uploaded:
            for file in file_uploaded:
                slot = st.empty()
                slot.text('Running inference....')

                image = Image.open(file).convert('RGB')
                st.image(image, caption="Input Image")

                result = get_prediction(crnn, converter, image)
                translated = translation.translate(transformer, result)
                output = 'Predicted: ' + result 
                translate_seq = 'Translated: ' + translated
                slot.text('Your image')
                st.success(output)
                st.success(translate_seq)
            
    with tab2:
        st.header(":blue[Translate from Nom to Vietnamese]")
        st.text("Translate Chu Nom sequence to modern Vietnamese.")
        text_input = st.text_input( "Enter your sequence text here 👇",)
        btn = st.button('Translate')

        if btn or text_input:
            slot = st.empty()
            slot.text('Running inference....')

            result = translation.translate(transformer, text_input)
            output = 'Translated: ' + result
            slot.text('Done')
            st.success(output)

        st.markdown("""---""")

        st.text("Translate Chu Nom poem.")
        poem = st.text_area("Enter your poem here 👇", height = 200)
        btn_poem = st.button('Translate Poem')

        if btn_poem or poem:
            sentence = poem.splitlines()
            slot = st.empty()
            slot.text('Running inference....')
            trans_sentence = [translation.translate(transformer, sent) for sent in sentence]
            slot.text('Done')
            for res in trans_sentence:
                st.success(res)

if __name__ == '__main__':
    main()