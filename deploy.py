import streamlit as st
import torch
import torchaudio
import os
from models.vits import Vits, VitsArgs, VitsAudioConfig, CharactersConfig
from utils.text.tokenizer import TTSTokenizer
from configs.vits_config import VitsConfig
import numpy as np
import tempfile

# Set page config
st.set_page_config(page_title="Vietnamese TTS Demo", layout="wide")
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"
@st.cache_resource
def load_model():
    """Load the TTS model and configurations"""
    # Audio config
    audio_config = VitsAudioConfig(
        sample_rate=16000,
        hop_length=256,
        win_length=1024,
        fft_size=1024,
        mel_fmin=0.0,
        mel_fmax=None,
        num_mels=80,
    )

    # Model arguments
    model_args = VitsArgs(
        d_vector_file=["data/speakers.pth"],
        use_d_vector_file=True,
        d_vector_dim=512,
        speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
        speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
        num_layers_text_encoder=10,
        resblock_type_decoder="2",
    )

    # Characters config
    chars_config = CharactersConfig(
        characters_class="models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters="1234567890" + 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" +
        "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúý" +
        "ĂăĐđĨĩŨũƠơƯư" +
        "ẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệ" +
        "ỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰự" +
        "ỲỳỴỵỶỷỸỹ" +
        "\u00af\u00b7\u00df\u00e6\u00f1\u0105\u0107\u0113\u0119\u011b\u012b\u0131\u0142\u0144\u014d\u0151\u0153\u015b\u016b\u0171\u017a\u017c\u01ce\u01d0\u01d2\u01d4" +
        "\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451\u0454\u0456\u0457\u0491" +
        "!'\",(),-.:;? ",
        punctuations="!'(),-.:;? ",
        phonemes="",
        is_unique=True,
        is_sorted=True,
    )

    # Main config
    config = VitsConfig(
        audio=audio_config,
        model_args=model_args,
        characters=chars_config,
        max_audio_len=16000 * 10,
    )

    # Initialize model
    model = Vits.init_from_config(config)
    
    # Load the model weights
    model.load_state_dict(torch.load('best_model_7998.pth', map_location=torch.device('cpu'))['model'])
    model.eval()

    return model, config

def save_uploaded_file(uploaded_file):
    """Save uploaded file and return the path"""
    # Create a temporary directory if it doesn't exist
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a unique filename
    temp_path = os.path.join(temp_dir, f"temp_{uploaded_file.name}")
    
    # Write the file
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return temp_path

def generate_audio(model, config, text, speaker_path):
    """Generate audio from text using the model"""
    try:
        # Initialize tokenizer
        tokenizer = TTSTokenizer.init_from_config(config)[0]
        
        # Convert text to token ids
        token_ids = tokenizer.text_to_ids(text)
        token_ids = torch.tensor(token_ids).unsqueeze(0)

        # Compute speaker embedding using the file path
        encoder = model.speaker_manager
        d_vectors = encoder.compute_embedding_from_clip(speaker_path)
        d_vectors = torch.FloatTensor(d_vectors).unsqueeze(0)
        
        # Generate audio
        aux_input = {"d_vectors": d_vectors}
        output = model.inference(x=token_ids, aux_input=aux_input)
        
        return output["model_outputs"].squeeze(1)
    except Exception as e:
        st.error(f"Error in audio generation: {str(e)}")
        raise

def cleanup_temp_files(temp_path):
    """Clean up temporary files"""
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Remove temp directory if empty
        temp_dir = os.path.dirname(temp_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except Exception as e:
        st.warning(f"Warning: Could not clean up temporary files: {str(e)}")

# Main app
def main():
    st.title("Vietnamese Text-to-Speech Demo")

    # Load model
    with st.spinner("Loading model..."):
        model, config = load_model()

    # Text input
    text_input = st.text_area("Enter Vietnamese text:", value="cô gái đi tới chậu hoa và nhổ một vài bông hoa đẹp")

    # File uploader for speaker reference audio
    speaker_file = st.file_uploader("Upload speaker reference audio (WAV file)", type=['wav'])

    if st.button("Generate Speech") and speaker_file is not None:
        try:
            # Save uploaded file
            temp_path = save_uploaded_file(speaker_file)
            st.info(f"Processing audio file: {speaker_file.name}")

            # Generate audio
            with st.spinner("Generating speech..."):
                audio_output = generate_audio(model, config, text_input, temp_path)
                
                # Save the generated audio to a temporary file
                output_path = os.path.join("temp_audio", "output.wav")
                torchaudio.save(output_path, audio_output, sample_rate=16000)

                # Display audio player
                st.audio(output_path, format='audio/wav')

            # Cleanup
            cleanup_temp_files(temp_path)
            cleanup_temp_files(output_path)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please make sure the uploaded audio file is a valid WAV file with speech content.")
            
            # Ensure cleanup happens even if there's an error
            if 'temp_path' in locals():
                cleanup_temp_files(temp_path)
            if 'output_path' in locals():
                cleanup_temp_files(output_path)

if __name__ == "__main__":
    main()