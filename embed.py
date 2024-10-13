import os

from bin.compute_embeddings import compute_embeddings
from configs.shared_configs import BaseDatasetConfig


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

# VIVOS_DOWNLOAD_PATH = os.path.join(CURRENT_PATH, "data", "vivos")
VIVOS_DOWNLOAD_PATH = '/kaggle/input/vina-vivos/vivos'

vivos_config = BaseDatasetConfig(
    formatter="vivos",
    dataset_name="vivos",
    meta_file_train="",
    meta_file_val="",
    path=VIVOS_DOWNLOAD_PATH,
    language="vi",
      # Ignore the test speakers to full replicate the paper experiment
)

SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

embeddings_file = os.path.join(CURRENT_PATH, "data", "speakers.pth")
if not os.path.isfile(embeddings_file):
    compute_embeddings(
                SPEAKER_ENCODER_CHECKPOINT_PATH,
                SPEAKER_ENCODER_CONFIG_PATH,
                embeddings_file,
                old_speakers_file=None,
                config_dataset_path=None,
                formatter_name=vivos_config.formatter,
                dataset_name=vivos_config.dataset_name,
                dataset_path=vivos_config.path,
                meta_file_train=vivos_config.meta_file_train,
                meta_file_val=vivos_config.meta_file_val,
                disable_cuda=False,
                no_eval=False,
            )
    

