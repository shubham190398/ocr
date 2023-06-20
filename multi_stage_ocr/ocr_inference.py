from text_extraction_model import inference_model
from text_extraction_utils import create_vocab


vocab = create_vocab("C:\\Users\\Kare4U\\Downloads\\augmented_FUNSD\\augmented_FUNSD_texts")

model = inference_model(input_dim=(32, 128, 1), output_dim=len(vocab))
model.load_weights("models/text_model.hdf5")

