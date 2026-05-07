import gradio as gr
from shakkala import Shakkala
from pathlib import Path
import torch
from eo_pl import TashkeelModel as TashkeelModelEO
from ed_pl import TashkeelModel as TashkeelModelED
from tashkeel_tokenizer import TashkeelTokenizer
from utils import remove_non_arabic

# Initialize the Shakkala model
sh = Shakkala(version=3)
model, graph = sh.get_model()

def infer_shakkala(input_text):
    input_int = sh.prepare_input(input_text)
    logits = model.predict(input_int)[0]
    predicted_harakat = sh.logits_to_text(logits)
    final_output = sh.get_final_text(input_text, predicted_harakat)
    print(final_output)
    return final_output

# Initialize the CaTT model and tokenizer
tokenizer = TashkeelTokenizer()
eo_ckpt_path = Path(__file__).parent / 'models/best_eo_mlm_ns_epoch_193.pt'
ed_ckpt_path = Path(__file__).parent / 'models/best_ed_mlm_ns_epoch_178.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_seq_len = 1024
print('Creating Model...')
eo_model = TashkeelModelEO(tokenizer, max_seq_len=max_seq_len, n_layers=6, learnable_pos_emb=False)
ed_model = TashkeelModelED(tokenizer, max_seq_len=max_seq_len, n_layers=3, learnable_pos_emb=False)

eo_model.load_state_dict(torch.load("best_model.pt", map_location=device))
eo_model.eval().to(device)
ed_model.load_state_dict(torch.load(ed_ckpt_path, map_location=device))
ed_model.eval().to(device)


def infer_catt(input_text, choose_model):
    input_text = remove_non_arabic(input_text)
    batch_size = 16
    verbose = True
    if choose_model == 'Encoder-Only':
        output_text = eo_model.do_tashkeel_batch([input_text], batch_size, verbose)
    else:
        output_text = ed_model.do_tashkeel_batch([input_text], batch_size, verbose)

    return output_text[0]



examples = ["السلام عليكم ورحمة الله وبركاته", "العلم نور", "الحمد لله"]

with gr.Blocks(title="Arabic Tashkeel") as demo:
    gr.HTML("<center><h1>Arabic Tashkeel</h1></center>")
    gr.HTML(
        "<center><p>Compare different methods for adding tashkeel to Arabic text.</p></center>"
    )
    with gr.Tab(label="CATT"):
        gr.HTML("<center><h2>CATT: Character-based Arabic Tashkeel Transformer</h2></center>")
        gr.HTML("<center><a href='https://github.com/Panda2123/CATT-Character-based-Arabic-Tashkeel-Transformer'>GitHub</a></center>")

        with gr.Row():
            with gr.Column():
                text_input1 = gr.Textbox(label="Input Text", rtl=True, text_align="right")
                choose_model = gr.Radio(
                    label="Choose Model",
                    value="Encoder-Decoder",
                    choices=["Encoder-Only", "Encoder-Decoder"],
                )
                with gr.Row():
                    clear_button1 = gr.Button(value="Clear", variant="secondary")
                    submit_button1 = gr.Button(value="Add Tashkeel", variant="primary")

            with gr.Column():
                text_output1 = gr.Textbox(label="Output Text", rtl=True, text_align="right")

        gr.Examples(examples, text_input1, cache_examples=False)

        submit_button1.click(infer_catt, inputs=[text_input1, choose_model], outputs=text_output1)
        clear_button1.click(lambda: ("", ""), outputs=[text_input1, text_output1])

    with gr.Tab(label="Shakkala"):
        gr.HTML("<center><h2>Shakkala: Arabic Diacritization</h2></center>")
        gr.HTML("<center><a href='https://github.com/Barqawiz/Shakkala'>GitHub</a> - <a href='https://pypi.org/project/shakkala/'>PyPi Package</a></center>")
        with gr.Row():
            with gr.Column():
                text_input2 = gr.Textbox(label="Input Text", rtl=True, text_align="right")
                with gr.Row():
                    clear_button2 = gr.Button(value="Clear", variant="secondary")
                    submit_button2 = gr.Button(value="Apply Tashkeel", variant="primary")
            with gr.Column():
                text_output2 = gr.Textbox(
                    lines=1, label="Output Text", rtl=True, text_align="right"
                )

        submit_button2.click(infer_shakkala, inputs=text_input2, outputs=text_output2)
        clear_button2.click(lambda: ("", ""), outputs=[text_input2, text_output2])
        
        gr.Examples(examples, text_input2, cache_examples=False)

if __name__ == '__main__':
    demo.queue().launch(share=True)
