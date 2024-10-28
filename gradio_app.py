import gradio as gr

from model.inference import gradio_chat

with gr.Blocks() as demo:
    input_text = gr.Textbox(placeholder="Type something to start!")
    output_text = gr.Textbox()

    input_text.submit(gradio_chat, [input_text], [output_text])

demo.launch()