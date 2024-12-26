import gradio as gr



with gr.Blocks() as blocks:
    pad = gr.ImageEditor()

# blocks.launch(server_name="0.0.0.0", server_port=1234)

blocks.launch(share=True)

