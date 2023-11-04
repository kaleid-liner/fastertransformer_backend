from transformers import LlamaTokenizer, AutoTokenizer, AutoModel
import gradio as gr
import mdtex2html
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
from queue import Queue, Empty
import argparse
from functools import partial
from tritonclient.grpc.service_pb2 import ModelInferResponse
from tritonclient.utils import InferenceServerException
import re


parser = argparse.ArgumentParser()
parser.add_argument('-u', '--url', type=str, required=False, default='127.0.0.1:8001',
                    help='Inference server Appkey. Default is .')
parser.add_argument('-pro', '--protocol', type=str, required=False, default='grpc',
                    help='Inference server Appkey. Default is .')
parser.add_argument('--model_name', type=str, default="fastertransformer",
                    help='model_name')
parser.add_argument('--server_name', type=str, default="10.190.175.139",
                    help='Service hostname')
parser.add_argument('--server_port', type=int, default=80,
                    help='Service port')
parser.add_argument('--project_name', type=str, default="FasterTransformer")
FLAGS = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", cache_dir="/data/checkpoint_hub")
tokenizer.pad_token_id = 1
tokenizer.padding_side = "left"

verbose = 0


def lines2inputs(FLAGS, lines, tokenizer, max_length, top_p, temperature):
    encoded_inputs = tokenizer.batch_encode_plus(lines)
    input_ids = np.array(tokenizer.pad(encoded_inputs, pad_to_multiple_of=16)["input_ids"], dtype=np.uint32)
    input_lengths = np.array([len(ids) for ids in input_ids], dtype=np.uint32).reshape(-1, 1)
    request_output_len = max_length * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
    runtime_top_p = top_p * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    temperature = temperature * np.ones([input_ids.shape[0], 1]).astype(np.float32)

    def to_input(name, np_input):
        client_util = httpclient if FLAGS.protocol == "http" else grpcclient
        t = client_util.InferInput(
            name, np_input.shape, np_to_triton_dtype(np_input.dtype))
        t.set_data_from_numpy(np_input)
        return t        

    inputs = [to_input("input_ids", input_ids),
              to_input("input_lengths", input_lengths),
              to_input("request_output_len", request_output_len),
              to_input("runtime_top_p", runtime_top_p),
              to_input("temperature", temperature)]
    return inputs, input_lengths


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        line = re.sub(r"Answer:\ *", "", line)
        if line in "Answer:  ":
            line = ""
        lines[i] = line
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def stream_callback(queue, result, error):
    if error:
        queue.put(error)
    else:
        queue.put(result)


def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                 concurrency=concurrency,
                                                 verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                 verbose=verbose)


def predict(input, chatbot, max_length, top_p, temperature):
    chatbot.append((parse_text(input), ""))

    for token_text in model.stream_chat(input, max_length, top_p, temperature):
        chatbot[-1] = (parse_text(input), parse_text(token_text))
        yield chatbot


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return []


def set_user_input(user_input):
    return gr.update(value=user_input)


class TritongRPCModel:
    def stream_chat(self, input, max_length, top_p, temperature):
        with create_inference_server_client(FLAGS.protocol,
                                            FLAGS.url,
                                            concurrency=1,
                                            verbose=verbose) as client:
            result_queue = Queue()
            client.start_stream(callback=partial(stream_callback, result_queue))
            inputs, input_lengths = lines2inputs(FLAGS, [input], tokenizer, max_length, top_p, temperature)
            client.async_stream_infer(FLAGS.model_name, inputs)

            while True:
                try:
                    result = result_queue.get(timeout=2)
                except Empty:
                    break

                if type(result) == InferenceServerException:
                    raise result

                seq_len = result.as_numpy("sequence_length")[0, 0]
                output_ids = result.as_numpy("output_ids")[0][0][input_lengths[0][0]+1:seq_len]
                token_text = tokenizer.decode(output_ids)
                yield token_text


model = TritongRPCModel()

choices = [
    "Hey, are you consciours? Can you talk to me?",
    "Tell me the story of the Titanic",
]

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">{}</h1>""".format(FLAGS.project_name))

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10)
                user_input_choices = gr.Dropdown(choices=choices, label="Or choose one")
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.6, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.9, step=0.01, label="Temperature", interactive=True)

    user_input_choices.input(set_user_input, [user_input_choices], [user_input])
    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature], [chatbot],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot], show_progress=True)

demo.queue().launch(share=True, inbrowser=True, server_port=FLAGS.server_port, server_name=FLAGS.server_name)
