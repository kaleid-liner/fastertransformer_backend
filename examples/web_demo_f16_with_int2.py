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


left_protocol = "grpc"
left_url = '127.0.0.1:8004'


right_protocol = "grpc"
right_url = '127.0.0.1:8007'

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--url', type=str, required=False, default='127.0.0.1:8001',
                    help='Inference server Appkey. Default is .')
parser.add_argument('-pro', '--protocol', type=str, required=False, default='grpc',
                    help='Inference server Appkey. Default is .')
parser.add_argument('--model_name', type=str, default="fastertransformer",
                    help='model_name')
parser.add_argument('--server_name', type=str, default="0.0.0.0",
                    help='Service hostname')
parser.add_argument('--server_port', type=int, default=80,
                    help='Service port')
parser.add_argument('--project_name', type=str, default="FasterTransformer")
FLAGS = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("/dataHDD/ckpts/hf-llama-2-13b/int2-g128", cache_dir="/dataHDD/checkpoint_hub")
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

verbose = 0

prompt = "[INST] «SYS»\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.\n«/SYS»\n\n{} [/INST]"
prompt2 = "[INST] «SYS»\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.\n«/SYS»\n\n{} [/INST]"
prompt2 = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{}\n\n### Response: Let's think step by step."
)


def lines2inputs(protocol, lines, tokenizer, max_length, top_p, temperature, repetition_penalty):
    encoded_inputs = tokenizer.batch_encode_plus(lines, truncation=True, max_length=256)
    input_ids = np.array(tokenizer.pad(encoded_inputs, pad_to_multiple_of=64)["input_ids"], dtype=np.uint32)
    input_lengths = np.array([len(ids) for ids in input_ids], dtype=np.uint32).reshape(-1, 1)
    request_output_len = max_length * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
    runtime_top_p = top_p * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    temperature = temperature * np.ones([input_ids.shape[0], 1]).astype(np.float32)
    repetition_penalty = repetition_penalty * np.ones([input_ids.shape[0], 1]).astype(np.float32)

    def to_input(name, np_input):
        client_util = httpclient if protocol == "http" else grpcclient
        t = client_util.InferInput(
            name, np_input.shape, np_to_triton_dtype(np_input.dtype))
        t.set_data_from_numpy(np_input)
        return t        

    inputs = [to_input("input_ids", input_ids),
              to_input("input_lengths", input_lengths),
              to_input("request_output_len", request_output_len),
              to_input("runtime_top_p", runtime_top_p),
              to_input("temperature", temperature),
              to_input("repetition_penalty", repetition_penalty)]
    return inputs, input_lengths


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
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


def predict_left(input, chatbot, max_length, top_p, temperature, repetition_penalty):
    chatbot.append((parse_text(input), ""))

    for token_text in model_left.stream_chat(input, max_length, top_p, temperature, repetition_penalty):
        chatbot[-1] = (parse_text(input), parse_text(token_text))
        yield chatbot

def predict_right(input, chatbot, max_length, top_p, temperature, repetition_penalty):
    chatbot.append((parse_text(input), ""))

    for token_text in model_right.stream_chat(input, max_length, top_p, temperature, repetition_penalty):
        chatbot[-1] = (parse_text(input), parse_text(token_text))
        yield chatbot

def reset_user_input():
    return gr.update(value='')


def reset_state():
    return []


def set_user_input(user_input):
    return gr.update(value=user_input)


class TritongRPCModel:
    def __init__(self, protocol, url) -> None:
        self.protocol = protocol
        self.url = url

    def stream_chat(self, input, max_length, top_p, temperature, repetition_penalty):
        with create_inference_server_client(self.protocol,
                                            self.url,
                                            concurrency=1,
                                            verbose=verbose) as client:
            result_queue = Queue()
            client.start_stream(callback=partial(stream_callback, result_queue))
            input = prompt2.format(input)
            inputs, input_lengths = lines2inputs(self.protocol, [input], tokenizer, max_length, top_p, temperature, repetition_penalty)
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


model_left = TritongRPCModel(
    protocol=left_protocol,
    url=left_url
)

model_right = TritongRPCModel(
    protocol=right_protocol,
    url=right_url
)

choices = [
    "How to put an elephant into a fridge?",
]

with gr.Blocks() as demo:

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""<h1 align="center">{}</h1>""".format("LLAMA2-13B-16Bit"))
            chatbot_left = gr.Chatbot()
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Column(scale=12):
                        user_input_left = gr.Textbox(show_label=False, placeholder="Input...", lines=10)
                        examples_left = gr.Examples(examples=choices, inputs=user_input_left, label="Or choose one")
                    with gr.Column(min_width=32, scale=1):
                        submitBtn_left = gr.Button("Submit", variant="primary")
                with gr.Column(scale=1):
                    emptyBtn_left = gr.Button("Clear History")
                    max_length_left = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                    top_p_left = gr.Slider(0, 1, value=0.6, step=0.01, label="Top P", interactive=True)
                    temperature_left = gr.Slider(0, 1, value=0.9, step=0.01, label="Temperature", interactive=True)
                    repetition_penalty_left = gr.Slider(0, 2, value=1.0, step=0.1, label="Repetition penalty", interactive=False)
        with gr.Column(scale=1):
            gr.HTML("""<h1 align="center">{}</h1>""".format("LLAMA2-13B-2Bit"))
            chatbot_right = gr.Chatbot()
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Column(scale=12):
                        user_input_right = gr.Textbox(show_label=False, placeholder="Input...", lines=10)
                        examples_left = gr.Examples(examples=choices, inputs=user_input_right, label="Or choose one")
                    with gr.Column(min_width=32, scale=1):
                        submitBtn_right = gr.Button("Submit", variant="primary")
                with gr.Column(scale=1):
                    emptyBtn_right = gr.Button("Clear History")
                    max_length_right = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                    top_p_right = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
                    temperature_right = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
                    repetition_penalty_right = gr.Slider(0, 2, value=1.1, step=0.1, label="Repetition penalty", interactive=False)
    
    submitBtn_left.click(predict_left, [user_input_left, chatbot_left, max_length_left, top_p_left, temperature_left, repetition_penalty_left], [chatbot_left], show_progress=True) 
    submitBtn_left.click(reset_user_input, [], [user_input_left])
    emptyBtn_left.click(reset_state, outputs=[chatbot_left], show_progress=True)
    
    submitBtn_right.click(predict_right, [user_input_right, chatbot_right, max_length_right, top_p_right, temperature_right, repetition_penalty_right], [chatbot_right], show_progress=True) 
    submitBtn_right.click(reset_user_input, [], [user_input_right])
    emptyBtn_right.click(reset_state, outputs=[chatbot_right], show_progress=True)

demo.queue().launch(share=True, inbrowser=True, server_port=FLAGS.server_port, server_name=FLAGS.server_name)
