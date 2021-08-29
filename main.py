import numpy as np
from network import Network
import layers
from jasper_block import Jasper_block
from ruamel.yaml import YAML
import argparse
import datetime
import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import tkinter.scrolledtext as scrolledtext

def process():
    np.show_config()
    start_time = datetime.datetime.now()

    #Read files
    preprocessed = np.load(preprocessed_entry_text.get())
    encoderparams = np.load(encoder_params_entry_text.get(),allow_pickle='TRUE').item()
    decoderparams = np.load(decoder_params_entry_text.get(),allow_pickle='TRUE').item()
    yaml = YAML(typ="safe")
    with open(model_descriptor_entry_text.get()) as f:
        jasper_params = yaml.load(f)

    jasper_encoder = jasper_params['JasperEncoder']
    jasper = jasper_encoder['jasper']

    #Build network
    net = Network()
    for block in range(len(jasper)):
        unit = jasper[block]
        if block == len(jasper)-1:
            net.add(Jasper_block(encoderparams=[ v for k,v in encoderparams.items() if k.startswith('encoder.'+str(block)+'.')], residual=unit['residual'], repeat=unit['repeat'], stride=unit['stride'], dilation=unit['dilation'], tcs=False))
        else:
            net.add(Jasper_block(encoderparams=[ v for k,v in encoderparams.items() if k.startswith('encoder.'+str(block)+'.')], residual=unit['residual'], repeat=unit['repeat'], stride=unit['stride'], dilation=unit['dilation']))

    out = net.predict(np.round(preprocessed, 4))
    out = layers.ConvLayer(decoderparams['decoder_layers.0.weight'], [1], [1], False, True, decoderparams['decoder_layers.0.bias']).forward_propagation(out)
    out = layers.SoftmaxLayer().forward_propagation(out)
    max_vals = np.argmax(out, axis=1)
    hypothesis = layers.ctc_decoder(max_vals, jasper_params['labels'])
    ellapsed_time_label_text.set(str(datetime.datetime.now() - start_time)[:-4])
    set_result_textbox(hypothesis)

def open_preprocessed_audio():
    rep = filedialog.askopenfilenames(
    	parent=root,
    	initialdir='/',
    	initialfile='tmp',
    	filetypes=[
    		("NPY", "*.npy"),
    		("All files", "*")])
    try:
        preprocessed_entry_text.set(rep[1])
    except IndexError:
        print("No file selected")

def open_model_descriptor():
    rep = filedialog.askopenfilenames(
        parent=root,
        initialdir='/',
        initialfile='tmp',
        filetypes=[
            ("YAML", "*.yaml"),
            ("All files", "*")])
    try:
        model_descriptor_entry_text.set(rep[1])
    except IndexError:
        print("No file selected")

def open_encoder_params():
    rep = filedialog.askopenfilenames(
        parent=root,
        initialdir='/',
        initialfile='tmp',
        filetypes=[
            ("NPY", "*.npy"),
            ("All files", "*")])
    try:
        encoder_params_entry_text.set(rep[1])
    except IndexError:
        print("No file selected")

def open_decoder_params():
    rep = filedialog.askopenfilenames(
        parent=root,
        initialdir='/',
        initialfile='tmp',
        filetypes=[
            ("NPY", "*.npy"),
            ("All files", "*")])
    try:
        decoder_params_entry_text.set(rep[1])
    except IndexError:
        print("No file selected")

def set_result_textbox(result):
    result_text_box.insert(tk.INSERT, result)

root = tk.Tk()
root.title("QuartzNet ASR - CPU engine")

#Preprocessed file selection
preprocessed_label = tk.Label(text="Preprcessed audio file", anchor="e").grid(row=0, column=0, padx=4, pady=(15, 4), sticky='ew')
preprocessed_entry_text = tk.StringVar()
preprocessed_entry = tk.Entry(root, textvariable=preprocessed_entry_text).grid(row=0, column=1, padx=4, pady=(15, 4), sticky='ew') 
ttk.Button(root, text="Browse", command=open_preprocessed_audio).grid(row=0, column=2, padx=4, pady=(15, 4), sticky='ew')

#Model config file selection
model_descriptor_label = tk.Label(text="Model descriptor file", anchor="e").grid(row=1, column=0, padx=4, pady=4, sticky='ew')
model_descriptor_entry_text = tk.StringVar()
model_descriptor_entry = tk.Entry(root, textvariable=model_descriptor_entry_text).grid(row=1, column=1, padx=4, pady=4, sticky='ew') 
ttk.Button(root, text="Browse", command=open_model_descriptor).grid(row=1, column=2, padx=4, pady=4, sticky='ew')

#Encoder params file selection
encoder_params_label = tk.Label(text="Encoder params file", anchor="e").grid(row=2, column=0, padx=4, pady=4, sticky='ew')
encoder_params_entry_text = tk.StringVar()
encoder_params_entry = tk.Entry(root, textvariable=encoder_params_entry_text).grid(row=2, column=1, padx=4, pady=4, sticky='ew') 
ttk.Button(root, text="Browse", command=open_encoder_params).grid(row=2, column=2, padx=4, pady=4, sticky='ew')

#Decoder params file selection
decoder_params_label = tk.Label(text="Decoder params file", anchor="e").grid(row=3, column=0, padx=4, pady=4, sticky='ew')
decoder_params_entry_text = tk.StringVar()
decoder_params_entry = tk.Entry(root, textvariable=decoder_params_entry_text).grid(row=3, column=1, padx=4, pady=4, sticky='ew') 
ttk.Button(root, text="Browse", command=open_decoder_params).grid(row=3, column=2, padx=4, pady=4, sticky='ew')

#Result text box
result_text_box = scrolledtext.ScrolledText(root, wrap = tk.WORD)
result_text_box.grid(row=0, column=3, padx=20, pady=(15,0), sticky='news', rowspan=5, columnspan=3)

#Footer
ellapsed_time_label = tk.Label(text="Process time:", anchor="w").grid(row=5, column=3, padx=(20,0), pady=15)
ellapsed_time_label_text = tk.StringVar()
ellapsed_time_result_label = tk.Label(textvariable=ellapsed_time_label_text, anchor="w").grid(row=5, column=4, padx=4, pady=4)
ttk.Button(root, text="Process", command=process).grid(row=5, column=5, padx=20, pady=15)

#Grid settings
root.grid_columnconfigure(5, weight=1)
root.grid_rowconfigure(0, weight=0)
root.grid_rowconfigure(1, weight=0)
root.grid_rowconfigure(2, weight=0)
root.grid_rowconfigure(3, weight=0)
root.grid_rowconfigure(4, weight=1)

root.mainloop()