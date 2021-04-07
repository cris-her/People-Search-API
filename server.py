import json
import os
import tempfile
import onnxruntime
import torch
import librosa
import IPython.display as ipd
import nemo.collections.asr as nemo_asr
from nemo.collections import nlp as nemo_nlp
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.metrics.wer import WER
import onnxruntime as rt
import numpy as np
import joblib
import numpy as np

from flask import Flask, redirect, url_for, render_template
from flask import jsonify
from pydub import AudioSegment as am
  

app = Flask(__name__, template_folder='templates')
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def setup_transcribe_dataloader(cfg, vocabulary):
    config = {
        'manifest_filepath': os.path.join(cfg['temp_dir'], 'manifest.json'),
        'sample_rate': 16000,
        'labels': vocabulary,
        'batch_size': min(cfg['batch_size'], len(cfg['paths2audio_files'])),
        'trim_silence': True,
        'shuffle': False,
    }
    dataset = AudioToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        sample_rate=config['sample_rate'],
        int_values=config.get('int_values', False),
        augmentor=None,
        max_duration=config.get('max_duration', None),
        min_duration=config.get('min_duration', None),
        max_utts=config.get('max_utts', 0),
        blank_index=config.get('blank_index', -1),
        unk_index=config.get('unk_index', -1),
        normalize=config.get('normalize_transcripts', False),
        trim=config.get('trim_silence', True),
        load_audio=config.get('load_audio', True),
        parser=config.get('parser', 'en'),
        add_misc=config.get('add_misc', False),
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        collate_fn=dataset.collate_fn,
        drop_last=config.get('drop_last', False),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
    )
#POSTMAN PARA PRUEBAS

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html", content=2)

@app.route('/<name>')
def user(name):
    return f"Hello {name}!"

@app.route('/admin/')
def admin():
    return redirect(url_for("user", name="ADMIN!"))

@app.route('/predict', methods=['GET'])
def predict():
    sound = am.from_file('static/onetwo.wav', format='wav', frame_rate=48000)
    sound = sound.set_frame_rate(16000)
    sound.export('example_16bit_16k_1ch.wav', format='wav')
    files = ['example_16bit_16k_1ch.wav']
    for fname, transcription in zip(files, quartznet.transcribe(paths2audio_files=files)):
        print(f"Audio in {fname} was recognized as: {transcription}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w') as fp:
            for audio_file in files:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': 'nothing'}
                fp.write(json.dumps(entry) + '\n')

        config = {'paths2audio_files': files, 'batch_size': 4, 'temp_dir': tmpdir}
        temporary_datalayer = setup_transcribe_dataloader(config, quartznet.decoder.vocabulary)
        for test_batch in temporary_datalayer:
            processed_signal, processed_signal_len = quartznet.preprocessor(
                input_signal=test_batch[0].to(quartznet.device), length=test_batch[1].to(quartznet.device)
            )
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(processed_signal),}
            ologits = ort_session.run(None, ort_inputs)
            alogits = np.asarray(ologits)
            logits = torch.from_numpy(alogits[0])
            greedy_predictions = logits.argmax(dim=-1, keepdim=False)
            wer = WER(vocabulary=quartznet.decoder.vocabulary, batch_dim_index=0, use_cer=False, ctc_decode=True)
            hypotheses = wer.ctc_decoder_predictions_tensor(greedy_predictions)
            print(hypotheses)
            break

    return render_template("new.html", hypotheses=hypotheses)


if __name__ == "__main__":
    quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")
    ort_session = onnxruntime.InferenceSession('models/asr.onnx')
    app.run(host="0.0.0.0", port=6006, debug=True)