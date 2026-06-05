# Config

Here are the config files used to train the single/multi-speaker TTS models.

4 different configurations are given:

- LJSpeech: suggested configuration for LJSpeech dataset.
- LibriTTS: suggested configuration for LibriTTS dataset.
- AISHELL3: suggested configuration for AISHELL-3 dataset.
- LJSpeech_paper: close to the setting proposed in the original FastSpeech 2 paper.

Some important hyper-parameters are explained here.

## preprocess.yaml

- **path.lexicon_path**: the lexicon used by Montreal Forced Aligner.
- **mel.stft.mel_fmax**: set it to 8000 if HiFi-GAN vocoder is used, and set it to null if MelGAN is used.
- The preprocessor saves mel spectrograms and explicit phoneme durations from TextGrid alignments.

## train.yaml

- **optimizer.grad_acc_step**: the number of batches of gradient accumulation before updating the model parameters and calling optimizer.zero_grad().
- **optimizer.anneal_steps & optimizer.anneal_rate**: the learning rate is reduced at the anneal steps by the configured rate.
- **data.group_size**: the number of same-length sorted mini-batches grouped by the DataLoader before splitting back into training batches. Lower this if CPU memory is tight.
- **logging.log_audio**: whether to write reconstructed and synthesized audio to TensorBoard.
- **logging.log_figure**: whether to write mel figures to TensorBoard.
- **stylecode_logging**: controls stylecode JSONL export at checkpoint save time. The default logs 128 fixed validation utterances with token IDs and phoneme text for t-SNE analysis.

## model.yaml

- **transformer.decoder_layer**: the original paper used a 4-layer decoder, but a 6-layer decoder can be better for multi-speaker TTS.
- **variance_predictor**: duration predictor convolution size and dropout.
- **phoneme_style**: frame-level mel style encoder and bottleneck settings. The default stylecode bottleneck is 32 dimensions, with strict duration/mel alignment checks enabled.
- **loss.duration_weight**: scalar weight for duration prediction loss.
- **multi_speaker**: whether to apply a speaker embedding table.
- **vocoder.speaker**: should be set to `universal` if any dataset other than LJSpeech is used.
