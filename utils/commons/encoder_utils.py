def get_speech_encoder(speech_encoder, device=None, **kargs):
    if speech_encoder == "vec768l12":
        from vencoder.ContentVec768L12 import ContentVec768L12

        speech_encoder_object = ContentVec768L12(device=device)
    elif speech_encoder == "vec256l9":
        from vencoder.ContentVec256L9 import ContentVec256L9

        speech_encoder_object = ContentVec256L9(device=device)
    elif speech_encoder == "hubertsoft":
        from vencoder.HubertSoft import HubertSoft

        speech_encoder_object = HubertSoft(device=device)
    elif speech_encoder == "whisper-ppg":
        from vencoder.WhisperPPG import WhisperPPG

        speech_encoder_object = WhisperPPG(device=device)
    elif speech_encoder == "whisper-ppg-large":
        from vencoder.WhisperPPGLarge import WhisperPPGLarge

        speech_encoder_object = WhisperPPGLarge(device=device)
    elif speech_encoder == "wavlmbase+":
        from vencoder.WavLMBasePlus import WavLMBasePlus

        speech_encoder_object = WavLMBasePlus(device=device)
    elif speech_encoder == "HuBERT-Large":
        from vencoder.HubertLarge import HuBERTLarge

        speech_encoder_object = HuBERTLarge(device=device)
    elif speech_encoder == "hubertbase":
        from vencoder.hubertbase import HuBERTBase

        speech_encoder_object = HuBERTBase(device=device)
    else:
        raise Exception("Unknown speech encoder")
    return speech_encoder_object
