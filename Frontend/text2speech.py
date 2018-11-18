try:
	from google.cloud import texttospeech
except:
	texttospeech = None

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'codejam-2018-43e63ace5e4e.json'

def getSpeech(myText):
	myClient = texttospeech.TextToSpeechClient()
	textInput = texttospeech.types.SynthesisInput(text=myText)
	voiceConfig = texttospeech.types.VoiceSelectionParams(
		language_code='en-US',
		ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE)
	audioConfig = texttospeech.types.AudioConfig(
		audio_encoding=texttospeech.enums.AudioEncoding.MP3)
	response = myClient.synthesize_speech(textInput, voiceConfig, audioConfig)
	return response


def main():
	myText = 'Hello world!'
	response = getSpeech(myText)

	with open('ourput.mp3', 'wb') as outFile:
		outFile.write(response.audio_content)


if __name__ == '__main__':
	main()