try:
	from google.cloud import texttospeech
except:
	texttospeech = None

captionDict = {}

def getSpeech(myText):

	global captionDict

	if myText not in captionDict:
		myClient = texttospeech.TextToSpeechClient()
		textInput = texttospeech.types.SynthesisInput(text=myText)
		voiceConfig = texttospeech.types.VoiceSelectionParams(
			language_code='en-US',
			ssml_gender=texttospeech.enums.SsmlVoiceGender.FEMALE)
		audioConfig = texttospeech.types.AudioConfig(
			audio_encoding=texttospeech.enums.AudioEncoding.MP3)
		captionDict[myText] = myClient.synthesize_speech(textInput, voiceConfig, audioConfig)

	return captionDict[myText]


def getMp3(response):
	return response.audio_content


def saveMp3(response, filename):
	with open(filename, 'wb') as mp3File:
		mp3File.write(getMp3(response))
	return filename


def main():
	myText = 'Hello world!'
	response = getSpeech(myText)
	saveMp3(response)


if __name__ == '__main__':
	main()