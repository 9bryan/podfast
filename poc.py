from transformers import pipeline
from nltk import tokenize
import nltk
import whisper
import requests
import subprocess

#get url of mp3
podcast_url = input('Input the URL of the podcast or dialog that you wish to summarize:\n')
temp_file = subprocess.check_output("mktemp", shell=True).decode('utf-8')
request = requests.get(podcast_url, allow_redirects=True)
open(temp_file, 'wb').write(request.content)
print(temp_file)


#for tokenization
nltk.download('punkt')

whisper_model = whisper.load_model("small")
transcribed_text = whisper_model.transcribe(temp_file)

#Break transcription into a python list of sentences
tokenized_text = tokenize.sent_tokenize(transcribed_text["text"])

batch_size = 30
sentence_index = chunk_index = index = 0
chunk_size = 30
chunked_text = ['']

# chunked_text is a list full 30 sentance strings.  
# Pipeline has a limit on how much it can summarize at a time
for sentence in tokenized_text:
    chunked_text[chunk_index] = chunked_text[chunk_index] + " " + sentence
    index += 1
    if index > chunk_size:
        chunked_text.append('')
        chunk_index += 1
        index = 0

# Using  pipeline API for summarization task
#summarization = pipeline("summarization", model = "facebook/bart-large-xsum")
summarization = pipeline("summarization")
for chunk in chunked_text:
    #print(chunk + "\n")
    print(summarization(chunk)[0]['summary_text'])

