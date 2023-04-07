# PodFast

This script is a rough proof of concept bringing the openai-whisper project, 
HuggingFace Transformers, and NLTK together to transcribe and summarize spoken
audio.  I wrote this to summarize my favorite podcasts.

As you can see there is no advanced useage or tuning of these tools.  Which is
what I find so impressive about the current ML ecosystem.  I didnt have to
know anything special to accomplish something useful and there's no advanced
usage of python going on.

## Table of Contents

- [Installation](#installation)
- [ToDo](#ToDo)

## Installation

I suggest using Python 3.10
```sh
# work from a virtualenv
virtualenv ~/.envs/podfast
source ~/.envs/podfast/bin/activate

# install requirements
pip install -r requirements.txt

# Run the script (First time will take time and bandwidth to download models)
python poc.py

# Enter the direct URL to the mp3 you'd like to summarize.  Example:
https://chtbl.com/track/5899E/podtrac.com/pts/redirect.mp3/traffic.omny.fm/d/clips/e73c998e-6e60-432f-8610-ae210140c5b1/f5d5fac6-77be-47e6-9aee-ae32006cd8c3/98b09fa6-21e6-4a6f-9674-afaa016d3419/audio.mp3?utm_source=Podcast&in_playlist=b26cbbeb-86eb-4b97-9b34-ae32006cd8d6

```

## ToDo

- Cleanup tokenization whitespace and empty sentences
- Catch exceptions
- Modularize
- Add queuing for transcription and summarization jobs
- Add web interface (pywebio)
- Filter sentences containing words that distract from the subject (such as 
'iheart', 'podcast', etc) 
- Containerize
- Run on more formidable hardware (my poor old lappy takes 40 minutes for a 50
minute podcast)
- Tuning
- Add scheduling capability
- Store results in datasource, make it browsable, and retrieve stored copy if
job has already been done before.

## Shortcomings
- I've only spent a few hours on it
- No tuning has been done
- This readme is longer than the code
