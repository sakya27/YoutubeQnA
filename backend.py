# imports
import os
import glob
import json
from llm_utils import llm_summary, llm_answer
from yt_utils import get_subtitles






# function to generate summary of a video

def get_summary(video_link):

    global subtitles
    status, subtitles = get_subtitles(video_link)
    if status == 'success':
        status, data = llm_summary(subtitles)

    return status, data
    

# function to get an answer

def get_answer(question):
    print(subtitles)
    status, data = llm_answer(question, subtitles)
    return status, data