"""Gets Politics Speeches from YouTube videos

This script allows you to retrieve the speeches of each politician
from a file in which all the youtube links are given for which to retrieve
the link for.

#Library
https://github.com/codenamewei/youtube2text

"""
from youtube_transcript_api import YouTubeTranscriptApi
import substring
import csv
# Google API library
import os
import io

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaIoBaseDownload

# YouTube API parameters
scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
# Disable OAuthlib's HTTPS verification when running locally.
# *DO NOT* leave this option enabled in production.
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"


api_service_name = "youtube"
api_version = "v3"

#add youtube client secret file
client_secrets_file = ""

# Get credentials and create an API client
flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
credentials = flow.run_console()
youtube = googleapiclient.discovery.build(api_service_name, api_version, credentials=credentials)

def get_video_informations(video_id: str) -> dict:
    """
    Gets the information about the video, such as date, likes, etc., given the video id
    :param video_id: id of the video
    :return: a dict with all the needed information
    """

    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=video_id
    )
    response = request.execute()
    # create the return dict from the response
    ret = {"id": response["items"][0]["id"],
           "publishedAt": None if "publishedAt" not in response["items"][0]["snippet"] else response["items"][0]["snippet"]["publishedAt"],
           "title": "none" if "title" not in response["items"][0]["snippet"] else response["items"][0]["snippet"]["title"],
           "channelTitle": "none" if "channelTitle" not in response["items"][0]["snippet"] else response["items"][0]["snippet"]["channelTitle"],
           "tags": [] if "tags" not in response["items"][0]["snippet"] else response["items"][0]["snippet"]["tags"],
           "viewCount": 0 if "viewCount" not in response["items"][0]["statistics"] else response["items"][0]["statistics"]["viewCount"],
           "likeCount": 0 if "likeCount" not in response["items"][0]["statistics"] else response["items"][0]["statistics"]["likeCount"],
           "commentCount": 0 if "commentCount" not in response["items"][0]["statistics"] else response["items"][0]["statistics"]["commentCount"]}
    # add value to the dict
    return ret


def get_video_transcript(video_id: str) -> dict:
    """
    get the transcript of a video given the video id
    :param video_id: id of the video to get the transcript for
    :return: the transcript of the video
    """
    text_transcript = ""
    # get the transcript
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = transcript_list.find_generated_transcript(['it'])
    transcript = transcript.fetch()
    for trans in transcript:
        text_transcript += " " + trans["text"]
    ret = {"transcript": text_transcript}
    return ret


def get_speech_from_links(path_to_links: str) -> []:
    speech_ret = []
    with open(path_to_links) as f:
        # read the links from the file
        for line in f.readlines():
            # get the video_id
            video_id = substring.substringByInd(line, startInd=line.index("=") + 1).strip()
            try:
                # get the informations
                informations = get_video_informations(video_id)
                # get the transcrip
                transcript = get_video_transcript(video_id)
                # join the dicts
                informations.update(transcript)
                speech_ret.append(informations)
            except Exception as e:
                print(f"\nVideo {video_id} does not have informations or transcripts")
                print(f"\nError: {e}")
    return speech_ret


def main():
    #Defines the paths
    PATH = ""
    path_links = f"{PATH}/politicians_speech_link/"
    output_path = f"{PATH}/text/"

    screen_names = ["EnricoLetta", "CarloCalenda", "MatteoRenzi", "SilvioBerlusconi", "MatteoSalvini",  "GiorgiaMeloni", "GiuseppeConte"]


    for politician_name in screen_names:
        print(f"Get {politician_name} speeches")
        pol_name_links = path_links + politician_name + ".txt"

        ret = get_speech_from_links(pol_name_links)

        #write the output on a csv file
        with open(output_path + politician_name + ".csv", "w") as file:
            writer = csv.DictWriter(file, fieldnames=ret[0].keys())
            writer.writeheader()
            writer.writerows(ret)


if __name__ == '__main__':
    main()
