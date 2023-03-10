"""Get Politicians Tweets

This script allows the user to retrieve a politician's tweets, given the Twitter account name.
The script obtains the id associated with the politician and returns all tweets from that particular politician.

"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import csv
import tweepy


class Tweets:
    """
    This class contains all the necessary method to retrieve the user's tweets
    """
    def __init__(self):
        # Get your Twitter API credentials and enter them here
        baerer_token = ""
        access_key = ""
        access_secret = ""

        # https://docs.tweepy.org/en/stable/client.html#tweets
        self.client = tweepy.Client(baerer_token, access_token=access_key, access_token_secret=access_secret)

    def get_tweets_by_username(self, username, output_dir):
        """
        gets tweets by the user username
        :param username: a string, username to use
        :return: creates a csv file with the retrieved_tweets of the user
        """
        user = self.client.get_user(username=username)
        user_id = user.data.id
        tweets_for_csv, header = self.get_tweets_by_user_id(user_id)

        #save retrieved_tweets to a csv file
        outfile = f"{output_dir}/{username}_25-09-21_to_25-09-2022.csv"
        print("writing to " + outfile)
        with open(outfile, 'w+') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(header)
            writer.writerows(tweets_for_csv)

    def get_tweets_by_user_id(self, user_id):
        """
        gets retrieved_tweets by the used id
        :param user_id: the id of the users
        :return: a list with the retrieved_tweets of the user
        """
        # set count to however many retrieved_tweets you want
        number_of_tweets = 500
        tweet_fields = ["author_id", "id", "created_at", "text",  "public_metrics"]
        # put retrieved_tweets in a csv file
        tweets_for_csv = []
        for tweet in tweepy.Paginator(self.client.search_all_tweets,
                                      query=f"from:{user_id} -is:retweet",
                                      tweet_fields=tweet_fields,
                                      max_results=number_of_tweets,
                                        start_time="2021-09-25T00:00:01Z",
                                        end_time="2022-09-25T00:00:01Z"
                                      ).flatten():
            # creates array of tweet information: username, tweet id, date/time, text
            print(f"tweet {tweet.id} retrived")
            tweets_for_csv.append([user_id, tweet.id, tweet.created_at, tweet.text, tweet.public_metrics["retweet_count"],
                                   tweet.public_metrics["reply_count"], tweet.public_metrics["like_count"], tweet.public_metrics["quote_count"]])

        header = ["author_id", "id", "created_at", "text", "retweet_count", "reply_count", "like_count", "quote_count"]
        return tweets_for_csv, header


#Define the main to retrieve data for all relevant policies
def main():
    t = Tweets()
    HOME_PATH = ""
    output_dir = f"{HOME_PATH}/data/retrieved_tweets"
    screen_names = ["EnricoLetta", "GiuseppeConteIT" ,"CarloCalenda", "MatteoRenzi", "berlusconi", "matteosalvinimi",  "GiorgiaMeloni" ]
    for screen_name in screen_names:
        print(f"Get {screen_name} retrieved_tweets\n")
        t.get_tweets_by_username(screen_name, output_dir)

if __name__ == '__main__':
    main()