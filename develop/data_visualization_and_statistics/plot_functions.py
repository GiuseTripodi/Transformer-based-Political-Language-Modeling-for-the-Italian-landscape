import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from develop.inference.compute_metrics import today, PLOT_PATH


def define_structure_for_line_plots(dataset_input, eval_predict) -> pd.DataFrame:
    """
    creates the dataframe that will be used to generate all line graphs
    :return: dataframe
    """
    df = pd.DataFrame(eval_predict).drop(["logits", "best_class_code"], axis=1)
    df = df.rename(columns={"label": "assigned_label"})

    # input dataframe
    df_input = dataset_input.copy()
    df_input["impression"] = df_input[
        ["viewCount", "likeCount", "commentCount", "retweet_count", "reply_count", "quote_count"]].sum(axis=1)
    df_input.drop(df_input.columns.difference(["video_id", "created_at", "text", "label", "tweet_id", "impression"]), 1,
                  inplace=True)
    df_input.rename(columns={"label": "original_label"}, inplace=True)

    # concat the dataframe
    df_input = pd.concat([df_input, df], axis=1)
    return df_input


def line_plot_correlation_score_impression(df: pd.DataFrame):
    fontsize = 12
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))

    # TWEETS PLOTTING
    df_tweets = df.loc[df["tweet_id"] != 0]
    # plotting tweets
    ax[0][0].set_title("Tweets score impression correlation", fontdict={"fontsize": fontsize})
    sns.histplot(data=df_tweets, x="impression", y="score", cbar=True, bins=30, ax=ax[0][0])

    ax[0][1].set_title("Tweets impression Distributions", fontdict={"fontsize": fontsize})
    sns.kdeplot(data=df_tweets, x="impression", weights="score", hue="original_label", ax=ax[0][1])

    # SPEECH plotting
    df_speech = df.loc[df["video_id"] != "0"]
    df_speech = df_speech.groupby(["video_id", "original_label"])["score", "impression"].mean()
    # plotting
    ax[1][0].set_title("Speech score impression correlation", fontdict={"fontsize": fontsize})
    sns.histplot(data=df_speech, x="impression", y="score", cbar=True, bins=30, ax=ax[1][0])

    ax[1][1].set_title("Speech impression Distributions", fontdict={"fontsize": fontsize})
    sns.kdeplot(data=df_speech, x="impression", weights="score", hue="original_label", ax=ax[1][1])
    plt.savefig(f"{PLOT_PATH}/score_impression_correlation_{today}.png")


def line_plot_correlation_score_period(df: pd.DataFrame):
    df["created_at"] = pd.to_datetime(df["created_at"], format="%Y-%m-%d %H:%M:%S+00:00")
    fontsize = 12
    fig, ax = plt.subplots(2, 2, figsize=(20, 10))

    # TWEETS PLOTTING
    df_tweets = df.loc[df["tweet_id"] != 0]
    df_tweets = df_tweets.groupby([df.created_at.dt.month_name().rename('month'), "original_label"])[
        "score", "impression"].mean()

    # plotting tweets
    ax[0][0].tick_params(labelrotation=25)
    ax[0][0].set_title("Tweets score period correlation", fontdict={"fontsize": fontsize})
    sns.histplot(data=df_tweets, x="month", y="score", cbar=True, bins=30, ax=ax[0][0])

    ax[0][1].tick_params(labelrotation=25)
    ax[0][1].set_title("Tweets impression Distributions", fontdict={"fontsize": fontsize})
    hist = sns.histplot(data=df_tweets, x="month", weights="score", multiple="dodge", shrink=1, binwidth=5,
                        hue="original_label", ax=ax[0][1])
    sns.move_legend(hist, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # SPEECH PLOTTING
    df_speech = df.loc[df["video_id"] != "0"]
    df_speech = df_speech.groupby([df.created_at.dt.month_name().rename('month'), "original_label"])[
        "score", "impression"].mean()

    # plotting
    ax[1][0].tick_params(labelrotation=15)
    ax[1][0].set_title("Speech score period correlation", fontdict={"fontsize": fontsize})
    sns.histplot(data=df_speech, x="month", y="score", cbar=True, bins=30, ax=ax[1][0])

    ax[1][1].tick_params(labelrotation=15)
    ax[1][1].set_title("Speech impression Distributions", fontdict={"fontsize": fontsize})
    hist_s = sns.histplot(data=df_speech, x="month", weights="score", multiple="dodge", shrink=1, binwidth=5,
                          hue="original_label", ax=ax[1][1])
    sns.move_legend(hist_s, bbox_to_anchor=(1.02, 0.5), loc='upper left', borderaxespad=0)
    fig.tight_layout(pad=2.0)
    plt.savefig(f"{PLOT_PATH}/score_period_correlation_{today}.png")