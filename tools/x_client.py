import os
import tweepy


def post_to_x(content: str) -> str:
    client = tweepy.Client(
        consumer_key=os.environ["X_API_KEY"],
        consumer_secret=os.environ["X_API_SECRET"],
        access_token=os.environ["X_ACCESS_TOKEN"],
        access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"],
    )

    try:
        response = client.create_tweet(text=content)
    except tweepy.TweepyException as e:
        raise RuntimeError(
            f"X API error: {e}\n"
            "Check: app has Read+Write permissions and tokens were regenerated after changing permissions."
        ) from e

    tweet_id = response.data["id"]
    return f"https://twitter.com/i/web/status/{tweet_id}"
