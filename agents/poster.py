from tools.linkedin_client import post_to_linkedin
from tools.x_client import post_to_x


def post_linkedin(content: str) -> str:
    return post_to_linkedin(content)


def post_tweet(content: str) -> str:
    return post_to_x(content)
