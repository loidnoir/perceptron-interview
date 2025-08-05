import argparse
import csv
import json
import logging
import mimetypes
import os
import sys
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI

from src.schema import ModalityType
from utils import get_base64

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DOCUMENT_LIMIT = 20
SYSTEM_MESSAGE = """
# Instruction
- You are a high expert in reddit thread analysis.
- Your will be given a reddit thread sorted in chronological order.
- Read through the question and comments and find the answer.

# Key steps
- Pay attention to score to determine the importance and value of content
- Pay attention to chronological order of comments to use it when generating chain of thoughts
- In case of not clear answer try to generate relevant chain of thoughts and answer

# Chain of thoughts
- Chain of thoughts should be a detailed reasoning on how you came up with the answer. It should be heavily dependent on the order of comments and the way users discussed and answered the question.
- Chain of thoughts should be sufficient for a human to understand the question and the answer even from ground up.

# Scoring
- Provide a confidence score (0-100) indicating how certain you are about the answer
- Consider factors like: comment scores, number of confirming sources, clarity of evidence, and expert verification

# Response

- You MUST format your response exactly as follows:
- Do not add ``` or ```json or any formatting to your response

## Structure

chain_of_thoughts: [Your detailed reasoning on how you arrived at the answer, based on the thread's content.]
answer: [The final, concise answer.]
score: [Confidence score 0-100 indicating certainty of the answer.]

## Example:

chain_of_thoughts: The user posted an image of a small metallic object they found. Looking at the comments, user1 (score: 15) identified it as a vintage button from the 1940s based on the markings. User2 (score: 8) confirmed this and added that the specific pattern indicates it's from a military uniform. User3 (score: 12) provided additional context about the manufacturer. The high scores and consistent identification across multiple users gives high confidence.
answer: This is a vintage military uniform button from the 1940s, likely from a US Army uniform based on the eagle design and markings.
score: 85

"""

log = logging.getLogger("generator")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

def extract_media(submission):
    def get_type(url):
        match True:
            case _ if any(domain in url for domain in ['i.redd.it', 'imgur.com', 'i.imgur.com']):
                return ModalityType.IMAGE
            case _ if any(domain in url for domain in ['v.redd.it', 'youtube.com', 'youtu.be']):
                return ModalityType.VIDEO
            case _ if 'soundcloud.com' in url:
                return ModalityType.AUDIO
            case _ if url.startswith('data:image'):
                return ModalityType.IMAGE
            case _:
                parsed_url = urlparse(url)
                path = parsed_url.path.lower()
                match True:
                    case _ if any(ext in path for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        return 'image'
                    case _ if any(ext in path for ext in ['.mp4', '.webm', '.avi', '.mov']):
                        return 'video'
                    case _ if any(ext in path for ext in ['.mp3', '.wav', '.ogg', '.m4a']):
                        return 'audio'
                    case _:
                        return 'unknown'
    
    def get_metadata(fields, url):
        if not fields or pd.isna(fields):
            return {}
        
        try:
            if str(fields).startswith('{') and 'oembed' in str(fields):
                media_data = json.loads(str(fields))
                if 'oembed' in media_data:
                    oembed = media_data['oembed']
                    
                    if oembed.get('url') == url or oembed.get('thumbnail_url') == url:
                        metadata = {}

                        for key in ['width', 'height', 'thumbnail_width', 'thumbnail_height', 'duration_seconds', 'title']:
                            match key:
                                case 'width' if 'width' in oembed:
                                    metadata['width'] = oembed['width']
                                case 'height' if 'height' in oembed:
                                    metadata['height'] = oembed['height']
                                case 'duration_seconds' if 'duration_seconds' in oembed:
                                    metadata['duration_seconds'] = oembed['duration_seconds']
                                case _:
                                    pass

                        return metadata
                    
        except (json.JSONDecodeError, KeyError):
            pass
        
        return {}
    
    url = submission.get('url')
    fields = submission.get('media') or submission.get('media_embed')
    
    if not url or pd.isna(url) or str(url).strip() == "":
        return ""
    
    type = get_type(url)

    if type == 'unknown':
        log.warning(f"Warning: unknown media type for URL ({url})")
        return ""
    
    metadata = get_metadata(fields, url)
    result = get_base64(url, type, metadata)
    
    if result:
        return json.dumps(result)
    else:
        return ""

def get_summary(submission, comments_df):
    thread = []

    thread.append(f"Title: {submission['title']}")
    thread.append(f"Author: {submission['author']}")

    if pd.notna(submission['media']):
        thread.append(f"Media: {submission['media']}")
    if pd.notna(submission['url']):
        thread.append(f"URL: {submission['url']}")

    thread.append("\n--- Comments ---")
    comments_df = comments_df.sort_values(by="created_utc")

    for _, comment in comments_df.iterrows():
        comment_text = (
            f"\nAuthor: {comment['author']} | Score: {comment['score']}\n"
            f"{comment['body']}"
        )

        thread.append(comment_text)

    return "\n".join(thread)

def get_generation(thread_text, client):
    try:
        response = client.responses.create(
            model="gpt-4o",
            instructions=SYSTEM_MESSAGE,
            input=thread_text,
        )

        content = response.output_text
        chain_of_thoughts = ""
        answer = ""
        score = ""

        if "chain_of_thoughts:" in content and "answer:" in content and "score:" in content:
            sections = content.split("answer:")
            chain_of_thoughts = sections[0].replace("chain_of_thoughts:", "").strip()
            
            remaining = sections[1]
            score_split = remaining.split("score:")
            answer = score_split[0].strip()
            score = score_split[1].strip() if len(score_split) > 1 else ""

        elif "chain_of_thoughts:" in content and "answer:" in content:
            parts = content.split("answer:")
            chain_of_thoughts = parts[0].replace("chain_of_thoughts:", "").strip()
            answer = parts[1].strip()

        else:
            answer = content

        return chain_of_thoughts, answer, score

    except Exception as e:
        log.error(f"Error calling OpenAI API: {e}")
        return f"Error: {e}", f"Error: {e}", "0"

def main():
    parser = argparse.ArgumentParser(description="Generate thread analysis")
    parser.add_argument("sub_file", type=str)
    parser.add_argument("com_file", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--limit", type=int, default=DOCUMENT_LIMIT)

    args = parser.parse_args()

    try:
        log.info(f"Loading submissions ({args.sub_file}) and comments ({args.com_file})")
        submissions_df = pd.read_csv(args.sub_file)
        comments_df = pd.read_csv(args.com_file)
    except FileNotFoundError as e:
        log.error(f"Error loading files: {e}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    output_filename = f"thread-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    output_filepath = os.path.join(args.out_dir, output_filename)
    output_file = open(output_filepath, "w", encoding='utf-8', newline="")

    writer = csv.writer(output_file)
    header = ["question", "media", "chain_of_thoughts", "answer", "score"]
    writer.writerow(header)
        
    processed = 0
    for i, submission in submissions_df.iterrows():
        if args.limit > 0 and processed >= args.limit:
            log.info(f"Specified limit reached ({args.limit})")
            break

        id = submission['id']
        log.info(f"Processing submission: {processed+1} ({id})")

        link_id = f"t3_{id}"
        comments = comments_df[comments_df['link_id'] == link_id]

        if comments.empty:
            log.info(f"Skipping: no comments ({id})")
            continue
            
        media = extract_media(submission)
        summary = get_summary(submission, comments)
        cot, answer, score = get_generation(summary, client)

        writer.writerow([
            submission['title'], # Question
            media, # Media
            cot, # Chain of thoughts
            answer, # Answer
            score # Score
        ])
        
        processed += 1
        log.info(f"Finished submission ({id})")

    output_file.close()
    log.info(f"Successfully generated ({output_filepath})")

if __name__ == "__main__":
    main()
