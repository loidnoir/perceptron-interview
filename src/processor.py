import argparse
import csv
import json
import logging.handlers
import os
import sys
from datetime import datetime

from utils import read_lines_zst

DOCUMENT_LIMIT = 0
COMMENT_LIMIT = 20
REPLY_LIMIT = 5

log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Process data files")

	parser.add_argument("file_path", type=str)
	parser.add_argument("out_path", type=str)
	parser.add_argument("--limit", type=int, default=DOCUMENT_LIMIT)
	parser.add_argument("--comment-limit", type=int, default=COMMENT_LIMIT)
	parser.add_argument("--reply-limit", type=int, default=REPLY_LIMIT)
	
	args = parser.parse_args()
	
	if not os.path.isfile(args.file_path):
		print(f"Error: Submission file not found: {args.submission_path}")
		sys.exit(1)
	
	input_file_path = args.file_path
	output_file_path = args.out_path

	DOCUMENT_LIMIT = args.limit
	COMMENT_LIMIT = args.comment_limit
	REPLY_LIMIT = args.reply_limit
	
	fields = []
	is_submission = "submissions" in input_file_path
	
	if not len(fields):
		if is_submission:
			fields = ["id","author","title","media","media_embed","url","num_comments","score","created_utc", "subreddit_id"]
		else:
			fields = ["id", "link_id", "parent_id", "author","body","score","created_utc","subreddit_id"]

	file_size = os.stat(input_file_path).st_size
	file_lines, bad_lines = 0, 0
	line, created = None, None
	
	thread_comment_counts = {}
	comment_reply_counts = {}
	
	output_file = open(output_file_path, "w", encoding='utf-8', newline="")
	writer = csv.writer(output_file)
	
	writer.writerow(fields)
	try:
		i = 0
		for line, file_bytes_processed in read_lines_zst(input_file_path):
			i += 1
			try:
				obj = json.loads(line)
				
				if not is_submission:
					link_id = obj.get('link_id')
					parent_id = obj.get('parent_id')
					
					if COMMENT_LIMIT > 0 and link_id:
						current_thread_count = thread_comment_counts.get(link_id, 0)
						if current_thread_count >= COMMENT_LIMIT:
							continue
					if REPLY_LIMIT > 0 and parent_id and parent_id != link_id:
						current_reply_count = comment_reply_counts.get(parent_id, 0)
						if current_reply_count >= REPLY_LIMIT:
							continue

					if link_id:
						thread_comment_counts[link_id] = thread_comment_counts.get(link_id, 0) + 1
					if parent_id and parent_id != link_id:
						comment_reply_counts[parent_id] = comment_reply_counts.get(parent_id, 0) + 1
				
				output_obj = []
				for field in fields:
					match field:
						case "created":
							value = datetime.fromtimestamp(int(obj['created_utc'])).strftime("%Y-%m-%d %H:%M")
						case "link":
							if 'permalink' in obj:
								value = f"https://www.reddit.com{obj['permalink']}"
							else:
								value = f"https://www.reddit.com/r/{obj['subreddit']}/comments/{obj['link_id'][3:]}/_/{obj['id']}/"
						case "author":
							value = f"u/{obj['author']}"
						case "text":
							if 'selftext' in obj:
								value = obj['selftext']
							else:
								value = ""
						case _:
							value = obj[field]

					output_obj.append(str(value).encode("utf-8", errors='replace').decode())
					
				writer.writerow(output_obj)
				created = datetime.utcfromtimestamp(int(obj['created_utc']))
				
				if i >= DOCUMENT_LIMIT and DOCUMENT_LIMIT > 0:
					break

			except json.JSONDecodeError as err:
				bad_lines += 1

			file_lines += 1

			if file_lines % 50000 == 0:
				log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : {(file_bytes_processed / file_size) * 100:.0f}%")

	except KeyError as err:
		log.info(f"Object has no key: {err}")
		log.info(line)
	except Exception as err:
		log.info(err)
		log.info(line)

	output_file.close()
	
	if not is_submission and thread_comment_counts:
		total_threads = len(thread_comment_counts)
		max_comments = max(thread_comment_counts.values()) if thread_comment_counts else 0
		avg_comments = sum(thread_comment_counts.values()) / total_threads if total_threads > 0 else 0
		log.info(f"Thread Summary\n\nTotal threads: {total_threads:,} \nMax comments per thread: {max_comments} \nAvg comments per thread: {avg_comments:.1f}")
		
		if comment_reply_counts:
			total_comments_with_replies = len(comment_reply_counts)
			max_replies = max(comment_reply_counts.values()) if comment_reply_counts else 0
			avg_replies = sum(comment_reply_counts.values()) / total_comments_with_replies if total_comments_with_replies > 0 else 0
			log.info(f"Reply Summary\n\nComments with replies: {total_comments_with_replies:,} \nMax replies per comment: {max_replies} \nAvg replies per comment: {avg_replies:.1f}")
	
	log.info(f"Complete : {file_lines:,} : {bad_lines:,}")