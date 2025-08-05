import argparse
import json
import logging
import os
import sys

import pandas as pd

from src.schema import Audio, Document, Image, Reasoning, Role, Text, Video

DOCUMENT_LIMIT = 0

log = logging.getLogger("formatter")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

def main():
    parser = argparse.ArgumentParser(description="Format generated Q&A CSV into Perceptron Document objects.")
    parser.add_argument("input_csv", help="Path to the generated Q&A CSV file from the 'results/' directory.")
    parser.add_argument("output_dir", help="Directory to save the formatted sample files (e.g., 'samples/').")
    parser.add_argument("--limit", type=int, default=DOCUMENT_LIMIT, help="Maximum number of samples to create (0 for all).")
    
    args = parser.parse_args()

    try:
        log.info(f"Loading data from {args.input_csv}")
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError as e:
        log.error(f"Error: {e}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    log.info(f"Output will be saved in: {args.output_dir}")

    if args.limit > 0:
        df_to_process = df.head(args.limit)
        log.info(f"Processing {args.limit} rows...")
    else:
        df_to_process = df
        log.info(f"Processing all {len(df)} rows...")

    for index, row in df_to_process.iterrows():
        content = []

        content.append(Text(
            role=Role.USER,
            content=row['question'],
            metadata={'purpose': 'question', 'language': 'en'}
        ))

        if pd.notna(row['media']) and str(row['media']).strip():
            try:
                media_data = json.loads(str(row['media']))
                if 'content' in media_data and 'type' in media_data:
                    media_type = media_data['type']
                    encoded_content = media_data['content']
                    metadata = media_data.get('metadata', {})
                    
                    if media_type == 'image':
                        content.append(Image(
                            role=Role.USER, 
                            content=encoded_content,
                            metadata=metadata if metadata else {}
                        ))
                    elif media_type == 'audio':
                        content.append(Audio(
                            role=Role.USER, 
                            content=encoded_content,
                            metadata=metadata if metadata else {}
                        ))
                    elif media_type == 'video':
                        content.append(Video(
                            role=Role.USER, 
                            content=encoded_content,
                            tracks={},
                            metadata=metadata if metadata else {}
                        ))
                        
            except json.JSONDecodeError:
                pass
        
        if pd.notna(row['chain_of_thoughts']):
            content.append(Reasoning(
                role=Role.AGENT,
                content=row['chain_of_thoughts'],
                metadata={'purpose': 'chain_of_thoughts', 'language': 'en'}
            ))

        if pd.notna(row['answer']):
            content.append(Text(
                role=Role.AGENT, 
                content=row['answer'],
                metadata={'purpose': 'answer', 'confidence': row['score'] or "unknown", 'language': 'en'}
            ))


        doc_metadata = {
            'source': 'reddit_thread_analysis',
            'generated_at': pd.Timestamp.now().isoformat(),
        }
        
        doc = Document(content=content, metadata=doc_metadata)

        output_filename = f"sample_{index}.json"
        output_filepath = os.path.join(args.output_dir, output_filename)
        
        try:
            with open(output_filepath, "w") as f:
                json.dump(doc.to_json_dict(), f, indent=2)
            log.info(f"Successfully created {output_filepath}")
        except Exception as e:
            log.error(f"Could not write file {output_filepath}: {e}")

    log.info("Formatting complete.")


if __name__ == "__main__":
    main()
