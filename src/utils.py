import base64
import logging
import tempfile
from io import BytesIO

import cv2
import requests
import zstandard
from PIL import Image as PILImage

log = logging.getLogger("utils")

def get_base64(url, type, metadata):
    try:
        if url.startswith('data:'):
            header, encoded = url.split(",", 1)

            if 'image/' in header:
                format_type = header.split('image/')[1].split(';')[0]
                return {
                    'content': encoded,
                    'type': 'image',
                    'metadata': {'format': format_type, **metadata}
                }
            return None
        
        if 'imgur.com' in url and '/a/' not in url and not any(ext in url for ext in ['.jpg', '.png', '.gif', '.webp']):
            path = url.split('/')[-1]
            if path:
                url = f"https://i.imgur.com/{path}.jpg"
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        
        if type == 'image' or content_type.startswith('image/'):
            image = PILImage.open(BytesIO(response.content))
            
            metadata['width'] = image.width
            metadata['height'] = image.height
            
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return {
                'content': encoded,
                'type': 'image',
                'metadata': {'format': 'png', **metadata}
            }
            
        elif type == 'video' or content_type.startswith('video/'):
            encoded = base64.b64encode(response.content).decode('utf-8')
            
            if content_type:
                format_type = content_type.split('/')[1].split(';')[0]
            else:
                format_type = 'mp4'
            
            try:
                with tempfile.NamedTemporaryFile(suffix=f'.{format_type}') as temp_file:
                    temp_file.write(response.content)
                    temp_file.flush()
                    
                    cap = cv2.VideoCapture(temp_file.name)
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        
                        if fps > 0:
                            duration_seconds = frame_count / fps
                            metadata['duration_seconds'] = round(duration_seconds, 2)
                        
                        if width > 0 and height > 0:
                            metadata['resolution'] = f"{width}x{height}"
                        
                        cap.release()
            except Exception:
                pass
            
            metadata['format'] = format_type
            
            return {
                'content': encoded,
                'type': 'video', 
                'metadata': metadata
            }
            
        elif type == 'audio' or content_type.startswith('audio/'):
            encoded = base64.b64encode(response.content).decode('utf-8')
            
            if content_type:
                format_type = content_type.split('/')[1].split(';')[0]
            else:
                format_type = 'mp3'
            
            return {
                'content': encoded,
                'type': 'audio',
                'metadata': {'format': format_type, **metadata}
            }
        
        else:
            log.warning(f"Unknown content type for {url}: {content_type}")
            return None
            
    except Exception as e:
        log.error(f"Error downloading media {url}: {e}")
        return None

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
	chunk = reader.read(chunk_size)
	bytes_read += chunk_size
	if previous_chunk is not None:
		chunk = previous_chunk + chunk
	try:
		return chunk.decode()
	except UnicodeDecodeError:
		if bytes_read > max_window_size:
			raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
		return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

def read_lines_zst(file_name):
	with open(file_name, 'rb') as file_handle:
		buffer = ''
		reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
		while True:
			chunk = read_and_decode(reader, 2**27, (2**29) * 2)
			if not chunk:
				break
			lines = (buffer + chunk).split("\n")

			for line in lines[:-1]:
				yield line, file_handle.tell()

			buffer = lines[-1]
		reader.close()