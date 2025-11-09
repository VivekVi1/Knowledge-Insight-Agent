import boto3
import os
import sys
import time
import json
import base64
import cv2  # OpenCV for video frame extraction
import pprint
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# LangChain & AWS Imports
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---
# Import the prompt loader
# ---
try:
    from decoupled.utils import load_prompt
except ImportError:
    from utils import load_prompt


# ---
# 1. BEDROCK & CLIENT INITIALIZATIONS
# ---

load_dotenv()
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.getenv("BEDROCK_API")
AWS_TRANSCRIBE_S3_BUCKET = os.getenv("AWS_TRANSCRIBE_S3_BUCKET")
AWS_REGION = "us-east-1" # Hardcoding region for bucket creation

os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_SESSION_TOKEN"] = os.getenv("AWS_SESSION_TOKEN")


try:
    boto_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION
    )
    s3_client = boto3.client("s3")
    transcribe_client = boto3.client("transcribe")

    llm = ChatBedrockConverse(
        model="amazon.nova-pro-v1:0",  # This model handles text and vision
        client=boto_client,
        max_tokens=2048,
        temperature=0.0  # Use 0.0 for consistent, factual outputs
    )
   
    
    print("MediaProcessor: Bedrock, S3, and Transcribe clients initialized.")

except Exception as e:
    print(f"Error initializing Boto3 clients or LLMs in MediaProcessor: {e}")
    sys.exit(1)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)

try:
    PROMPT_AUDIO_ANALYSIS = load_prompt("audio_analysis.txt")
    PROMPT_VIDEO_FRAME = load_prompt("video_frame_description.txt")
    PROMPT_VIDEO_SUMMARY = load_prompt("video_summary.txt")
except Exception as e:
    print("Error loading prompts. Make sure 'prompts/' directory is in the project root.")
    sys.exit(1)


def setup_transcribe_bucket(bucket_name):
    """Creates the S3 bucket required for Transcribe."""
    if not bucket_name:
        raise ValueError("AWS_TRANSCRIBE_S3_BUCKET env variable is not set.")
    
    print(f"\n[S3 Setup] Checking for bucket: {bucket_name}...")
    try:
        if AWS_REGION == "us-east-1":
            # 'us-east-1' is the default and has no LocationConstraint
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
            )
        print(f"[S3 Setup] ✓ Successfully created bucket: {bucket_name}")
        
    except s3_client.exceptions.BucketAlreadyOwnedByYou:
        print(f"[S3 Setup] ✓ Bucket already exists and you own it. Reusing.")
    except s3_client.exceptions.BucketAlreadyExists:
        print(f"[S3 Setup] ✗ ERROR: Bucket '{bucket_name}' already exists and is owned by someone else.")
        print("   Please choose a different bucket name in your .env file.")
        sys.exit(1)
    except Exception as e:
        print(f"[S3 Setup] ✗ ERROR creating bucket: {e}")
        sys.exit(1)

def cleanup_transcribe_bucket(bucket_name):
    """Deletes all objects in the bucket, then deletes the bucket itself."""
    if not bucket_name:
        print("[S3 Cleanup] No bucket name specified, skipping cleanup.")
        return
        
    print(f"\n[S3 Cleanup] Cleaning up bucket: {bucket_name}...")
    try:
        # 1. List all objects in the bucket
        objects_to_delete = []
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name):
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects_to_delete.append({'Key': obj['Key']})
        
        # 2. Delete all objects if any exist
        if objects_to_delete:
            print(f"   - Deleting {len(objects_to_delete)} objects...")
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': objects_to_delete}
            )
            print("   - ✓ Objects deleted.")
        else:
            print("   - Bucket is already empty.")

        # 3. Delete the bucket itself
        s3_client.delete_bucket(Bucket=bucket_name)
        print(f"[S3 Cleanup] ✓ Successfully deleted bucket: {bucket_name}")
        
    except Exception as e:
        print(f"[S3 Cleanup] ✗ ERROR during cleanup: {e}")
        print("   You may need to manually delete the S3 bucket.")


def create_output_directories(base_dir):
    """Creates the necessary output subdirectories for media."""
    for dir_name in ["audio", "video", "transcripts", "text", "other"]:
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "video", "frames"), exist_ok=True)


def upload_to_s3(filepath, bucket, s3_key):
    """Uploads a file to S3 for Transcribe."""
    try:
        print(f"   - Uploading {filepath} to s3://{bucket}/{s3_key}...")
        s3_client.upload_file(filepath, bucket, s3_key)
        print(f"   - ✓ Upload complete.")
        return f"s3://{bucket}/{s3_key}"
    except Exception as e:
        print(f"   - ✗ Error uploading to S3: {e}")
        raise

def transcribe_media(s3_uri, job_name, s3_key):
    """Starts and polls an AWS Transcribe job."""
    print(f"   - Starting Transcribe job: {job_name}...")
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat=s3_key.split('.')[-1],
            LanguageCode='en-US',
            OutputBucketName=AWS_TRANSCRIBE_S3_BUCKET # Specify output bucket
        )
    except transcribe_client.exceptions.ConflictException:
        print(f"   - Job '{job_name}' already exists. Reusing...")
        pass
    except Exception as e:
        print(f"   - ✗ Error starting Transcribe job: {e}")
        raise
    while True:
        try:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            
            if job_status in ['COMPLETED', 'FAILED']:
                print(f"   - Job {job_status}.")
                # Clean up the Transcribe job entry (but not the S3 files yet)
                transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)

                if job_status == 'COMPLETED':
                    output_key = f"{job_name}.json"
                    print(f"   - Fetching transcript from s3://{AWS_TRANSCRIBE_S3_BUCKET}/{output_key}")
                    
                    obj = s3_client.get_object(
                        Bucket=AWS_TRANSCRIBE_S3_BUCKET, 
                        Key=output_key
                    )
                    transcript_json = json.loads(obj['Body'].read().decode('utf-8'))
                    return transcript_json['results']['transcripts'][0]['transcript']
                else:
                    raise Exception("Transcription job failed.")
            else:
                print(f"   - Job status: {job_status}. Waiting 15s...")
                time.sleep(15)
        except Exception as e:
            print(f"   - ✗ Error polling Transcribe job: {e}")
            # Ensure job is deleted even on poll failure
            try:
                transcribe_client.delete_transcription_job(TranscriptionJobName=job_name)
            except:
                pass # Ignore errors on cleanup
            raise
    
def convert_image_to_base64(image_bytes: bytes) -> str:
    """Converts raw image bytes to a PNG base64 string."""
    try:
        image = Image.open(BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        base64_string = base64.b64encode(buffer.read()).decode('utf-8')
        return base64_string
    except Exception as e:
        print(f"   - Error converting image to base64: {e}")
        raise

def extract_video_frames(filepath, base_dir, frames_per_minute=2):
    """Extracts frames from a video file."""
    print(f"   - Extracting frames from {filepath}...")
    frame_paths = []
    try:
        vidcap = cv2.VideoCapture(filepath)
        if not vidcap.isOpened():
            print(f"   - ✗ Error opening video file: {filepath}")
            return []
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        interval_sec = 60 / frames_per_minute
        frame_timestamps = [i * interval_sec for i in range(int(duration_sec / interval_sec) + 1)]
        print(f"   - Video is {duration_sec:.2f}s long ({fps:.2f} FPS).")
        print(f"   - Capturing {len(frame_timestamps)} frames (approx {frames_per_minute}/min).")
        for ts in frame_timestamps:
            frame_num = int(ts * fps)
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            success, image = vidcap.read()
            if success:
                success, buffer = cv2.imencode('.png', image)
                if not success:
                    print(f"   - ✗ Failed to encode frame {frame_num}")
                    continue
                image_bytes = buffer.tobytes()
                frame_filename = f"{os.path.basename(filepath)}_frame_at_{int(ts)}s.png"
                frame_path = os.path.join(base_dir, "video", "frames", frame_filename)
                with open(frame_path, 'wb') as f:
                    f.write(image_bytes)
                frame_paths.append((frame_path, image_bytes, int(ts)))
        vidcap.release()
        print(f"   - ✓ Extracted {len(frame_paths)} frames.")
        return frame_paths
    except Exception as e:
        print(f"   - ✗ Error extracting frames: {e}")
        return []


def process_audio_file(filepath, base_dir, items):
    """Transcribes and analyzes an audio file."""
    file_basename = os.path.basename(filepath)
    s3_key = f"media-ingestion/{file_basename}"
    job_name = f"transcribe_{file_basename.split('.')[0]}_{int(time.time())}"
    
    try:
        s3_uri = upload_to_s3(filepath, AWS_TRANSCRIBE_S3_BUCKET, s3_key)
        transcript = transcribe_media(s3_uri, job_name, s3_key)
        
        transcript_path = f"{base_dir}/transcripts/{file_basename}_transcript.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"   - ✓ Saved transcript: {transcript_path}")
        items.append({"page": 0, "type": "transcript", "text": transcript, "path": transcript_path})
        
        print("   - Analyzing audio transcript...")
        prompt = PROMPT_AUDIO_ANALYSIS.format(transcript=transcript)
        
        # --- MODIFICATION: Use single llm instance ---
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content
        
        summary_path = f"{base_dir}/text/{file_basename}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"   - ✓ Saved audio summary: {summary_path}")
        items.append({"page": 0, "type": "audio_summary", "text": summary, "path": summary_path})
        
    except Exception as e:
        print(f"   - ✗ Failed to process audio file {filepath}: {e}")

def process_video_file(filepath, base_dir, items):
    """Transcribes, extracts frames, and analyzes a video file."""
    file_basename = os.path.basename(filepath)
    s3_key = f"media-ingestion/{file_basename}"
    job_name = f"transcribe_{file_basename.split('.')[0]}_{int(time.time())}"
    transcript = ""
    try:
        print("   - Starting video audio transcription...")
        s3_uri = upload_to_s3(filepath, AWS_TRANSCRIBE_S3_BUCKET, s3_key)
        transcript = transcribe_media(s3_uri, job_name, s3_key)
        
        transcript_path = f"{base_dir}/transcripts/{file_basename}_transcript.txt"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"   - ✓ Saved video transcript: {transcript_path}")
        items.append({"page": 0, "type": "transcript", "text": transcript, "path": transcript_path})
        
    except Exception as e:
        print(f"   - ✗ Failed to transcribe video audio {filepath}: {e}")
        print("   - Proceeding with visual analysis only.")
    
    try:
        frames = extract_video_frames(filepath, base_dir, frames_per_minute=2)
        frame_descriptions = []
        
        for i, (frame_path, frame_bytes, timestamp) in enumerate(frames):
            print(f"\n   - Analyzing frame {i+1}/{len(frames)} (at {timestamp}s)...")
            try:
                base64_image = convert_image_to_base64(frame_bytes)
                messages = [
                    HumanMessage(content=[
                        {"type": "text", "text": PROMPT_VIDEO_FRAME},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ])
                ]
                
                # --- MODIFICATION: Use single llm instance ---
                response = llm.invoke(messages)
                description = response.content
                
                desc_text = f"[Frame at {timestamp}s]: {description}"
                frame_descriptions.append(desc_text)
                desc_path = f"{base_dir}/text/{os.path.basename(frame_path)}_desc.txt"
                with open(desc_path, 'w', encoding='utf-8') as f:
                    f.write(desc_text)
                print(f"   - ✓ Generated description for frame at {timestamp}s")
                items.append({"page": timestamp, "type": "video_frame_description", "text": desc_text, "path": desc_path})
            except Exception as e:
                print(f"   - ✗ Error analyzing frame {i+1}: {e}")
                
        if transcript or frame_descriptions:
            print("\n   - Generating holistic video summary...")
            combined_context = (
                "VIDEO TRANSCRIPT:\n"
                f"{transcript if transcript else 'No transcript available.'}\n\n"
                "KEY FRAME DESCRIPTIONS:\n"
                f"{'\n'.join(frame_descriptions) if frame_descriptions else 'No frames analyzed.'}"
            )
            
            prompt = PROMPT_VIDEO_SUMMARY.format(context=combined_context)
            
            # --- MODIFICATION: Use single llm instance ---
            response = llm.invoke([HumanMessage(content=prompt)])
            summary = response.content
            
            summary_path = f"{base_dir}/text/{file_basename}_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"   - ✓ Saved holistic video summary: {summary_path}")
            items.append({"page": 0, "type": "video_summary", "text": summary, "path": summary_path})
            
    except Exception as e:
        print(f"   - ✗ Failed during video frame analysis: {e}")

def load_and_process_media(filepath):
    """
    Loads a media file, processes it, and extracts text, transcripts,
    and visual descriptions.
    Manages the lifecycle of the S3 bucket.
    """
    
    # Check for S3 bucket name
    if not AWS_TRANSCRIBE_S3_BUCKET:
        print("Error: 'AWS_TRANSCRIBE_S3_BUCKET' env variable must be set.")
        return []
    
    try:
        # --- BUCKET SETUP ---
        setup_transcribe_bucket(AWS_TRANSCRIBE_S3_BUCKET)
        
        # --- START PROCESSING ---
        if not os.path.exists(filepath):
            print(f"Error: File not found at '{filepath}'")
            return []

        file_root, file_ext = os.path.splitext(os.path.basename(filepath))
        file_ext = file_ext.lower()
        output_base_dir = f"output_{file_root}"
        
        print(f"\n{'='*60}")
        print(f"Creating output directory: {output_base_dir}")
        print(f"{'='*60}")
        create_output_directories(output_base_dir)

        processed_items = []
        audio_types = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        video_types = ['.mp4', '.mov', '.avi', 'mkv', '.webm']
        
        print(f"\n{'='*60}")
        print(f"Processing media file: {filepath}")
        print(f"{'='*60}")
        
        if file_ext in audio_types:
            process_audio_file(filepath, output_base_dir, processed_items)
        elif file_ext in video_types:
            process_video_file(filepath, output_base_dir, processed_items)
        else:
            print(f"Unsupported media file type: {file_ext}")
            return []
            
        print(f"\n{'='*60}")
        print(f"Media processing complete for: {filepath}")
        print(f"Total items extracted: {len(processed_items)}")
        print(f"{'='*60}")
        
        return processed_items

    except Exception as e:
        print(f"\n❌ A critical error occurred in 'load_and_process_media': {e}")
        return [] # Return empty list on failure
        
    finally:
        # --- BUCKET CLEANUP ---
        # This will run whether the try block succeeded or failed
        cleanup_transcribe_bucket(AWS_TRANSCRIBE_S3_BUCKET)

if __name__ == "__main__":

    print("="*80)
    print("MEDIA PROCESSOR - STANDALONE TEST MODE")
    print("="*80)
    
    if not AWS_TRANSCRIBE_S3_BUCKET:
        print("Error: 'AWS_TRANSCRIBE_S3_BUCKET' env variable must be set to run.")
        print("Please set this in your .env file.")
        sys.exit(1)
    
    # Ask the user for a single file path
    filepath_to_process = ""
    while not filepath_to_process:
        try:
            raw_input = input("📁 Enter the path to your media file: ").strip()
            
            if not raw_input:
                print("⚠ Please enter a file path.")
                continue
                
            if not os.path.exists(raw_input):
                print(f"⚠ Error: File not found at '{raw_input}'")
                print("   Please check the path and try again.")
                continue
            
            filepath_to_process = raw_input

        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Process cancelled by user. Exiting.")
            sys.exit(0)
    
    print(f"\nStarting to process file: {filepath_to_process}...")
    
    # Call the main function. Setup and cleanup are handled inside it.
    try:
        processed_items = load_and_process_media(filepath_to_process)
        
        if processed_items:
            print("\n" + "="*80)
            print("STANDALONE PROCESSING COMPLETE")
            print("="*80)
            print(f"Successfully extracted {len(processed_items)} items.")
            
            for i, item in enumerate(processed_items):
                print(f"\n--- Item {i+1} (Type: {item['type']}) ---")
                print(f"  Path: {item['path']}")
                print(f"  Page/TS: {item['page']}")
                print(f"  Text: {item['text'][:150]}...")
            
            print(f"\nAll local output files saved to: output_{os.path.splitext(os.path.basename(filepath_to_process))[0]}/")
        else:
            print("\n" + "="*80)
            print("STANDALONE PROCESSING FAILED")
            print("="*80)
            print("No items were extracted. See error messages above.")
        
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during processing: {e}")
        sys.exit(1)