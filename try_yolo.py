import cv2
import time
from ultralytics import YOLO
import pyttsx3
import logging
import struct
import pyaudio
import pvporcupine
import threading # New library for threading

# -------------------------------------
# Configuration
# -------------------------------------
PICOVOICE_ACCESS_KEY = "YOUR-ACCESS-KEY-HERE"
WAKE_WORD_PATH = "hey_sid.ppn" # Use your Mac wake word file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# A flag to prevent multiple detections from running at once
is_detecting = False

# -------------------------------------
# Object Detection and TTS Functions
# -------------------------------------
def create_summary_sentence(detections):
    if not detections:
        return "I could not find any objects."

    descriptions = [f"a {item['label']}" for item in detections]
    
    if len(descriptions) == 1:
        summary = f"I see {descriptions[0]}."
    elif len(descriptions) == 2:
        summary = f"I see {descriptions[0]} and {descriptions[1]}."
    else:
        last_item = descriptions.pop()
        summary = f"I see {', '.join(descriptions)}, and {last_item}."
        
    return summary

def analyze_frame(frame, model):
    logging.info("Analyzing frame with local model...")
    results = model(frame, conf=0.45, verbose=False)
    
    logging.info("Analysis Complete!")
    if not results or len(results[0].boxes) == 0:
        return []

    detections = []
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        confidence = round(box.conf.item() * 100)
        detections.append({'label': label, 'confidence': confidence})
        print(f"- Found a {label} (Confidence: {confidence}%)")
    
    return detections

def run_detection_cycle(model):
    """The main object detection and TTS workflow that will run in a separate thread."""
    global is_detecting
    is_detecting = True # Set the flag to true
    
    logging.info("--- Wake word detected! Starting detection cycle. ---")
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logging.error("CRITICAL: Could not open camera.")
        is_detecting = False # Reset the flag
        return

    time.sleep(1)
    ret, frame = camera.read()
    
    if ret:
        detections = analyze_frame(frame, model)
        summary = create_summary_sentence(detections)
        
        logging.info(f"Speaking: \"{summary}\"")
        engine = pyttsx3.init()
        engine.say(summary)
        engine.runAndWait() # This will only block the new thread, not the main listening loop
    else:
        logging.error("Failed to capture image from the camera.")
        
    camera.release()
    cv2.destroyAllWindows()
    logging.info("--- Detection cycle finished. Returning to listening. ---")
    is_detecting = False # Reset the flag so we can detect again

# -------------------------------------
# Main Script with Wake Word Loop
# -------------------------------------
def main():
    porcupine = None
    pa = None
    audio_stream = None
    
    try:
        logging.info("--- Script starting ---")
        
        logging.info("Loading YOLOv8 model...")
        model = YOLO('yolov8n.pt')
        logging.info("✅ Model loaded successfully.")

        logging.info("Initializing Porcupine wake word engine...")
        porcupine = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY,
            keyword_paths=[WAKE_WORD_PATH]
        )

        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        
        logging.info("✅ Ready and listening for 'Hey Sid'...")

        while True:
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False) # Important: prevent crash on minor overflow
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            
            result = porcupine.process(pcm)
            
            if result >= 0 and not is_detecting:
                # Start the detection cycle in a new thread
                detection_thread = threading.Thread(target=run_detection_cycle, args=(model,))
                detection_thread.start()

    except KeyboardInterrupt:
        logging.info("Received stop signal. Shutting down.")
    finally:
        logging.info("Cleaning up resources.")
        if audio_stream is not None:
            audio_stream.close()
        if pa is not None:
            pa.terminate()
        if porcupine is not None:
            porcupine.delete()
        logging.info("--- Script finished ---")

if __name__ == "__main__":
    main()
