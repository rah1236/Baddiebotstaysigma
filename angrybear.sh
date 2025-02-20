#!/bin/bash

# Configuration
RECORD_DURATION=5
MODEL_NAME="llama3.2"
VOICE_MODEL="en_US-lessac-medium.onnx"
TEMP_TRANSCRIPT="transcript.txt"
TEMP_RESPONSE="response.txt"

# Function to check if required commands exist
check_dependencies() {
    for cmd in python3 ollama piper play; do
        if ! command -v "$cmd" > /dev/null 2>&1; then
            echo "Error: Required command '$cmd' not found"
            exit 1
        fi
    done
}

# Function to clean up temporary files
cleanup() {
    rm -f "$TEMP_TRANSCRIPT" "$TEMP_RESPONSE"
}

# Set up cleanup on script exit
trap cleanup EXIT

# Check dependencies
check_dependencies

echo "Starting voice processing pipeline..."

# Step 1: Record and transcribe audio
# echo "Recording for $RECORD_DURATION seconds..."
if ! python3 transcriber.py "$RECORD_DURATION" > "$TEMP_TRANSCRIPT"; then
    echo "Error: Transcription failed"
    exit 1
fi

# Check if transcription was successful
if [ ! -s "$TEMP_TRANSCRIPT" ]; then
    echo "Error: No transcription generated"
    exit 1
fi

# Get transcription content and construct LLM prompt
TRANSCRIPTION=$(cat "$TEMP_TRANSCRIPT")
LLM_PROMPT="You are a rude assistant. Please respond to: $TRANSCRIPTION"

# Print the full prompt for debugging
echo -e "\nSending the following prompt to LLM:"
echo "----------------------------------------"
echo "$LLM_PROMPT"
echo "----------------------------------------"

# Step 2: Process through Ollama
echo -e "\nProcessing through LLM..."
if ! ollama run "$MODEL_NAME" "$LLM_PROMPT" > "$TEMP_RESPONSE"; then
    echo "Error: LLM processing failed"
    exit 1
fi

# Print LLM response for debugging
echo -e "\nLLM Response:"
echo "----------------------------------------"
cat "$TEMP_RESPONSE"
echo "----------------------------------------"

# Step 3: Text-to-speech conversion
echo -e "\nConverting response to speech..."
if ! cat "$TEMP_RESPONSE" | piper --model "$VOICE_MODEL" --output-raw | play -r 22050 -e signed -b 16 -c 1 -t raw -; then
    echo "Error: Text-to-speech conversion failed"
    exit 1
fi

echo "Processing complete!"