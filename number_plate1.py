import google.generativeai as genai
import PIL.Image
import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# --- Configuration ---
# It's best practice to load the API key from an environment variable.
# This prevents you from accidentally sharing your secret key in your code.
API_KEY = os.environ.get('GOOGLE_API_KEY')
if not API_KEY:
    # A fallback for local testing, but not recommended for production.
    # Replace with your actual key if not using environment variables.
    API_KEY = 'AIzaSyDP8Ka-Iz1A0Qv5Xm_a-ocVOhZeFEPDXcs' 

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Failed to configure Generative AI: {e}")
    # Consider exiting if the API key is essential for the app to start
    # exit()

# --- Initialize the FastAPI App ---
# This creates an instance of our web application.
# We also add metadata for the automatic documentation.
app = FastAPI(
    title="Vehicle Number Detection API",
    description="Upload an image of a vehicle to extract the license plate number using Google Gemini.",
    version="1.0.0"
)

# --- The Core API Logic in a Reusable Function ---
# We define this as an 'async' function to work well with FastAPI's asynchronous nature.
async def analyze_vehicle_image(image_bytes: bytes):
    """
    Takes image bytes, sends them to the Gemini API, and returns the result.
    """
    try:
        # Create an in-memory binary stream from the bytes
        image_stream = io.BytesIO(image_bytes)
        # Open the image from the stream
        img = PIL.Image.open(image_stream)

        # Use a reliable and capable model for this task.
        # 'gemini-1.5-pro-latest' is excellent for vision tasks.
        # 'gemini-2.0-flash' is not a valid model name.
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = """
        Analyze the provided image. Your primary task is to identify and extract a vehicle's registration number from its license plate.

        Follow these rules strictly:
        1. If you find a clear license plate, provide ONLY the extracted registration number in a machine-readable format (e.g., MH12AB3456). Do not add any extra words.
        2. If the image does not contain a vehicle, respond with the exact text: NO_VEHICLE_FOUND
        3. If the image contains a vehicle but the license plate is not visible or is unreadable, respond with the exact text: PLATE_UNREADABLE
        """

        # Make the API call to Gemini
        response = model.generate_content([prompt, img])
        
        # Return the clean text result
        return response.text.strip()

    except Exception as e:
        # If anything goes wrong during the analysis, log it and raise an exception.
        print(f"An error occurred during Gemini analysis: {e}")
        # Raising an HTTPException is the standard FastAPI way to send error responses.
        raise HTTPException(status_code=500, detail=e)

# --- API Endpoint Definition ---
# This decorator defines the URL and the allowed method.
# Our API will be at http://127.0.0.1:8000/analyze
@app.post("/analyze")
async def handle_analysis_request(image: UploadFile = File(...)):
    """
    This endpoint accepts an image file, analyzes it to find a vehicle
    registration number, and returns the result.
    """
    # 1. Read the contents (bytes) of the uploaded file.
    # The 'await' keyword is used because 'read()' is an async operation.
    image_bytes = await image.read()

    # 2. Call our core logic function to get the analysis result.
    result_text = await analyze_vehicle_image(image_bytes)
    
    # 3. Prepare a successful JSON response based on the result.
    response_data = {}

    if result_text == 'NO_VEHICLE_FOUND':
        response_data['code'] = 'NO_VEHICLE'
        response_data['message'] = 'The uploaded image does not appear to contain a vehicle.'
    elif result_text == 'PLATE_UNREADABLE':
        response_data['code'] = 'PLATE_UNREADABLE'
        response_data['message'] = 'A vehicle was found, but the license plate could not be read.'
    else:
        # This is the successful case!
        response_data['code'] = 'SUCCESS'
        response_data['message'] = 'Successfully extracted vehicle number.'
        response_data['vehicle_number'] = result_text

    # 4. Return the response. FastAPI automatically converts the dictionary to JSON.
    return JSONResponse(content=response_data)

# --- A simple root endpoint for health checks ---
@app.get("/")
def read_root():
    return {"status": "Vehicle Analysis API is running."}
