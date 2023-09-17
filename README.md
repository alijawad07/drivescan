# DriveScan

This system processes video streams or image inputs to detect cars and their license plates using Optical Character Recognition (OCR). The results are saved in a CSV file for record-keeping. 

## Features

1. **Multiple Input Types**: Can process individual images, a series of images from a directory, RTSP video streams, or standard video files.
2. **CSV Records**: Detected license plates along with additional information (e.g., timestamp, confidence level) are stored in a CSV file.
3. **Configurable OCR**: Users have the flexibility to choose from multiple OCR methods.
4. **Real-time Streaming**: The processed video can be streamed in real-time to a web browser.
5. **Dynamic Confidence Adjustment**: Users can adjust the confidence levels for both detection and OCR tasks on-the-go.

## Installation

1. Clone the repository:
\```
git clone https://github.com/your_username/LicensePlateRecognition.git
\```

2. Change directory:
\```
cd LicensePlateRecognition
\```

3. Install the required dependencies (ensure you have Python installed):
\```
pip install -r requirements.txt
\```

## Usage

1. Run the Flask app:
\```
python main.py
\```

2. Open a web browser and navigate to:
\```
http://localhost:5000
\```

3. Follow the on-screen instructions to upload a video or image and start the license plate recognition process.

## Contribution

Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.
