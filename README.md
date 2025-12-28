# AI Fitness Tracker

A real-time exercise tracking application that uses computer vision to count repetitions for various exercises. Built with Flask, OpenCV, and MediaPipe for DeltaHacks.

## Features

- **Real-time Exercise Detection**: Uses your webcam to track and count exercise repetitions
- **Multiple Exercise Support**:
  - Bicep Curls
  - Shoulder Press
  - Lateral Raises
- **Live Angle Visualization**: Displays arm angles in real-time for form feedback
- **Web-Based Interface**: Clean, modern UI accessible through your browser
- **Rep Counter**: Automatically counts repetitions for each exercise

## Technologies Used

- **Flask**: Web framework for the application
- **OpenCV**: Computer vision library for video capture and processing
- **MediaPipe**: Google's ML solution for pose detection and landmark tracking
- **NumPy**: Numerical computing for angle calculations
- **HTML/CSS**: Frontend interface

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd deltahacks
   ```

2. **Install dependencies**:
   ```bash
   pip install flask opencv-python mediapipe numpy
   ```

   Or create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install flask opencv-python mediapipe numpy
   ```

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Select an exercise** from the home page:
   - Click on "Bicep Curl" to track bicep curls
   - Click on "Shoulder Press" to track shoulder presses
   - Click on "Lateral Raise" to track lateral raises

4. **Allow camera access** when prompted by your browser

5. **Perform the exercise** in front of your camera:
   - The app will display your pose landmarks in real-time
   - The angle at your elbow will be shown on screen
   - Rep count will automatically increment as you complete each repetition

## How It Works

The application uses MediaPipe's Pose Detection to identify key body landmarks (shoulder, elbow, wrist) and calculates the angle at the elbow joint. Based on the angle thresholds:

- **Extended Position** (angle > 160°): Arm is straight (down position)
- **Flexed Position** (angle < 30°): Arm is fully curled (up position)

A repetition is counted when you move from the extended position to the flexed position.

## Project Structure

```
deltahacks/
├── app.py                 # Main Flask application with exercise tracking logic
├── bicep.py              # Standalone bicep curl counter (OpenCV only)
├── static/
│   └── styles.css        # Styling for the web interface
├── templates/
│   ├── index.html        # Home page
│   ├── bicepweb.html     # Bicep curl tracking page
│   ├── shoulderpress.html # Shoulder press tracking page
│   └── lateralraise.html  # Lateral raise tracking page
└── README.md             # This file
```

## Development

### Running in Debug Mode

The application runs in debug mode by default. To disable:

```python
# In app.py
if __name__ == '__main__':
    app.run(debug=False)
```

### Adding New Exercises

To add a new exercise:

1. Create a generator function in `app.py` following the pattern of existing exercises
2. Define the appropriate angle thresholds for the exercise
3. Create a new route for the exercise
4. Create a corresponding HTML template in the `templates/` folder
5. Add a link to the new exercise on the home page

## Troubleshooting

- **Camera not working**: Ensure your webcam is connected and not being used by another application
- **Slow performance**: Try reducing the video resolution or lowering the MediaPipe confidence thresholds
- **Inaccurate counting**: Adjust the angle thresholds in the generator functions to better match your exercise form

## Future Enhancements

- Add more exercise types (squats, push-ups, etc.)
- Implement form correction feedback
- Add workout history and statistics tracking
- Support for multiple camera angles
- Mobile device support

## Credits

Built for DeltaHacks using OpenCV, MediaPipe, and Python.

## License

MIT License - Feel free to use and modify as needed.
