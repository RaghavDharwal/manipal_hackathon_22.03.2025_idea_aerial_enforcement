from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Function to detect brick-red and concrete gray colors, displaying color name on bounding box
def detect_colors(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ### Brick-Red Color Detection
    lower_brick_red_light = np.array([50, 50, 200])
    upper_brick_red_light = np.array([80, 127, 255])
    
    mask_brick_red_light = cv2.inRange(hsv_frame, lower_brick_red_light, upper_brick_red_light)

    ### Concrete Gray Color Detection
    lower_concrete_gray = np.array([150, 150, 150])
    upper_concrete_gray = np.array([220, 220, 220])
    mask_concrete_gray = cv2.inRange(hsv_frame, lower_concrete_gray, upper_concrete_gray)

    ### Combine both masks (brick-red, concrete gray)
    combined_mask = cv2.bitwise_or(mask_brick_red_light, mask_concrete_gray)

    # Apply the combined mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # Find contours of the detected areas
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area_threshold = 6000 # Pixels

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > area_threshold:  # Only consider contours with area larger than 8623 pixels
            # Get bounding box coordinates for the large contour
            x, y, w, h = cv2.boundingRect(contour)
            # Determine which color is detected based on which mask the contour matches
            if cv2.countNonZero(cv2.bitwise_and(mask_brick_red_light, cv2.drawContours(np.zeros_like(mask_brick_red_light), [contour], -1, 255, cv2.FILLED))) > 0:
                color_name = 'concrete gray'
            elif cv2.countNonZero(cv2.bitwise_and(mask_concrete_gray, cv2.drawContours(np.zeros_like(mask_concrete_gray), [contour], -1, 255, cv2.FILLED))) > 0:
                color_name = 'Brick Red'
            
            # Draw a rectangle around the large area
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Put the color name on top of the rectangle
            cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Optional: Put the area size on the rectangle
            cv2.putText(frame, f"Area: {int(area)}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

# Function to generate video frames
def generate_frames():
    cap = cv2.VideoCapture('http://192.168.43.1:8080/video')  # Use 0 to capture from the default webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process the frame for color detection
        frame = detect_colors(frame)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return frame as byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/golive.html')
def index():
    return render_template('golive.html')

@app.route('/weather.html')
def weather():
    return render_template('weather.html')

@app.route('/control.html')
def control():
    return render_template('control.html')

@app.route('/compare.html')
def compare():
    return render_template('compare.html')

@app.route('/compare2.html')
def compare2():
    return render_template('compare2.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/location1.html')
def location1():
    return render_template('location1.html')


@app.route('/shedule.html')
def shedule():
    return render_template('shedule.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == "__main__":

    app.run(debug=True)