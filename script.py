import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import os
import cv2
from twilio.rest import Client
import tempfile

# Initialize Twilio Client


account_sid = ""                           # ACCESS IT FROM KEY.TXT  
auth_token = ""                            # ACESS IT FROM KEY.TXT 
client = Client(account_sid, auth_token)

# Force the use of CPU
device = "cpu"
st.write(f"Using device: {device}")


# Load the trained YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("ppe.pt")  # Replace with the path to your ppe.pt file


model = load_model()
model.to(device)

st.title("YOLOv8 Object Detection with Streamlit")


# # Function to send WhatsApp notification
def send_whatsapp_notification(missing_classes):
    missing_classes_str = ", ".join(missing_classes)
    # Send WhatsApp Message with missing classes
    message = client.messages.create(
        from_="whatsapp:+14155238886",  # Twilio Sandbox WhatsApp number
        body=f"Alert: The following required equipment is missing: {missing_classes_str}",
        to="whatsapp:+917263002829",  # Replace with your WhatsApp number
    )
    print(f"WhatsApp notification sent. Message SID: {message.sid}")


# File uploader for images and videos
uploaded_file = st.file_uploader(
    "Choose an image or a video...", type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    no_detections = True  # Flag to track detections

    if file_extension in [".jpg", ".jpeg", ".png"]:
        # Process as image
        image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Running detection on image...")

        # Define a transformation to resize the image and convert it to a PyTorch tensor
        transform = transforms.Compose(
            [
                transforms.Resize((640, 640)),  # Resize image to 640x640 pixels
                transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
            ]
        )

        # Apply transformation to the image
        image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Run inference on the image
        results = model(image_tensor)

        # List of required classes
        required_classes = {
            "Boots",
            "Ear-protection",
            "Glass",
            "Glove",
            "Helmet",
            "Mask",
            "Vest",
        }
        detected_classes = set()

        # Extract the class names from the model's detections
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]  # Get the class name
                detected_classes.add(class_name)  # Add the detected class to the set

        # Determine which classes are missing after processing the image
        missing_classes = required_classes - detected_classes

        # Send a WhatsApp message if any required class is missing


        if missing_classes:
            send_whatsapp_notification(missing_classes)

        # Display the detected image
        for i, result in enumerate(results):
            result_img = result.plot()  # Create an image with bounding boxes drawn
            result_img_pil = Image.fromarray(result_img)  # Convert to PIL Image

            output_path = f"output_{i}.jpg"
            result_img_pil.save(output_path)  # Save the result image

            # Display the output image
            st.image(
                result_img_pil, caption=f"Detected Image {i+1}", use_column_width=True
            )

    elif file_extension == ".mp4":
        # Process as video
        # Process as video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.write("Running detection on video...")

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a temporary file to save the output video
        temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_video_path = temp_output_file.name

        # Use OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a progress bar
        progress_bar = st.progress(0)

        # Define transform for resizing frames
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ]
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize and convert frame to tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).unsqueeze(0).to(device)

            # Run inference on the frame
            results = model(frame_tensor)

            # Check for detections
            if results[0].boxes.shape[0] > 0:
                no_detections = False

            # Plot bounding boxes on the original frame
            result_frame = results[0].plot()

            # Resize result frame back to original size
            result_frame = cv2.resize(result_frame, (width, height))

            # Convert back to BGR for OpenCV and write the frame
            result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            out.write(result_frame_bgr)

            # Update progress bar
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)

        # Release resources
        cap.release()
        out.release()

        st.write("Detection complete. Preparing video for playback...")

        # Read the processed video file
        with open(output_video_path, "rb") as file:
            video_bytes = file.read()

        # Display the video using st.video
        st.video(video_bytes)

        # Provide a download button for the processed video
        st.download_button(
            label="Download Processed Video",
            data=video_bytes,
            file_name="processed_video.mp4",
            mime="video/mp4",
        )

        # Clean up temporary files
        os.unlink(video_path)
        os.unlink(output_video_path)

        pass

    else:
        st.error("Unsupported file type! Please upload an image or a video.")

    # Show an alert within Streamlit if no detections were made
    if no_detections:
        st.warning("Alert: No detections were found in the processed image.")
