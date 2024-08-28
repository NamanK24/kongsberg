import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import cv2
import tempfile
import os

# Force the use of CPU
device = "cpu"
st.write(f"Using device: {device}")

# Load the trained YOLOv8 model
model = YOLO("ppe.pt")  # Replace with the path to your ppe.pt file
model.to(device)

st.title("YOLOv8 Object Detection with Streamlit")

# File uploader for images and videos
uploaded_file = st.file_uploader(
    "Choose an image or a video...", type=["jpg", "jpeg", "png", "mp4"]
)

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

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

        # Iterate over the results and save/display each one
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
        tfile = tempfile.NamedTemporaryFile(delete=False)
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
        out = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 640)
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to 640x640
            frame_resized = cv2.resize(frame, (640, 640))

            # Convert frame to tensor and add batch dimension
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            image_tensor = transforms.ToTensor()(frame_pil).unsqueeze(0).to(device)

            # Run inference on the frame
            results = model(image_tensor)

            # Plot bounding boxes on the frame
            result_frame = results[0].plot()

            # Convert back to BGR for OpenCV and write the frame
            result_frame_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
            out.write(result_frame_bgr)

        # Release resources
        cap.release()
        out.release()

        # Ensure the video is completely written before trying to play it
        st.write("Detection complete. Video saved as 'processed_video.mp4'.")

        # Provide a video player for the processed video
        st.video(output_video_path, format="video/mp4")

    else:
        st.error("Unsupported file type! Please upload an image or a video.")
