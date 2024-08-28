import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
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
    # Check if the uploaded file is an image
    if uploaded_file.type in ["image/jpeg", "image/png"]:
        # Load image with PIL
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

    # Check if the uploaded file is a video
    elif uploaded_file.type == "video/mp4":
        st.write("Running detection on video...")

        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # Open video file using OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error opening video file")
        else:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_video_path = f"output_video.mp4"

            # Define the codec and create a VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # Process each frame in the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to PIL Image
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # Define the transformation
                transform = transforms.Compose(
                    [
                        transforms.Resize((640, 640)),  # Resize image to 640x640 pixels
                        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
                    ]
                )
                
                # Apply transformation to the frame
                frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

                # Run inference on the frame
                results = model(frame_tensor)

                # Draw bounding boxes on the frame
                for result in results:
                    frame_result = result.plot()

                # Convert result to BGR for OpenCV
                frame_result_bgr = cv2.cvtColor(frame_result, cv2.COLOR_RGB2BGR)

                # Write the frame to the output video
                out.write(frame_result_bgr)

            # Release resources
            cap.release()
            out.release()

            # Display the processed video
            st.video(output_video_path)

            # Optionally, provide a download link for the processed video
            st.write("Download the processed video:")
            with open(output_video_path, "rb") as video_file:
                st.download_button(
                    label="Download video",
                    data=video_file,
                    file_name="output_video.mp4",
                    mime="video/mp4",
                )
