import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.transforms as transforms

# Force the use of CPU
device = 'cpu'
st.write(f"Using device: {device}")

# Load the trained YOLOv8 model
model = YOLO("ppe.pt")  # Replace with the path to your ppe.pt file
model.to(device)

st.title("YOLOv8 Object Detection with Streamlit")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image or a video...", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    # Load image with PIL
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Running detection...")
    
    # Define a transformation to resize the image and convert it to a PyTorch tensor
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize image to 640x640 pixels
        transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
    ])
    
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
        st.image(result_img_pil, caption=f"Detected Image {i+1}", use_column_width=True)
