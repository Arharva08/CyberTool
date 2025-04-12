import re
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
import cv2
import os
from moviepy.editor import VideoFileClip
import hashlib
import pefile
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import mysql.connector
from datetime import datetime
import matplotlib.pyplot as plt

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Delldell123",
    database="cybersecurity_tool"
)
cursor = db.cursor()

# Email setup
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "pawaratharva808@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "pwol cqav onio udao"  # Replace with your app password or credentials


# File to store user data
USER_DATA_FILE = "users.json"

# Malware database (signature-based approach for simplicity)
MALWARE_HASHES = {
    "d41d8cd98f00b204e9800998ecf8427e": "Test Malware 1",
    "e99a18c428cb38d5f260853678922e03": "Test Malware 2",
}

# Utility to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize user data file if not exists
if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "w") as f:
        json.dump({}, f)

def load_users():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, "r") as file:
                data = json.load(file)
                if isinstance(data, dict):
                    return data
                else:
                    print("Error: users.json does not contain a valid dictionary.")
                    return {}
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON. The file might be corrupted.")
            return {}
    return {}


# Save user data
def save_users(users):
    try:
        with open(USER_DATA_FILE, "w") as file:
            json.dump(users, file, indent=4)
    except Exception as e:
        print(f"Error: Could not save user data. {e}")



def is_valid_email(email):
    """Check if the email address is valid using a regular expression."""
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email) is not None

def log_user_action(username, action, details):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    query = "INSERT INTO user_logs (username, action, details, timestamp) VALUES (%s, %s, %s, %s)"
    cursor.execute(query, (username, action, details, timestamp))
    db.commit()

users = load_users()

# Utility: Send emails to all users
def send_email_to_all_users(subject, message):
    users = load_users()
    for user in users.keys():
        send_email(user, subject, message)

def send_email(to_email, subject, message):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(message, 'plain'))
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Deepfake Detection Model
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Initial layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Custom blocks based on checkpoint
        self.block1 = self._make_block(64, 128, 2)
        self.block2 = self._make_block(128, 128, 2)
        self.block3 = self._make_block(128, 256, 2)
        self.block4 = self._make_block(256, 512, 2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)

    def _make_block(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Load model and weights
model = CustomModel()
state_dict = torch.load('ffpp_c40.pth', map_location=torch.device('cpu'))
new_state_dict = {k[6:]: v for k, v in state_dict.items()}  # Remove 'model.' prefix
model.load_state_dict(new_state_dict, strict=False)
model.eval()


# Utility functions for deepfake detection
def deepfake_detection_ui():
    st.markdown("<h2 style='text-align: center;'>DeepFake Detection</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        # Preprocessing the image
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            preds = model(img_tensor)
            preds_prob = torch.softmax(preds, dim=1)
            fake_prob = preds_prob[0][1].item()
            result = 'Fake' if fake_prob > 0.4 else 'Not Fake'

        # Displaying the uploaded image
        st.image(img, caption='Uploaded Image', use_container_width=True)

        # Showing result
        st.write(f"### Result: {result}")

        # Displaying additional information
        if result == 'Fake':
            with st.expander("Why is this image considered Fake?"):
                st.write("The model detected patterns in the image that are consistent with deepfake images, "
                         "such as unnatural skin textures, irregularities around the eyes, or inconsistent lighting. "
                         "These patterns were learned from training on a large dataset of genuine and fake images.")
                st.write(f"Confidence Level: {fake_prob * 100:.2f}%")
        else:
            with st.expander("Why is this image considered Real?"):
                st.write("The model did not detect any signs of manipulation and the image appears to be authentic.")
                st.write(f"Confidence Level: {(1 - fake_prob) * 100:.2f}%")

        # Provide some more context
        st.markdown("---")
        st.write("### What is DeepFake?")
        st.write("""
            DeepFake technology uses artificial intelligence to create hyper-realistic fake media, such as images, 
            videos, or audio clips, that manipulate reality in ways that can deceive viewers. Detecting DeepFakes is 
            important for protecting privacy and security, and it requires sophisticated AI algorithms.
        """)

        # Optional: Visualize the image with matplotlib (for better style)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img)
        ax.axis('off')  # Hide axes for a cleaner look
        st.pyplot(fig)


def extract_frames(video_path, frames_dir, interval=5000):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Cannot open video file.")
        return

    count = 0
    success, frame = cap.read()
    while success:
        if count % interval == 0:
            frame_path = os.path.join(frames_dir, f"frame_{count}.jpg")
            cv2.imwrite(frame_path, frame)
        success, frame = cap.read()
        count += 1
    cap.release()


def detect_deepfake_in_frames(frames_dir, model, batch_size=32):
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    results = []
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i in range(0, len(frame_files), batch_size):
        batch_files = frame_files[i:i + batch_size]
        batch_tensors = [transform(Image.open(f)).unsqueeze(0) for f in batch_files]
        batch_tensor = torch.cat(batch_tensors)

        with torch.no_grad():
            preds = model(batch_tensor)
            preds_prob = torch.softmax(preds, dim=1)
            fake_probs = preds_prob[:, 1].tolist()
            results.extend(fake_probs)
    return results


def aggregate_results(results, threshold=0.5):
    avg_fake_prob = sum(results) / len(results)
    return 'Fake' if avg_fake_prob > threshold else 'Not Fake'


def deepfake_detection_video_ui():
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        frames_dir = "frames"
        os.makedirs(frames_dir, exist_ok=True)

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        extract_frames("temp_video.mp4", frames_dir)
        results = detect_deepfake_in_frames(frames_dir, model)
        final_result = aggregate_results(results)

        st.write(f"Video Deepfake Detection Result: {final_result}")

        # Display Deepfake Detection Tips
        st.markdown("<h4 style='color: #FF5722;'>Deepfake Detection Tips:</h4>", unsafe_allow_html=True)
        st.write("""
            - **Look for unnatural eye movement or blinking.**
            - **Check for inconsistent lighting and shadows in the video.**
            - **Watch for facial inconsistencies or distortions, especially around the mouth.**
            - **Pay attention to audio and visual synchronization.**
            - **Look for unusual facial expressions or lack of emotion.**
            - **Examine the background for irregularities or inconsistencies.**
            - **Verify the source of the video or image through trusted platforms.**
            - **Use deepfake detection tools or software to cross-check media authenticity.**
            - **Look for video quality degradation or pixelation around the face.**
            - **Cross-reference the content with other media for consistency.**
            """)

        # Clean up extracted frames
        for file in os.listdir(frames_dir):
            os.remove(os.path.join(frames_dir, file))
        os.rmdir(frames_dir)


# Utility: Phishing Detection
def is_suspicious_content(url):
    suspicious_patterns = [
        r"http://",  # Unsecure URLs
        r"https?://(?!.*\.gov).*",  # URLs that are not from trusted domains
        r"@.*@",  # Double "@" patterns (used in phishing URLs)
        r"free|bonus|winner|prize|claim",  # Keywords indicating scams or fraud
        r"password|account locked|urgent",  # Common phrases in phishing emails
        r"bank|credit card|debit card",  # Financial terms used in scams
        r"login|verify|update|confirm",  # Social engineering keywords
        r"\bclick\b|\blink\b",  # Prompting the user to click suspicious links
        r"\d{10}",  # Phone numbers (often included for scams)
        r"OTP|one[-\s]?time password",  # Requests for sensitive OTP information
        r"transaction failed|payment declined",  # Phrases used in payment fraud emails
        r"subscription|renewal|invoice|receipt",  # Fake billing or subscription emails
        r"\$\d+|\₹\d+",  # Suspicious monetary amounts
        r"(support|help)@(?!.*\bcompanyname\b)",  # Fake support emails
        r"limited time|act now|immediately",  # Urgency keywords to pressure users
        r"unauthorized login|suspicious activity",  # Alarmist phrases
        r"we've detected|unusual behavior",  # Pretending to detect unusual behavior
        r"(download|attachment) now",  # Prompting to download malicious files
        r"(apple|microsoft|amazon|paypal) support",  # Fake branded support
        r"secure your account|update your credentials",  # Pretend security actions
        r"congratulations|you're selected",  # Pretending user won something
        r"get rich quick|investment scheme",  # Ponzi or financial scams
        r"unsubscribe link",  # Fake unsubscribe links in emails
        r"lottery|jackpot|free gift card",  # Lottery or giveaway scams
        r"bitcoin|crypto|investment opportunity",  # Cryptocurrency scams
        r"validate your information|security check",  # Fake security procedures
        r"IRS|tax refund|government notice",  # Government impersonation
        r"payment successful|failed attempt",  # Fraudulent payment notifications
    ]
    return any(re.search(pattern, url, re.IGNORECASE) for pattern in suspicious_patterns)


def phishing_detection_ui():
    st.subheader("Phishing Detection")
    user_input = st.text_area("Enter a URL, email content, or any suspicious message:")

    st.markdown("<h4 style='color: #FF5722;'>Phishing Detection Tips:</h4>", unsafe_allow_html=True)
    st.write("""
   - **Check the sender's email for suspicious addresses.**
    - **Look for urgent or alarming messages pushing immediate action.**
    - **Examine the greeting for generic phrases like "Dear Customer."**
    - **Hover over links to verify URLs before clicking.**
    - **Look for spelling and grammar errors in the email content.**
    - **Avoid opening unexpected attachments or downloading files.**
    - **Never share sensitive info (passwords, bank details) via email.**
    - **Check website URLs for unusual domains or spelling mistakes.**
    - **Use Two-Factor Authentication (2FA) for added security.**
    - **Report suspicious emails to your email provider or the company directly.**
    """)

    # Initialize detection history for report generation
    if "phishing_detection_history" not in st.session_state:
        st.session_state.phishing_detection_history = []

    if st.button("Scan"):
        if user_input:
            if is_suspicious_content(user_input):
                st.error("Warning: This might be a phishing attempt!")

                # Log the action if the user is logged in
                if st.session_state.logged_in:
                    log_user_action(st.session_state.username, "Phishing Detection", user_input)
                    send_email_to_all_users(
                        "Suspicious URL/Message Detected",
                        f"User {st.session_state.username} detected a suspicious URL/message: {user_input}"
                    )

                # Save detection result to session state
                detection_entry = {
                    "content": user_input,
                    "result": "Suspicious",
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.session_state.phishing_detection_history.append(detection_entry)
            else:
                st.success("No suspicious elements detected.")

                # Save detection result to session state
                detection_entry = {
                    "content": user_input,
                    "result": "Clean",
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                st.session_state.phishing_detection_history.append(detection_entry)
        else:
            st.warning("Please enter a valid input.")

    # Add a Generate Report button
    if st.button("Generate Report"):
        if st.session_state.phishing_detection_history:
            generate_report(st.session_state.phishing_detection_history)
        else:
            st.warning("No detection history to generate a report from.")


def generate_report(detection_history):
    """
    Generate a detailed phishing detection report with proper encoding.
    """
    report_content = "Phishing Detection Report\n"
    report_content += "=" * 50 + "\n"
    report_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report_content += "Detection History:\n"

    for idx, entry in enumerate(detection_history, start=1):
        report_content += f"{idx}. Time: {entry['timestamp']}\n"
        report_content += f"   Input: {entry['content']}\n"
        report_content += f"   Result: {entry['result']}\n\n"

    # Save as a downloadable file
    report_path = "phishing_detection_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:  # Specify utf-8 encoding
        f.write(report_content)

    # Provide download option
    with open(report_path, "rb") as f:
        st.download_button(
            label="Download Report",
            data=f,
            file_name="Phishing_Detection_Report.txt",
            mime="text/plain"
        )

    st.success("Report generated successfully!")


# Utility: Malware Detection
def calculate_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def analyze_pe_file(file_path):
    try:
        pe = pefile.PE(file_path)
        suspicious_sections = [section.Name.decode().strip() for section in pe.sections if b".text" not in section.Name]
        st.write("Suspicious Sections Found:", suspicious_sections)
        return suspicious_sections
    except Exception as e:
        st.error(f"Error analyzing PE file: {e}")
        return []


def malware_detection_ui():
    uploaded_file = st.file_uploader("Upload a file to scan for malware", type=["exe", "dll", "pdf", "docx", "txt"])
    st.markdown("<h4 style='color: #FF5722;'>Malware Detection Tips:</h4>", unsafe_allow_html=True)
    st.write("""
    - **Check file extensions**: Be cautious of files with double extensions (e.g., `filename.jpg.exe`).
    - **Scan files with antivirus software**: Always scan unknown files before opening them.
    - **Avoid downloading files from untrusted sources**: Download files only from reputable websites.
    - **Verify file signatures**: Ensure that files come from legitimate and trusted sources.
    - **Check for unusual file sizes**: Malicious files may have abnormal or unexpected file sizes.
    - **Monitor system performance**: Sudden slowdowns or crashes can be a sign of malware.
    - **Keep your software updated**: Regular updates help protect against new malware strains.
    - **Use firewalls**: Firewalls can block harmful inbound and outbound network traffic.
    - **Avoid opening suspicious email attachments**: Never open files from unknown senders.
    - **Educate yourself and others**: Be aware of common malware techniques such as phishing, Trojans, or ransomware.
    """)

    if uploaded_file is not None:
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())

        file_hash = calculate_file_hash(temp_file_path)
        st.write(f"File Hash: {file_hash}")

        if file_hash in MALWARE_HASHES:
            st.error(f"Malware Detected: {MALWARE_HASHES[file_hash]}")
        else:
            st.success("No known malware detected.")

        if uploaded_file.name.endswith((".exe", ".dll")):
            analyze_pe_file(temp_file_path)

        os.remove(temp_file_path)



def educational_page_ui():
    """
    Function to display the educational page with useful information about the cybersecurity tool.
    """
    # Set background color and text alignment
    st.markdown("""
                                        <style>
        /* General Body */
        body {
            background-color: #f4f6f9; /* Lighter background for better contrast */
            font-family: 'Roboto', sans-serif;
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        .container {
            text-align: center;
            padding: 50px;
            margin: 0 auto;
        }

        /* Title */
        .title {
            color: #2c3e50; /* Darker blue for the title */
            font-size: 56px;
            font-weight: 800;
            margin-bottom: 30px;
            letter-spacing: 2px;
            text-transform: capitalize;
            font-family: 'Poppins', sans-serif;
        }

        /* Subtitle */
        .subtitle {
            color: #e74c3c; /* Red for emphasis */
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 20px;
            letter-spacing: 1px;
            font-family: 'Poppins', sans-serif;
        }

        /* Section Header */
        .section-header {
            color: #ffffff;
            font-size: 28px;
            font-weight: 700;
            margin-top: 50px;
            background-color: #16a085; /* Teal for section header */
            padding: 15px 20px;
            border-radius: 12px;
            text-transform: uppercase;
            font-family: 'Poppins', sans-serif;
        }

        /* Description */
        .description {
            color: #34495e; /* Slightly darker text for readability */
            font-size: 20px;
            line-height: 1.8;
            text-align: left;
            margin-bottom: 30px;
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }

        /* List Items */
        .list-items {
            color: #2980b9; /* Lively blue for list items */
            font-size: 18px;
            text-align: left;
            padding-left: 25px;
            margin-bottom: 25px;
        }

        .list-items a {
            color: #8e44ad; /* Purple for links */
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }

        .list-items a:hover {
            color: #c0392b; /* Dark red for hover effect */
        }

        /* Card Styling */
        .card {
            background-color: #ffffff;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 40px;
            max-width: 950px;
            margin-left: auto;
            margin-right: auto;
            transition: all 0.3s ease;
        }

        .card:hover {
            box-shadow: 0px 15px 30px rgba(0, 0, 0, 0.2);
            transform: translateY(-10px);
        }

        /* Button */
        .btn {
            background-color: #8e44ad; /* Purple for buttons */
            color: white;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 20px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.3s ease;
            text-align: center;
            width: 280px;
            margin-top: 40px;
            border: none;
            box-shadow: 0px 5px 10px rgba(0, 0, 0, 0.1);
        }

        .btn:hover {
            background-color: #9b59b6; /* Light purple on hover */
            transform: scale(1.05);
            box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.2);
        }

        /* Footer */
        .footer {
            font-size: 16px;
            color: #7f8c8d;
            margin-top: 60px;
            padding: 30px;
            text-align: center;
            background-color: #ecf0f1;
        }

        .footer a {
            color: #2980b9; /* Blue for footer links */
            text-decoration: none;
            font-weight: 600;
            transition: color 0.3s ease;
        }

        .footer a:hover {
            color: #1abc9c; /* Greenish blue for footer link hover */
        }

        /* Image Section */
        .image-section {
            margin-top: 40px;
            margin-bottom: 40px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
        }

        .image-caption {
            font-size: 14px;
            color: #757575; /* Light gray for captions */
            text-align: center;
            margin-top: 10px;
        }

        .image-section img {
            max-width: 100%;
            border-radius: 12px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .image-section img:hover {
            transform: scale(1.05);
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.1);
        }
    </style>
        """, unsafe_allow_html=True)

    # Title and subtitle
    st.markdown("<div class='container'><h1 class='title'>Educational Page</h1></div>", unsafe_allow_html=True)
    st.markdown("<div class='container'><h3 class='subtitle'>Welcome to the Cybersecurity Protection Tool</h3></div>", unsafe_allow_html=True)

    st.markdown("<div class='container'><h3 class='section-header'>Stay Safe Online</h3></div>", unsafe_allow_html=True)
    st.video("https://youtu.be/JpfEBQn2CjM?si=xQSmC6MXVrr7DzWY")  # Replace with a relevant video link

    # Tool Description Section with card layout
    st.markdown("<div class='container'><div class='card'><h3 class='section-header'>Tool Description</h3></div></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='container'>
            <p class='description'>
                The Cybersecurity Protection Tool is designed to detect and prevent various types of cybersecurity threats. Below are the modules available:
            </p>
            <ul class='list-items'>
                <li><b>Phishing Detection</b>: Identifies phishing attempts in emails, websites, or files.</li>
                <li><b>Malware Detection</b>: Analyzes files for malicious content.</li>
                <li><b>Deepfake Detection</b>: Detects fake media content (images and videos).</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Image Section for Modules
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/841/841795.png", caption="Phishing Detection", use_container_width=True)
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/4080/4080669.png", caption="Malware Detection", use_container_width=True)
    with col3:
        st.image("https://cdn-icons-png.flaticon.com/512/3663/3663712.png", caption="Deepfake Detection", use_container_width=True)

    # Cybersecurity Threat Examples Section
    st.markdown("<div class='container'><div class='card'><h3 class='section-header'>Real-World Cybersecurity Threats</h3></div></div>", unsafe_allow_html=True)
    st.write("""
        <div class='container'>
            <p class='description'>
                Here are a few examples of common cybersecurity threats that this tool can help mitigate:
            </p>
            <ul class='list-items'>
                <li><b>Phishing:</b> Fraudulent attempts to acquire sensitive information by pretending to be a trustworthy entity. Example: Fake bank emails.</li>
                <li><b>Malware:</b> Malicious software designed to disrupt, damage, or gain unauthorized access to systems. Example: Ransomware encrypting files and demanding payment.</li>
                <li><b>Deepfake:</b> AI-generated media content that manipulates real images or videos to deceive viewers. Example: Fake videos of politicians making false statements.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Cybersecurity Awareness Tips Section with card layout
    st.markdown("<div class='container'><div class='card'><h4 class='section-header'>Cybersecurity Awareness Tips</h4></div></div>", unsafe_allow_html=True)
    st.write("""
        <div class='container'>
            <ul class='list-items'>
                <li><b>Always be cautious of unsolicited emails</b>, especially ones that ask for personal information.</li>
                <li><b>Ensure your software is up-to-date</b> to minimize vulnerabilities.</li>
                <li><b>Use strong, unique passwords</b> for each account.</li>
                <li><b>Be aware of deepfake technology</b> and its potential for fraud.</li>
                <li><b>Enable two-factor authentication</b> wherever possible for added security.</li>
                <li><b>Back up your important data regularly</b> to protect against ransomware.</li>
                <li><b>Use a password manager</b> to keep track of complex passwords.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # How the Tool Works Section with card layout
    st.markdown("<div class='container'><div class='card'><h4 class='section-header'>How the Tool Works</h4></div></div>", unsafe_allow_html=True)
    st.write("""
        <div class='container'>
            <ul class='list-items'>
                <li><b>Phishing Detection</b>: Scans content for suspicious patterns, like fake URLs or email addresses.</li>
                <li><b>Malware Detection</b>: Checks files against a database of known malicious signatures and behaviors.</li>
                <li><b>Deepfake Detection</b>: Uses machine learning models to analyze media for inconsistencies, such as unnatural facial expressions or voice mismatches.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='container'><h4 class='section-header'>Test Your Cybersecurity Knowledge</h4></div>",
                unsafe_allow_html=True)
    quiz1 = st.radio("Which of the following is a good cybersecurity practice?",
                     ["Clicking on links in unsolicited emails", "Using strong, unique passwords",
                      "Disabling firewall"])
    if quiz1 == "Using strong, unique passwords":
        st.success("Correct! Using strong passwords is a crucial part of cybersecurity.")
    elif quiz1:
        st.error("Oops! That's not correct. Strong passwords are important for your security.")

    quiz2 = st.radio("What should you do if you suspect a phishing email?",
                     ["Ignore it", "Click the link to verify", "Report it to your IT team"])
    if quiz2 == "Report it to your IT team":
        st.success("Correct! Reporting phishing emails helps protect everyone.")
    elif quiz2:
        st.error("That's not right. Always report suspicious emails!")


    # Button for More Information
    st.markdown("<div class='container'><a href='https://www.cyberaware.gov/' target='_blank'><button class='btn'>Learn More</button></a></div>", unsafe_allow_html=True)

    # Separator line for visual appeal
    st.markdown("<div class='container'><hr style='border: 2px solid #3498db; width: 50%; margin: 30px auto;'/></div>", unsafe_allow_html=True)

    # Interactive Links for User Engagement
    st.markdown("<div class='container'><h4 class='section-header'>Useful Resources</h4></div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='container'>
            <ul class='list-items'>
                <li><a href='https://www.us-cert.cisa.gov/' target='_blank'>Cybersecurity & Infrastructure Security Agency (CISA)</a></li>
                <li><a href='https://www.csoonline.com/' target='_blank'>CSO Online - Cybersecurity News and Analysis</a></li>
                <li><a href='https://www.krebssecurity.com/' target='_blank'>Krebs on Security</a></li>
            </ul>   
        </div>
    """, unsafe_allow_html=True)

    # Footer Section
    st.markdown("<div class='footer'>This tool is designed to help you stay informed and safe in the digital world. For more information, visit <a href='https://www.cyberaware.gov/' target='_blank'>CyberAware.gov</a>.</div>", unsafe_allow_html=True)


# User Authentication UI
def login_registration_ui():
    # Fresh and modern styling
    st.markdown("""
        <style>
            /* Background styling */
            body {
                background-image: url('https://cdn.pixabay.com/photo/2021/07/04/13/29/fence-6386394_640.jpg');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                height: 90vh; /* Full viewport height */
                margin: 0; /* Removes default margin */
                display: flex;
                justify-content: center; 
                align-items: center; /* Centers the content vertically */
            }

            /* Main container styling */
            .stContainer {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
                margin-top: 20px;
                max-width: 420px;
                margin-left: auto;
                margin-right: auto;
                overflow-y: auto; /* Prevents scroll overflow inside the container */
                max-height: 90vh; /* Prevents container from exceeding the viewport height */
            }

            /* Headers styling */
            h1, h3 {
                color: #90caf9; /* Light blue for dark theme */
                text-align: center;
                font-family: 'Roboto', sans-serif;
                font-weight: bold;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.6); /* Adds depth with subtle shadow */
                margin-bottom: 20px;
            }

            .header h2 {
                color: #FF7043; /* Warm orange for contrast */
                text-align: center;
                font-family: 'Roboto', sans-serif;
                font-weight: 700; /* Heavier font weight for emphasis */
                text-transform: uppercase; /* Makes text stand out more */
                text-shadow: 0 2px 6px rgba(0, 0, 0, 0.7); /* Stronger shadow for elevation */
                margin-bottom: 15px;
                letter-spacing: 1.2px; /* Slight spacing for better readability */
                border-bottom: 2px solid #FF7043; /* Adds a divider line under the header */
                display: inline-block;
                padding-bottom: 5px;
            }

            /* Input fields styling */
            .stTextInput>div>div>input {
                background-color: #1E1E1E; /* Matches dark theme */
                border-radius: 8px;
                border: 2px solid #64B5F6;
                padding: 12px;
                font-size: 16px;
                color: #E0E0E0; /* Light text for better contrast */
                font-family: 'Roboto', sans-serif;
                transition: all 0.3s ease-in-out; /* Smooth transition */
            }

            .stTextInput>div>div>input:focus {
                border-color: #90CAF9; /* Softer blue for focus */
                outline: none;
                background-color: #2E2E2E; /* Slightly lighter for focus */
                box-shadow: 0 0 8px rgba(144, 202, 249, 0.8); /* Glow effect */
            }

            /* Buttons styling */
            .stButton>button {
                background: linear-gradient(90deg, #00796B, #48C9B0); /* Gradient effect */
                color: white;
                border-radius: 8px;
                width: 100%;
                padding: 14px; /* Increased padding for larger click area */
                font-size: 18px; /* Slightly larger font for emphasis */
                font-weight: bold;
                font-family: 'Roboto', sans-serif;
                border: none;
                cursor: pointer;
                transition: all 0.4s ease; /* Smooth transition on hover */
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Subtle shadow for elevation */
            }

            .stButton>button:hover {
                background: linear-gradient(90deg, #004D40, #1DE9B6);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4); /* Stronger shadow for hover */
                transform: scale(1.05); /* Slight zoom effect */
            }

            /* Radio buttons styling for dark theme */
.stRadio>div {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 0; /* Remove padding */
    border-radius: 0; /* Remove border radius */
    border: none; /* Remove the border */
    background-color: transparent; /* Remove the background */
    box-shadow: none; /* Remove box shadow */
}

.stRadio>div>label {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff; /* White text */
    font-family: 'Roboto', sans-serif;
    padding: 8px 15px;
    border-radius: 8px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: all 0.3s ease-in-out;
    background-color: #2e2e3f; /* Slightly lighter background */
}

.stRadio>div>label:hover {
    background-color: #424256; /* Highlight on hover */
    color: #90caf9; /* Light blue on hover */
    border: 1px solid #90caf9;
}

.stRadio>div>label input {
    margin-right: 10px;
    accent-color: #90caf9; /* Matches the blue theme for checkboxes */
}

/* Add spacing between the labels and the inputs for better alignment */
.stRadio>div>label:hover input {
    border-color: #90caf9;
}


        </style>
    """, unsafe_allow_html=True)

    # Title and introductory text
    st.markdown("<div class='header'><h2>Cybersecurity Protection Tool</h2></div>", unsafe_allow_html=True)
    st.markdown("<h1>Login or Register</h1>", unsafe_allow_html=True)

    # Load users data
    users = load_users()

    # Choose between Login and Register
    choice = st.radio("Select an option", ["Login", "Register", "Forgot Password"], index=0)

    # Login form
    if choice == "Login":
        st.markdown("<h3>Please enter your credentials to login</h3>", unsafe_allow_html=True)

        with st.container():
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            if st.button("Login"):
                if email and password:
                    if is_valid_email(email):  # Validate email format
                        if email in users:  # Check if the email is registered
                            stored_password = users[email].get("password", "")  # Retrieve the stored password
                            if stored_password == hash_password(password):  # Compare hashed passwords
                                st.success("Logged in successfully!")
                                st.session_state.logged_in = True
                                st.session_state.username = email
                            else:
                                st.error("Invalid email or password")
                        else:
                            st.error("No account found with this email. Please register.")
                    else:
                        st.error("Please enter a valid email address")
                else:
                    st.error("Please enter both email and password")

    # Registration Form
    elif choice == "Register":
        st.markdown("<h3>Create a new account</h3>", unsafe_allow_html=True)

        with st.container():
            email = st.text_input("Email", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Choose a strong password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")

            if st.button("Register"):
                if email and password and confirm_password:
                    if is_valid_email(email):  # Validate email format
                        if password == confirm_password:
                            if email not in users:
                                hashed_password = hash_password(password)
                                # Initialize the new user's data structure
                                users[email] = {
                                    "password": hashed_password,
                                    "profile": {
                                        "name": "",
                                        "contact": "",
                                        "location": "",
                                        "summary": "",
                                        "work_experience": "",
                                        "education": "",
                                        "skills": "",
                                        "hobbies": ""
                                    }
                                }
                                save_users(users)  # Save to JSON or your data storage
                                st.success("User registered successfully! You can now log in.")
                            else:
                                st.error("Email is already registered. Please use a different email.")
                        else:
                            st.error("Passwords do not match. Please try again.")
                    else:
                        st.error("Please enter a valid email address")
                else:
                    st.error("Please fill in all fields")

    # Forgot Password form
    elif choice == "Forgot Password":
        st.markdown("<h3>Forgot your password?</h3>", unsafe_allow_html=True)

        with st.container():
            email = st.text_input("Email", placeholder="Enter your registered email")

            if st.button("Send Reset Link"):
                if email:
                    if email in users:
                        reset_link = f"https://example.com/reset-password?email={email}"
                        send_email(
                            to_email=email,
                            subject="Password Reset Request",
                            message=f"Click the link to reset your password: {reset_link}"
                        )
                        st.success("Password reset link sent successfully.")
                    else:
                        st.error("Email not found. Please register first.")
                else:
                    st.error("Please enter your email address")
def profile_page():
    # Load user data
    users = load_users()
    if not isinstance(users, dict):  # Validate that users is a dictionary
        st.error("User data is corrupted! Please contact support.")
        return

    # Get logged-in user's email
    email = st.session_state.get("username")
    if not email:
        st.warning("Session expired. Please log in again.")
        st.session_state.page = "Login"  # Redirect to login page
        return

    # Ensure the email entry exists, and initialize if not present
    if email not in users or not isinstance(users[email], dict):
        users[email] = {
            "password": "",
            "profile": {
                "name": "",
                "contact": "",
                "location": "",
                "summary": "",
                "work_experience": "",
                "education": "",
                "skills": "",
                "hobbies": ""
            }
        }

    # Ensure user profile structure
    user_entry = users[email]
    user_data = user_entry.get("profile", {})
    if not isinstance(user_data, dict):
        user_data = {}
    expected_fields = ["name", "contact", "location", "summary", "work_experience", "education", "skills", "hobbies"]
    for field in expected_fields:
        if field not in user_data:
            user_data[field] = ""

    st.title("Profile Page")
    st.markdown("### Personal Details")
    if st.button("Back to Home"):
        st.session_state.page = "Home"  # Navigate back to the Home page
    name = st.text_input("Name", value=user_data.get("name", ""))
    contact = st.text_input("Contact Information", value=user_data.get("contact", ""))
    location = st.text_input("Location", value=user_data.get("location", ""))

    st.markdown("### Professional Summary")
    professional_summary = st.text_area("Summary", value=user_data.get("summary", ""))

    st.markdown("### Work Experience")
    work_experience = st.text_area("Work Experience", value=user_data.get("work_experience", ""))

    st.markdown("### Education")
    education = st.text_area("Education", value=user_data.get("education", ""))

    st.markdown("### Skills")
    skills = st.text_area("Skills", value=user_data.get("skills", ""))

    st.markdown("### Interests and Hobbies")
    hobbies = st.text_area("Interests and Hobbies", value=user_data.get("hobbies", ""))

    # Save profile details
    if st.button("Save Profile", key="save_profile"):
        # Debugging step
        print("Before updating profile:", users)

        # Ensure the user's entry exists
        if email not in users:
            users[email] = {"password": "", "profile": {}}  # Initialize structure if missing

        # Preserve the existing password
        current_password = users[email].get("password", "")

        # Update only the profile section of the user's data
        users[email]["profile"] = {
            "name": name,
            "contact": contact,
            "location": location,
            "summary": professional_summary,
            "work_experience": work_experience,
            "education": education,
            "skills": skills,
            "hobbies": hobbies,
        }

        # Re-assign the existing password to ensure it's not overwritten
        users[email]["password"] = current_password

        # Debugging step
        print("After updating profile:", users)

        # Save the updated user data
        save_users(users)
        st.success("Profile updated successfully!")


# Sidebar with Profile and Logout Buttons
# if "logged_in" in st.session_state and st.session_state.logged_in:
#     st.sidebar.write(f"**Logged in as:** {st.session_state.username}")
#
#     # Unique keys for sidebar buttons to prevent conflicts
#     if st.sidebar.button("Profile", key="profile_button", use_container_width=True):
#         profile_page()
#     if st.sidebar.button(
#             "Log out",
#             key="logout_button",
#             on_click=lambda: st.session_state.update({"logged_in": False, "username": None}),
#             use_container_width=True
#     ):
#         st.success("Logged out successfully!")
# else:
#     st.sidebar.write("Please log in to access the application.")


# Define the chatbot UI
def generate_response(user_input, context):
    """
    Generate chatbot responses based on user input and conversation context.
    """
    user_input = user_input.lower()

    if "email scam" in user_input:
        context["topic"] = "email scam"
        return "Are you asking about email scams, phishing emails, or suspicious attachments?"

    # Check if the user is asking about phishing, based on the previous topic
    elif "phishing" in user_input and context.get("topic") == "email scam":
        return "Phishing emails often try to trick you into providing sensitive information. Do you want tips to recognize them?"

    elif "phishing" in user_input:
        context["topic"] = "phishing"
        return "Phishing is a type of cyber threat. Are you asking about emails or other methods of phishing?"

    elif "scam" in user_input:
        context["topic"] = "scam"
        return "Online scams are a common threat. Would you like to learn about how to identify them?"

    else:
        context["topic"] = None
        return "I'm here to assist with cybersecurity topics. Could you clarify your question?"


def chatbot_page():
    """Enhanced Chatbot Page with Context Awareness"""
    st.markdown(
        """
        <style>
            .chat-header {
                text-align: center;
                color: #4CAF50;
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 20px;
            }
            .chat-container {
                background-color: #f7f7f7;
                border-radius: 10px;
                padding: 15px;
                max-width: 600px;
                margin: auto;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            }
            .chat-message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 8px;
                font-size: 16px;
                line-height: 1.5;
            }
            .user-message {
                text-align: right;
                background-color: #e8f5e9;
                color: #2e7d32;
                font-weight: bold;
            }
            .bot-message {
                text-align: left;
                background-color: #ede7f6;
                color: #512da8;
                font-weight: bold;
            }
            .chat-input {
                margin-top: 20px;
                text-align: center;
            }
            .back-button {
                margin-top: 20px;
                text-align: center;
            }
        </style>
        <div class="chat-header">Cybersecurity Chatbot</div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize chat history and context
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = {"topic": None}

    # Chat container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Display chat history
    for chat in st.session_state.chat_history:
        if chat["is_user"]:
            st.markdown(
                f"<div class='chat-message user-message'>{chat['text']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='chat-message bot-message'>{chat['text']}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # User input field and send button
    with st.container():
        user_input = st.text_input(
            "Type your message:",
            placeholder="Ask me about cybersecurity topics...",
            key="user_input",
        )
        if st.button("Send", key="send_message"):
            if user_input.strip():
                # Add user input to chat history
                st.session_state.chat_history.append({"is_user": True, "text": user_input})

                # Generate and add bot response with context
                bot_response = generate_response(user_input, st.session_state.chat_context)
                st.session_state.chat_history.append({"is_user": False, "text": bot_response})
            else:
                st.error("Please enter a valid message.")

    # Back to Home button
    with st.container():
        if st.button("Back to Home", key="back_to_home"):
            st.session_state.page = "Home"


def report_generation_ui():
    pass
# Main Streamlit flow
def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    if "page" not in st.session_state:
        st.session_state.page = "Home"  # Default page is "Home"

    if not st.session_state.logged_in:
        login_registration_ui()
    else:
        # Main title and header
        st.markdown("""
            <div style='text-align: center;'>
                <h2 style='color: blue; font-weight: bold;'>Cybersecurity Protection Tool</h2>
            </div>
        """, unsafe_allow_html=True)

        # Sidebar content
        st.sidebar.markdown("""
                    <style>
                        .sidebar-title {
                            color: #3498db;
                            font-size: 24px;
                            font-weight: bold;
                            padding: 10px;
                            text-align: center;
                        }
                        .sidebar-header {
                            color: #2c3e50;
                            font-size: 18px;
                            font-weight: bold;
                            padding-top: 20px;
                            padding-bottom: 10px;
                        }
                        .sidebar-button {
                            background-color: #3498db;
                            color: white;
                            font-size: 16px;
                            font-weight: bold;
                            width: 100%;
                            border-radius: 5px;
                            padding: 10px;
                            margin-top: 10px;
                            border: none;
                            cursor: pointer;
                            transition: background-color 0.3s;
                        }
                        .sidebar-button:hover {
                            background-color: #2980b9;
                        }
                        .sidebar-info {
                            color: #7f8c8d;
                            font-size: 14px;
                            margin-top: 20px;
                        }
                    </style>
                    <div class='sidebar-title'>Cybersecurity Protection Tool</div>
                """, unsafe_allow_html=True)

        # Sidebar User Info and Logout Button
        if st.session_state.logged_in:
            st.sidebar.write(f"**Logged in as:** {st.session_state.username}")
            if st.sidebar.button("Profile",use_container_width=True):
                st.session_state.page = "Profile"  # Navigate to Profile page
            if st.sidebar.button("Log out"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.page = "Home"
                st.sidebar.success("Logged out successfully!")
        else:
            st.sidebar.write("Please log in to access the application.")

        # Sidebar: Module Selection
        st.sidebar.header("Modules")
        module = st.sidebar.selectbox('Choose Module',
                                      ['Home', 'Phishing Detection', 'Malware Detection', 'Deepfake Detection',
                                       'Report Generation'])

        # Toggle Chatbot Button
        if st.sidebar.button("Chatbot"):
            st.session_state.page = "Chatbot"  # Navigate to Chatbot page

        # Handle navigation
        if st.session_state.page == "Profile":
            profile_page()
        elif st.session_state.page == "Chatbot":
            chatbot_page()
        else:
            # Show the respective module content
            if module == 'Home':
                educational_page_ui()
            elif module == 'Phishing Detection':
                phishing_detection_ui()
            elif module == 'Malware Detection':
                malware_detection_ui()
            elif module == 'Deepfake Detection':
                deepfake_detection_ui()
            elif module == 'Report Generation':
                st.write("Report Generation Page")

        # Sidebar footer
        st.sidebar.markdown("<hr>", unsafe_allow_html=True)
        st.sidebar.markdown("<div class='sidebar-info'>For support, contact us at support@cybersec.com</div>",
                            unsafe_allow_html=True)

if __name__ == "__main__":
    main()
