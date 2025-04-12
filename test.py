import streamlit as st

# Define your page rendering functions
def profile_page():
    st.write("Profile Page")
    # Add your profile content here

def chatbot_ui():
    st.write("Chatbot UI")
    # Add your chatbot content here

def educational_page_ui():
    st.write("Educational Page UI")
    # Add your educational content here

def login_registration_ui():
    st.write("Login/Registration UI")
    # Add your login form here

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

    if "page" not in st.session_state:
        st.session_state.page = "Home"  # Default page is "Home"

    if "show_chatbot" not in st.session_state:
        st.session_state.show_chatbot = False  # Initialize chatbot visibility state

    if not st.session_state.logged_in:
        login_registration_ui()
    else:
        # Main title and header
        st.markdown("""
            <div style='text-align: center;'>
                <h2 style='color: blue; font-weight: bold;'>Cybersecurity Protection Tool</h2>
            </div>
        """, unsafe_allow_html=True)

        # Sidebar content with improvements
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
        if "logged_in" in st.session_state and st.session_state.logged_in:
            st.sidebar.write(f"**Logged in as:** {st.session_state.username}")
            if st.sidebar.button("Profile", use_container_width=True):
                st.session_state.page = "Profile"  # Navigate to the Profile page
            if st.sidebar.button("Log out"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.page = "Home"  # Go back to Home after logout
                st.sidebar.success("Logged out successfully!")
        else:
            st.sidebar.write("Please log in to access the application.")

        # Sidebar: Module Selection
        st.sidebar.header("Modules")
        module = st.sidebar.selectbox('Choose Module',
                                      ['Home', 'Phishing Detection', 'Malware Detection', 'Deepfake Detection',
                                       'Report Generation'])

        # Handle navigation
        if st.session_state.page == "Profile":
            profile_page()  # Display profile page
        else:
            # Show the respective module content
            if module == 'Home':
                educational_page_ui()
            elif module == 'Phishing Detection':
                st.write("Phishing Detection Page")
            elif module == 'Malware Detection':
                st.write("Malware Detection Page")
            elif module == 'Deepfake Detection':
                st.write("Deepfake Detection Page")
            elif module == 'Report Generation':
                st.write("Report Generation Page")

            # Toggle Chatbot Button
            if st.sidebar.button("Toggle Chatbot"):
                st.session_state.show_chatbot = not st.session_state.show_chatbot

            # Conditionally render the chatbot inside sidebar
            if st.session_state.show_chatbot:
                chatbot_ui()

        # Footer in the sidebar for additional info
        st.sidebar.markdown("<hr style='border-color: #3498db;'>", unsafe_allow_html=True)
        st.sidebar.markdown("<div class='sidebar-info'>For support, contact us at support@cybersec.com</div>",
                            unsafe_allow_html=True)


if __name__ == "__main__":
    main()
