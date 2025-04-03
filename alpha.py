import streamlit as st
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.gemini import GeminiModel
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import re
from load_api import Settings
import os
from groq import Groq

settings = Settings()

def sanitize_transcript(transcript: str) -> str:
    """Sanitizes sensitive information in the transcript.
    
    For credit cards: Shows first 6 and last 4 digits, redacts middle (e.g., 123456XXXXXX1234)
    Handles both fully visible and partially masked cards
    For SSNs: Shows only last 4 digits (e.g., XXX-XX-1234)
    """
    # Pattern for credit cards (matches 12-19 digit numbers with optional separators)
    # Handles both fully visible and partially masked cards
    card_pattern = r'\b((?:\d[ -]?){6})(?:(?:[Xx\- ]{1,6}|\d[ -]?){2,9})([ -]?\d{4})\b'
    
    def mask_card(match):
        first_part = re.sub(r'[^0-9]', '', match.group(1))[:6]  # Take first 6 digits
        last_part = re.sub(r'[^0-9]', '', match.group(2))[-4:]  # Take last 4 digits
        return f"{first_part}{'X' * max(0, 16 - len(first_part) - len(last_part))}{last_part}"
    
    sanitized = re.sub(card_pattern, mask_card, transcript)
    
    # Pattern for SSNs (matches XXX-XX-1234 format)
    ssn_pattern = r'\b(\d{3}|X{3})[- ]?(\d{2}|X{2})[- ]?(\d{4})\b'
    # Replace first 5 digits with X's while preserving last 4
    sanitized = re.sub(ssn_pattern, r'XXX-XX-\3', sanitized)
    
    return sanitized

model = GeminiModel('gemini-2.5-pro-exp-03-25', provider='google-gla')
agent = Agent(
    model,
    system_prompt='''You will be provided with a transcript of a customer support conversation.
Your task is to extract relevant details and generate a structured report in the following EXACT format:

Customer Support Report
---------------------------------

1. Customer Information:
- Full Name: [First Last]
- Age: [XX] (Category: [Young/Adult/Elderly])
- Locality: [If mentioned]
- Account Number: [#######]

2. Sensitive Information:
- SSN: XXX-XX-#### (Last 4: [####])
- Credit Card: ###-####-####-#### (Last 4: [####])

3. Issue Summary:
[Concise 2-3 sentence description of the issue]
- Amount in question: $XX.XX
- Date of transaction: MM/DD/YYYY

4. Resolution Details:
[Step-by-step actions taken by agent]
- Investigation status: [Initiated/Completed]
- Expected resolution time: [If mentioned]

5. Outcome:
[Clear statement of resolution status]
- [ ] Resolved during call
- [✔] Requires follow-up
- Next steps: [What customer should do]

6. Additional Insights:
[Any relevant observations or recommendations]

Notes:
- Do NOT include any introductory phrases like "Here is the report"
- Use ONLY plain text formatting (no markdown, asterisks, or special characters)
- Maintain consistent spacing between sections
- If information isn't available, write "Not specified"
- Always mask sensitive information as shown in the format'''
)

async def transcribe_audio(audio_path: str):
    """Transcribes audio using Groq's Whisper model properly."""
    client = Groq(api_key=settings.GROQ_API_KEY)
    
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(  # ✅ Run synchronously
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json"
        )
    responses = sanitize_transcript(response.text)
    return responses


async def generate_report(transcript: str):
    sanitized_transcript = sanitize_transcript(transcript)
    response = await agent.run(sanitized_transcript, model_settings={"temperature": 0.2})
    return response.data

import tempfile

def save_report_as_pdf(report_text: str):
    """Saves the structured report as a PDF file in a temporary directory."""
    if not report_text.strip():
        return None  # Prevent saving empty PDFs

    temp_dir = tempfile.gettempdir()  # Get system temp directory
    pdf_path = os.path.join(temp_dir, "customer_support_report.pdf")

    try:
        pdf_canvas = canvas.Canvas(pdf_path, pagesize=letter)
        pdf_canvas.setFont("Helvetica", 12)
        y_position = 750  

        for line in report_text.split("\n"):
            if y_position < 50:
                pdf_canvas.showPage()
                pdf_canvas.setFont("Helvetica", 12)
                y_position = 750
            pdf_canvas.drawString(50, y_position, line)
            y_position -= 20

        pdf_canvas.save()
        return pdf_path
    except Exception as e:
        st.error(f"❌ Error saving PDF: {e}")
        return None



def create_qa_agent(report_data: str):
    """Creates a chatbot for answering questions based on the report."""
    model = GroqModel(
        'llama-3.3-70b-versatile', provider=GroqProvider(api_key=settings.GROQ_API_KEY)
    )
    qa_agent = Agent(
        model,
        system_prompt=f"""
        You are a helpful assistant answering questions based on the structured report.
        Report: {report_data}
        """
    )
    return qa_agent

import asyncio
import streamlit as st
import tempfile
import base64
from io import BytesIO

st.title("📄 Customer Support Report Generator")

# --- Upload Audio ---
audio_file = st.file_uploader("🎤 Upload an audio file for transcription", type=["wav", "mp3", "m4a"])


if audio_file and st.button("📝 Transcribe and Generate Report"):
    # Reset session state for a clean UI
    st.session_state.clear()

    with st.spinner("Transcribing..."):
        temp_audio_path = f"temp_audio.{audio_file.name.split('.')[-1]}"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        # Use asyncio.run() in synchronous Streamlit context
        transcript = asyncio.run(transcribe_audio(temp_audio_path))
        st.session_state["transcript"] = transcript
        st.text_area("📝 Transcribed Text:", transcript, height=200)

    if transcript:
        with st.spinner("Generating report..."):
            report_result = asyncio.run(generate_report(transcript))
            st.session_state["report"] = report_result
            st.success("✅ Report Generated Successfully!")
            st.text_area("📑 Generated Report:", report_result, height=300)

# --- Paste Text Option ---
transcript_input = st.text_area("📝 Or paste a customer support transcript:")

if st.button("🚀 Generate Report"):
    final_transcript = transcript_input.strip() or st.session_state.get("transcript", "")
    if final_transcript:
        with st.spinner("Generating report..."):
            report_result = asyncio.run(generate_report(final_transcript))
            st.session_state["report"] = report_result
            st.success("✅ Report Generated Successfully!")
            st.text_area("📑 Generated Report:", report_result, height=300)
    else:
        st.error("❌ Please provide a transcript!")

# --- PDF Generation & Download Fix ---
def save_report_as_pdf(report_text):
    """Generates a PDF in memory and returns bytes for downloading."""
    if not report_text.strip():
        return None

    pdf_buffer = BytesIO()
    pdf_canvas = canvas.Canvas(pdf_buffer, pagesize=letter)
    pdf_canvas.setFont("Helvetica", 12)
    y_position = 750  

    for line in report_text.split("\n"):
        if y_position < 50:
            pdf_canvas.showPage()
            pdf_canvas.setFont("Helvetica", 12)
            y_position = 750
        pdf_canvas.drawString(50, y_position, line)
        y_position -= 20

    pdf_canvas.save()
    return pdf_buffer.getvalue()  # Return PDF as bytes

if "report" in st.session_state and st.session_state["report"]:
    pdf_bytes = save_report_as_pdf(st.session_state["report"])
    
    if pdf_bytes:
        st.download_button(
            "📥 Download Report as PDF",
            pdf_bytes,
            "customer_support_report.pdf",
            mime="application/pdf"
        )

# --- Display PDF Inline ---
def display_pdf(pdf_bytes):
    """Displays the generated PDF in Streamlit."""
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

if "report" in st.session_state and st.session_state["report"]:
    st.header("📑 Preview Report")
    display_pdf(pdf_bytes)

# --- Chat with the Report ---
if "report" in st.session_state and st.session_state["report"]:
    st.header("💬 Chat with Your Report")
    qa_agent = create_qa_agent(st.session_state["report"])
    
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_input = st.chat_input("Ask something about the report...")

    if user_input:
        st.session_state.setdefault("messages", []).append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = asyncio.run(qa_agent.run(user_input, model_settings={"temperature": 0.2}))
                bot_reply = response.data
                st.markdown(bot_reply)
        
        st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
        

