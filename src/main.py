import streamlit as st
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.gemini import GeminiModel
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from config import Settings
import os
from groq import Groq
from typing import List
import httpx
import tempfile
import base64
from io import BytesIO
from typing import List, Tuple, Optional


settings = Settings()

WHISPER_URL = "http://guest1.indominuslabs.in/transcribe-batch"
SANITIZE_URL = "http://guest1.indominuslabs.in/sanitize"

TIMEOUT_CONFIG = httpx.Timeout(300.0, connect=60.0)


# Send multiple audio files to FastAPI server for transcription.
async def transcribe_multiple_audio(files_data: List[bytes]) -> List[str]:

    files = [
        ("files", (f"audio_{i}.wav", file_bytes, "audio/wav"))
        for i, file_bytes in enumerate(files_data)
    ]

    if not files:
        return []

    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
        try:
            st.info(f"Sending {len(files)} files for transcription...")
            response = await client.post(
                settings.WHISPER_URL,
                files=files # Send the list of files
            )

            response.raise_for_status()

            data = response.json() # The endpoint should return a list of transcriptions

            if "transcriptions" in data and isinstance(data["transcriptions"], list):
                 st.success(f"Received {len(data['transcriptions'])} transcriptions.")
                 return data["transcriptions"]
            else:
                st.error(f"Unexpected response format from transcription server: {data}")
                return [f"Error: Unexpected response format from transcription server." for _ in files_data]

        except httpx.TimeoutException as exc:
            st.error(f"Transcription request timed out: {exc}")
            return [f"Error: Transcription request timed out." for _ in files_data]
        except httpx.RequestError as exc:
            st.error(f"Transcription request error: {exc}")
            return [f"Error: Transcription request error: {exc}" for _ in files_data]
        except Exception as exc:
            st.error(f"An unexpected error occurred during transcription request: {exc}")
            return [f"Error: An unexpected error occurred during transcription." for _ in files_data]

# Processes Transcripts 1 by 1, Could work on making this parallel.
async def sanitize_transcript(transcript: str) -> str:

    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
        try:
            # Safety Check to Prevent Error String to be passed.
            if transcript.startswith("Error:"):
                return transcript # Pass through the error

            response = await client.post(settings.SANITIZE_URL, json={"transcript": transcript})
            response.raise_for_status()
            data = response.json()
            if "sanitized" in data:
                 return data["sanitized"]
            else:
                 st.warning(f"Unexpected response format from sanitization server: {data}")
                 return "Error: Unexpected response format from sanitization server."

        except httpx.TimeoutException as exc:
            st.warning(f"Sanitization request timed out: {exc}")
            return "Error: Sanitization request timed out."
        except httpx.RequestError as exc:
            st.warning(f"Sanitization request error: {exc}")
            return f"Error: Sanitization request error: {exc}"
        except Exception as exc:
            st.warning(f"An unexpected error occurred during sanitization: {exc}")
            return "Error: An unexpected error occurred during sanitization."

# Generate a structured report that infers from the transcript
async def generate_report(transcript: str) -> str:
    try:
         # Only generate if the transcript doesn't already indicate an error
         if transcript.startswith("Error:"):
            return "Error: Cannot generate report from an error message."

         response = await agent.run(transcript, model_settings={"temperature": 0.2})

         # Addressing response might change in the future as it has recently changed
         report_data = response.output

         return report_data
     
    except Exception as e:
        st.error(f"Error during report generation: {e}")
        return f"Error: Could not generate report. Details: {e}"

# Processing Multiple Audio Files in Parallel

async def process_multiple_audio(files_data: List[bytes]) -> Tuple[List[str], List[str], List[str]]:

    if not files_data:
        return [], [], []

    transcripts = await transcribe_multiple_audio(files_data)

    sanitized_list = []
    reports_list = []

    total = len(transcripts)

    for i, transcript in enumerate(transcripts):
        st.info(f"Processing transcript {i+1}/{total}...")
        # Check if transcription itself failed
        if transcript.startswith("Error:"):
            sanitized = transcript # Pass through error
            report = "Error: Cannot process/report due to transcription error."
        else:
            with st.spinner(f"Sanitizing transcript {i+1}/{total}..."):
                 sanitized = await sanitize_transcript(transcript)
            with st.spinner(f"Generating report for transcript {i+1}/{total}..."):
                 report = await generate_report(sanitized) # Use original transcript for report

        sanitized_list.append(sanitized)
        reports_list.append(report)
        st.success(f"Completed processing for transcript {i+1}/{total}.")

    return transcripts, sanitized_list, reports_list

async def process_audio(file: bytes):
    transcript_result = await transcribe_multiple_audio(file)
    # Check if transcription returned an error string
    if isinstance(transcript_result, str) and transcript_result.startswith("Error:"):
        return transcript_result, "Error: Could not sanitize due to transcription failure."

    sanitized_result = await sanitize_transcript(transcript_result)
    # Check if sanitization returned an error string
    if isinstance(sanitized_result, str) and sanitized_result.startswith("Error:"):
         return transcript_result, sanitized_result 

    return transcript_result, sanitized_result

# Initialize Gemini model and agent
model = GroqModel(
        'llama-3.3-70b-versatile', provider=GroqProvider(api_key=settings.GROQ_API_KEY)
    )
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

def save_report_as_pdf(report_text: str):
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
    return pdf_buffer.getvalue()

def display_pdf(pdf_bytes):
    """Displays the generated PDF in Streamlit."""
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

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

st.title("📄 Customer Support Report Generator")


uploaded_files = st.file_uploader(
    "🎤 Upload audio files (up to 3) for transcription",
    type=["wav", "mp3", "m4a"],
    accept_multiple_files=True
)
if uploaded_files:
    # Limit to 3 files
    files_to_process = uploaded_files[:3]
    if len(uploaded_files) > 3:
        st.warning(f"Processing the first 3 files out of {len(uploaded_files)} selected.")

    if st.button(f"📝 Process {len(files_to_process)} Audio File(s)"):
        st.session_state.clear() # Clear previous results

        files_bytes_list = [file.getvalue() for file in files_to_process]
        original_filenames = [file.name for file in files_to_process] # Store original names

        with st.spinner(f"Processing {len(files_bytes_list)} audio file(s)... Please wait."):
            # Run the batch processing function
            transcripts, sanitized_texts, reports = asyncio.run(process_multiple_audio(files_bytes_list))

            # Store results in session state as lists
            st.session_state["original_filenames"] = original_filenames
            st.session_state["transcripts"] = transcripts
            st.session_state["sanitized_texts"] = sanitized_texts
            st.session_state["reports"] = reports
            st.success("Batch processing complete!")

# --- Text Input Section (Modified for Single Transcript) ---

st.subheader("📝 Or Process a Single Transcript")
transcript_input = st.text_area("Paste a customer support transcript here:")

if st.button("🚀 Generate Report for Pasted Text"):
    final_transcript = transcript_input.strip()
    if final_transcript:

        st.session_state.pop("original_filenames", None)
        st.session_state.pop("transcripts", None)
        st.session_state.pop("sanitized_texts", None)
        st.session_state.pop("reports", None)

        with st.spinner("Generating report..."):

            report_result = asyncio.run(generate_report(final_transcript))
            st.session_state["single_report"] = report_result
            st.session_state["single_transcript"] = final_transcript
    else:
        st.error("❌ Please provide a transcript!")


# --- Display Results ---

if "transcripts" in st.session_state and st.session_state["transcripts"]:
    st.header("📊 Batch Processing Results")
    num_results = len(st.session_state["transcripts"])

    for i in range(num_results):
        filename = st.session_state["original_filenames"][i]
        transcript = st.session_state["transcripts"][i]
        sanitized = st.session_state["sanitized_texts"][i]
        report = st.session_state["reports"][i]

        with st.expander(f"Results for File {i+1}: {filename}", expanded=(i==0)): # Expand first one
            st.subheader("📝 Transcription & Sanitization")
            col1, col2 = st.columns(2)
            with col1:
                st.text_area(f"Original Transcript {i+1}", transcript, height=200, key=f"orig_{i}")
            with col2:
                st.text_area(f"Sanitized Transcript {i+1}", sanitized, height=200, key=f"san_{i}")

            st.subheader(f"📑 Generated Report {i+1}")
            st.text_area(f"Report Preview {i+1}", report, height=300, key=f"rep_{i}")

            # PDF Handling for each report
            pdf_bytes = save_report_as_pdf(report) # Your existing function
            if pdf_bytes and not report.startswith("Error:"):
                 st.download_button(
                    f"📥 Download Report {i+1} as PDF",
                    pdf_bytes,
                    f"report_{filename}.pdf",
                    mime="application/pdf",
                    key=f"pdf_dl_{i}"
                 )
            elif report.startswith("Error:"):
                 st.warning("Cannot generate PDF for report containing an error.")


# Display Single Report Result if it exists
elif "single_report" in st.session_state:
     st.header("📑 Report from Pasted Text")
     st.text_area("Original Transcript", st.session_state.get("single_transcript", ""), height=200)
     st.text_area("Generated Report", st.session_state["single_report"], height=300)
     pdf_bytes = save_report_as_pdf(st.session_state["single_report"])
     if pdf_bytes and not st.session_state["single_report"].startswith("Error:"):
        st.download_button(
            "📥 Download Report as PDF",
            pdf_bytes,
            "pasted_text_report.pdf",
            mime="application/pdf"
        )
        st.subheader("📄 PDF Preview")
        display_pdf(pdf_bytes)
     elif st.session_state["single_report"].startswith("Error:"):
         st.warning("Cannot generate PDF for report containing an error.")


# --- Chat Interface ---
# Decide how chat should work with multiple reports.
# Option 1: Chat only with the first report generated in a batch.
# Option 2: Let user select which report to chat with.
# Option 3: Disable chat when multiple reports are present.
# Simple approach: Chat with the first report if available, or the single report.



report_for_chat = None
if "reports" in st.session_state and st.session_state["reports"]:
    # Check if the first report is valid
    if not st.session_state["reports"][0].startswith("Error:"):
        report_for_chat = st.session_state["reports"][0]
        st.info("Chatting with the report from the first processed file.")
elif "single_report" in st.session_state:
     if not st.session_state["single_report"].startswith("Error:"):
        report_for_chat = st.session_state["single_report"]
        st.info("Chatting with the report generated from pasted text.")

if report_for_chat:
    st.header("💬 Chat with Your Report")

    qa_agent = create_qa_agent(report_for_chat) # Your existing function

    # Initialize chat history specifically for the current report context
    # Use a key related to the report to reset history when the report changes

    chat_history_key = f"messages_{hash(report_for_chat)}"
    if chat_history_key not in st.session_state:
         st.session_state[chat_history_key] = []

    # Display existing messages

    for message in st.session_state[chat_history_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input

    if prompt := st.chat_input("Ask something about the report..."):
        st.session_state[chat_history_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(qa_agent.run(prompt, model_settings={"temperature": 0.2}))
                    bot_reply = getattr(response, 'data', str(response))
                    st.markdown(bot_reply)
                    st.session_state[chat_history_key].append({"role": "assistant", "content": bot_reply})
                except Exception as e:
                    st.error(f"Error getting response from chat agent: {e}")
                    st.session_state[chat_history_key].append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})
else:
     st.info("Upload files or paste text and generate a report to enable chat.")