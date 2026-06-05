import streamlit as st
from backend import run_seo_crew

# Page Configuration
st.set_page_config(
    page_title="AI SEO Content Crew",
    page_icon="✍️",
    layout="centered"
)

# Header Section
st.title("✍️ CrewAI SEO Content Generator")
st.markdown(
    "Leverage a collaborative crew of AI agents (SEO Specialist, Writer, and Editor) "
    "to build completely optimized, production-ready blog content."
)

st.divider()

# User Input Form
with st.form(key="crew_input_form"):
    topic_input = st.text_input(
        label="Enter Blog Topic:",
        value="How to use CrewAI for Multi-Agent Systems",
        placeholder="e.g., Introduction to Quantum Computing"
    )
    
    submit_button = st.form_submit_button(label="🚀 Launch Crew")

# Execution and Output Display
if submit_button:
    if not topic_input.strip():
        st.warning("Please enter a valid topic before running the crew.")
    else:
        with st.spinner("🤖 The Crew is collaborating on your content... This may take a moment."):
            try:
                # Call backend function
                final_output = run_seo_crew(topic_input)
                
                st.success("✅ Content generation complete!")
                st.subheader("📝 Final Polished Output")
                
                # Render the raw markdown directly onto the page
                st.markdown(final_output)
                
                # Optional Download button for convenience
                st.divider()
                st.download_button(
                    label="📥 Download as Markdown",
                    data=final_output,
                    file_name=f"{topic_input.lower().replace(' ', '_')}_draft.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"An error occurred during crew execution: {e}")