import streamlit as st
import os
import tempfile
import json
from pathlib import Path
import base64
from PIL import Image
import pandas as pd

# Import your main functions
try:
    from main import run_flow, get_file_data, docState, ocr_tool, pdf_parser_tool, csv_parser_tool
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="PHI Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sample-file-card {
        border: 2px solid #262730;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        background-color: #0e1117;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .sample-file-card:hover {
        border-color: #1f77b4;
        background-color: #1a1a2e;
    }
    
    .sample-file-card.selected {
        border-color: #1f77b4;
        background-color: #1a1a2e;
        box-shadow: 0 0 10px rgba(31, 119, 180, 0.5);
    }
    
    .file-preview {
        max-height: 200px;
        object-fit: cover;
        border-radius: 5px;
    }
    
    .error-message {
        color: #ff6b6b;
        background-color: #2d1b1b;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff6b6b;
    }
    
    .success-message {
        color: #51cf66;
        background-color: #1b2d1b;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #51cf66;
    }
    
    .phi-highlight {
        background-color: #ff6b6b;
        color: white;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }
    
    .phi-stats {
        background-color: #262730;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'phi_results' not in st.session_state:
    st.session_state.phi_results = None
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

def create_sample_files():
    """Create sample files for demonstration"""
    sample_files = {
        "Medical Document (PDF)": {
            "filename": "sample_files/sample_doc.pdf",
            "description": "Sample medical document with patient information",
            "type": "pdf"
        },
        "Patient Data (CSV)": {
            "filename": "sample_files/sample_sheet.csv",
            "description": "Sample CSV with patient records",
            "type": "csv"
        },
        "Medical Note (PNG)": {
            "filename": "sample_files/sample_note.png",
            "description": "Sample medical note image",
            "type": "png"
        }
    }
    return sample_files

def validate_file_type(file):
    """Validate uploaded file type"""
    if file is not None:
        file_extension = file.name.split('.')[-1].lower()
        return file_extension in ['png', 'pdf', 'csv']
    return False

def get_file_preview(file_path, file_type):
    """Generate preview for different file types"""
    if file_type == "png":
        try:
            img = Image.open(file_path)
            return img
        except:
            return None
    elif file_type == "pdf":
        return "📄 PDF Document"
    elif file_type == "csv":
        return "📊 CSV Data"
    return None

def highlight_phi_instances(text, phi_instances):
    """Highlight PHI instances in the text by finding values in the original text"""
    if not phi_instances or not isinstance(phi_instances, list):
        return text
    
    highlighted_text = text
    
    # Keep track of offset due to HTML insertions
    offset = 0
    
    # Process instances and find their positions in the text
    instances_with_positions = []
    
    for instance in phi_instances:
        if 'value' in instance and instance['value']:
            value = str(instance['value']).strip()
            phi_type = instance.get('PHI type', instance.get('type', 'Unknown'))
            
            # Find all occurrences of this value in the text
            search_start = 0
            while True:
                pos = text.find(value, search_start)
                if pos == -1:
                    break
                
                # Check if this position is already covered by another instance
                overlapping = False
                for existing in instances_with_positions:
                    if not (pos >= existing['end'] or pos + len(value) <= existing['start']):
                        overlapping = True
                        break
                
                if not overlapping:
                    instances_with_positions.append({
                        'start': pos,
                        'end': pos + len(value),
                        'value': value,
                        'phi_type': phi_type,
                        'original_instance': instance
                    })
                    break  # Only highlight first occurrence to avoid duplicates
                
                search_start = pos + 1
    
    # Sort by start position in descending order to avoid index shifting during replacement
    instances_with_positions.sort(key=lambda x: x['start'], reverse=True)
    
    # Apply highlighting
    for instance in instances_with_positions:
        start = instance['start']
        end = instance['end']
        value = instance['value']
        phi_type = instance['phi_type']
        
        # Create tooltip with additional information
        tooltip_info = f"PHI Type: {phi_type}"
        if 'PHI risk rationale' in instance['original_instance']:
            tooltip_info += f"&#10;Risk: {instance['original_instance']['PHI risk rationale']}"
        
        # Create highlighted span with tooltip
        highlighted_value = f'<span class="phi-highlight" title="{tooltip_info}">{value}</span>'
        
        # Replace in text
        highlighted_text = highlighted_text[:start] + highlighted_value + highlighted_text[end:]
    
    return highlighted_text

def display_phi_statistics(phi_instances):
    """Display PHI statistics"""
    if not phi_instances:
        return
    
    phi_types = {}
    for instance in phi_instances:
        phi_type = instance.get('PHI type', instance.get('type', 'Unknown'))
        phi_types[phi_type] = phi_types.get(phi_type, 0) + 1
    
    st.markdown('<div class="phi-stats">', unsafe_allow_html=True)
    st.subheader("📊 PHI Detection Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total PHI Instances", len(phi_instances))
    with col2:
        st.metric("Unique PHI Types", len(phi_types))
    
    st.write("**PHI Types Found:**")
    for phi_type, count in phi_types.items():
        st.write(f"• {phi_type}: {count}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main App
def main():
    st.markdown('<h1 class="main-header">🔍 PHIlter: A PHI Detection System</h1>', unsafe_allow_html=True)
    st.markdown("**Detect Personal Health Information in documents using LLMs**")
    
    # Sidebar for exclusion filters
    with st.sidebar:
        st.header("⚙️ Settings")
        exclude_options = [
            "Names",
            "Dates (except year), ages >89",
            "Telephone numbers",
            "Email addresses",
            "Social Security numbers",
            "Medical record numbers",
            "Geographic subdivisions < state"
        ]
        exclude_filter = st.multiselect(
            "Exclude PHI types:",
            exclude_options,
            help="Select PHI types to exclude from detection"
        )
    
    # File selection section
    st.header("📁 Select Input Document")
    
    # Create two tabs for sample files and upload
    tab1, tab2 = st.tabs(["📚 Sample Files", "📤 Upload File"])
    
    with tab1:
        st.write("Choose from sample documents:")
        sample_files = create_sample_files()
        
        # Display sample files in columns
        cols = st.columns(3)
        for idx, (name, info) in enumerate(sample_files.items()):
            with cols[idx]:
                # Create card for each sample file
                card_selected = st.session_state.selected_file == info['filename']
                
                if st.button(f"📋 {name}", key=f"sample_{idx}", use_container_width=True):
                    st.session_state.selected_file = info['filename']
                    st.session_state.uploaded_file = None
                    st.session_state.analysis_complete = False
                    st.rerun()
                
                st.write(f"**Type:** {info['type'].upper()}")
                st.write(f"**Description:** {info['description']}")
                
                if card_selected:
                    st.success("✅ Selected")
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['png', 'pdf', 'csv'],
            help="Upload a PNG image, PDF document, or CSV file"
        )
        
        if uploaded_file is not None:
            if validate_file_type(uploaded_file):
                st.session_state.uploaded_file = uploaded_file
                st.session_state.selected_file = None
                st.session_state.analysis_complete = False
                st.success(f"✅ File uploaded: {uploaded_file.name}")
            else:
                st.error("❌ Invalid file type. Please upload a PNG, PDF, or CSV file.")
    
    # Show current selection
    current_file = None
    if st.session_state.selected_file:
        current_file = st.session_state.selected_file
        st.info(f"📄 Selected: {current_file}")
    elif st.session_state.uploaded_file:
        current_file = st.session_state.uploaded_file.name
        st.info(f"📄 Selected: {current_file}")
    
    # Run analysis button
    st.header("🚀 Run Analysis")
    
    if current_file:
        if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing document for PHI instances..."):
                try:
                    # Prepare file for analysis
                    if st.session_state.uploaded_file:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{st.session_state.uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(st.session_state.uploaded_file.getvalue())
                            file_path = tmp_file.name
                    else:
                        # Use sample file
                        file_path = st.session_state.selected_file
                    
                    # Run PHI detection
                    original_text, phi_instances = run_flow(
                        file_path=file_path,
                        exclude_filter=exclude_filter
                    )
                    
                    st.session_state.phi_results = phi_instances
                    st.session_state.original_text = original_text
                    st.session_state.analysis_complete = True
                    
                    # Clean up temporary file
                    if st.session_state.uploaded_file and os.path.exists(file_path):
                        os.unlink(file_path)
                    
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ <strong>Error during analysis:</strong> {str(e)}<br><br>Please try again with a different file or check your document format.</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please select a sample file or upload your own document first.")
    
    # Display results
    if st.session_state.analysis_complete and st.session_state.phi_results is not None:
        st.header("📊 Analysis Results")
        
        # Display statistics
        display_phi_statistics(st.session_state.phi_results)
        
        # Display highlighted text
        st.subheader("📝 Document Text with PHI Highlighted")
        
        if st.session_state.phi_results:
            highlighted_text = highlight_phi_instances(st.session_state.original_text, st.session_state.phi_results)
            st.markdown(f'<div style="background-color: #262730; padding: 1rem; border-radius: 10px; line-height: 1.6;">{highlighted_text}</div>', unsafe_allow_html=True)
        else:
            st.info("✅ No PHI instances detected in the document.")
            st.markdown(f'<div style="background-color: #262730; padding: 1rem; border-radius: 10px; line-height: 1.6;">{st.session_state.original_text}</div>', unsafe_allow_html=True)
        
        # Display detailed results
        st.subheader("📋 Detailed PHI Instances")
        
        if st.session_state.phi_results:
            for idx, instance in enumerate(st.session_state.phi_results, 1):
                with st.expander(f"PHI Instance {idx}: {instance.get('value', 'N/A')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Type:** {instance.get('type', 'N/A')}")
                        st.write(f"**PHI Type:** {instance.get('PHI type', 'N/A')}")
                        st.write(f"**Value:** {instance.get('value', 'N/A')}")
                    with col2:
                        st.write(f"**Start Position:** {instance.get('start', 'N/A')}")
                        st.write(f"**End Position:** {instance.get('end', 'N/A')}")
                    
                    if 'PHI risk rationale' in instance:
                        st.write(f"**Risk Rationale:** {instance['PHI risk rationale']}")
                    
                    st.json(instance)
        else:
            st.info("✅ No PHI instances detected in the document.")

if __name__ == "__main__":
    main()