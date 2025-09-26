"""
Modern Document Layout Analysis Web Interface
============================================

A Streamlit-based web interface for document layout analysis with:
- Document upload and analysis
- Multiple model selection
- Real-time visualization
- Results export
- Analysis history

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import pandas as pd
import json
import io
import base64
from pathlib import Path
import tempfile
import time
from typing import Dict, List

# Import our modern analyzer
from modern_layout_analyzer import ModernDocumentLayoutAnalyzer

# Page configuration
st.set_page_config(
    page_title="Document Layout Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_analyzer():
    """Initialize the document analyzer with caching"""
    return ModernDocumentLayoutAnalyzer()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📄 Document Layout Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize analyzer
    analyzer = initialize_analyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = ['layoutparser']
        if hasattr(analyzer, 'models') and 'yolo' in analyzer.models:
            available_models.append('yolo')
        
        selected_model = st.selectbox(
            "Select Analysis Model",
            available_models,
            help="Choose the model for document layout analysis"
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        extract_text = st.checkbox("Extract Text (OCR)", value=True)
        save_to_db = st.checkbox("Save to Database", value=True)
        
        # Processing options
        st.subheader("Processing")
        batch_mode = st.checkbox("Batch Processing Mode", value=False)
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        try:
            history_df = analyzer.get_analysis_history()
            if not history_df.empty:
                st.metric("Total Analyses", len(history_df))
                st.metric("Avg Processing Time", f"{history_df['processing_time'].mean():.2f}s")
            else:
                st.info("No analyses yet")
        except Exception as e:
            st.warning("Could not load statistics")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze Document", "📊 Analysis History", "📈 Batch Processing", "⚙️ Settings"])
    
    with tab1:
        st.header("Single Document Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a document image for layout analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📄 Uploaded Document")
                st.image(uploaded_file, use_column_width=True)
                
                # File info
                file_info = {
                    "Name": uploaded_file.name,
                    "Size": f"{uploaded_file.size / 1024:.1f} KB",
                    "Type": uploaded_file.type
                }
                
                for key, value in file_info.items():
                    st.text(f"{key}: {value}")
            
            with col2:
                st.subheader("🚀 Analysis Controls")
                
                if st.button("Analyze Document", type="primary"):
                    with st.spinner("Analyzing document..."):
                        try:
                            # Save uploaded file temporarily
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            # Perform analysis
                            start_time = time.time()
                            results = analyzer.analyze_document(
                                tmp_path,
                                model_name=selected_model,
                                extract_text=extract_text,
                                save_results=save_to_db
                            )
                            processing_time = time.time() - start_time
                            
                            # Store results in session state
                            st.session_state['analysis_results'] = results
                            st.session_state['analysis_image_path'] = tmp_path
                            
                            st.success("Analysis completed successfully!")
                            
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
            
            # Display results if available
            if 'analysis_results' in st.session_state:
                results = st.session_state['analysis_results']
                
                st.markdown("---")
                st.header("📋 Analysis Results")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Processing Time", f"{results['processing_time']:.2f}s")
                with col2:
                    st.metric("Blocks Detected", len(results['layout_blocks']))
                with col3:
                    st.metric("Text Length", f"{len(results['ocr_text'])} chars")
                with col4:
                    st.metric("Model Used", results['model_used'])
                
                # Visualization
                st.subheader("🎨 Layout Visualization")
                
                if st.button("Generate Visualization"):
                    try:
                        # Create visualization
                        analyzer.visualize_results(
                            st.session_state['analysis_image_path'],
                            results,
                            save_path="temp_visualization.png"
                        )
                        
                        # Display the saved visualization
                        if Path("temp_visualization.png").exists():
                            st.image("temp_visualization.png", use_column_width=True)
                            
                    except Exception as e:
                        st.error(f"Visualization failed: {str(e)}")
                
                # Detailed results
                st.subheader("📊 Detailed Results")
                
                # Layout blocks table
                if results['layout_blocks']:
                    blocks_df = pd.DataFrame(results['layout_blocks'])
                    st.dataframe(blocks_df, use_container_width=True)
                
                # OCR text
                if results['ocr_text']:
                    st.subheader("📝 Extracted Text")
                    st.text_area("OCR Results", results['ocr_text'], height=200)
                
                # Export options
                st.subheader("💾 Export Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Export as JSON"):
                        json_data = analyzer.export_results(results, 'json')
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"analysis_{results['filename']}.json",
                            mime="application/json"
                        )
                
                with col2:
                    if st.button("Export as CSV"):
                        csv_data = analyzer.export_results(results, 'csv')
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name=f"analysis_{results['filename']}.csv",
                            mime="text/csv"
                        )
    
    with tab2:
        st.header("📊 Analysis History")
        
        try:
            history_df = analyzer.get_analysis_history()
            
            if not history_df.empty:
                # Summary statistics
                st.subheader("Summary Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Analyses", len(history_df))
                with col2:
                    st.metric("Average Processing Time", f"{history_df['processing_time'].mean():.2f}s")
                with col3:
                    st.metric("Most Used Model", history_df['model_used'].mode().iloc[0] if not history_df.empty else "N/A")
                
                # Data table
                st.subheader("Analysis Records")
                st.dataframe(history_df, use_container_width=True)
                
                # Charts
                st.subheader("📈 Charts")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Processing time distribution
                    st.bar_chart(history_df.set_index('filename')['processing_time'])
                
                with col2:
                    # Model usage
                    model_counts = history_df['model_used'].value_counts()
                    st.bar_chart(model_counts)
                
            else:
                st.info("No analysis history available. Upload and analyze some documents first!")
                
        except Exception as e:
            st.error(f"Could not load analysis history: {str(e)}")
    
    with tab3:
        st.header("📈 Batch Processing")
        
        st.info("Upload multiple documents for batch analysis")
        
        uploaded_files = st.file_uploader(
            "Upload Multiple Documents",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Select multiple document images for batch processing"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} files")
            
            if st.button("Start Batch Analysis", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing: {uploaded_file.name}")
                        
                        # Save file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Analyze
                        result = analyzer.analyze_document(
                            tmp_path,
                            model_name=selected_model,
                            extract_text=extract_text,
                            save_results=save_to_db
                        )
                        
                        batch_results.append(result)
                        
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Batch processing completed!")
                
                # Display batch results summary
                if batch_results:
                    st.subheader("Batch Results Summary")
                    
                    summary_data = []
                    for result in batch_results:
                        summary_data.append({
                            'Filename': result['filename'],
                            'Blocks': len(result['layout_blocks']),
                            'Processing Time': f"{result['processing_time']:.2f}s",
                            'Text Length': len(result['ocr_text'])
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Export batch results
                    st.subheader("Export Batch Results")
                    
                    batch_json = json.dumps(batch_results, indent=2)
                    st.download_button(
                        label="Download Batch Results (JSON)",
                        data=batch_json,
                        file_name="batch_analysis_results.json",
                        mime="application/json"
                    )
    
    with tab4:
        st.header("⚙️ Settings & Information")
        
        # System information
        st.subheader("System Information")
        
        info_data = {
            "Available Models": list(analyzer.models.keys()),
            "OCR Available": "Yes" if hasattr(analyzer, 'easyocr_reader') else "No",
            "Database Path": analyzer.db_path,
            "Modern Models Available": "Yes" if hasattr(analyzer, 'MODERN_MODELS_AVAILABLE') else "No"
        }
        
        for key, value in info_data.items():
            st.text(f"{key}: {value}")
        
        # Database management
        st.subheader("Database Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Analysis History"):
                try:
                    analyzer.session.query(analyzer.DocumentAnalysis).delete()
                    analyzer.session.commit()
                    st.success("Analysis history cleared!")
                except Exception as e:
                    st.error(f"Failed to clear history: {str(e)}")
        
        with col2:
            if st.button("Export Database"):
                try:
                    history_df = analyzer.get_analysis_history()
                    csv_data = history_df.to_csv(index=False)
                    st.download_button(
                        label="Download Database (CSV)",
                        data=csv_data,
                        file_name="analysis_database.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Failed to export database: {str(e)}")
        
        # About section
        st.subheader("About")
        st.markdown("""
        **Document Layout Analysis System**
        
        This application provides advanced document layout analysis capabilities using:
        - LayoutParser with Detectron2 models
        - YOLOv8 for modern object detection
        - OCR integration with EasyOCR
        - SQLite database for result storage
        
        **Features:**
        - Single and batch document processing
        - Multiple model support
        - Real-time visualization
        - Results export in multiple formats
        - Analysis history tracking
        """)

if __name__ == "__main__":
    main()
