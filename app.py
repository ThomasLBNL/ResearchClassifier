import streamlit as st
import google.generativeai as genai
import json
import re
import csv
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Research Paper Title Classifier",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Research Paper Title Classifier")
st.markdown("Enter research paper titles and get them automatically classified using Google Gemini AI")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# API Key input
api_key = st.sidebar.text_input(
    "Enter your Google Gemini API Key:",
    type="password",
    help="Get your free API key from https://makersuite.google.com/app/apikey"
)

if api_key:
    genai.configure(api_key=api_key)
    st.sidebar.success("✅ API key configured")

# Classification options
st.sidebar.header("📊 Classification Options")
include_confidence = st.sidebar.checkbox("Include confidence score", value=True)
include_keywords = st.sidebar.checkbox("Extract key terms", value=True)
include_field_suggestions = st.sidebar.checkbox("Suggest related fields", value=False)

def create_classification_prompt(title, include_confidence, include_keywords, include_field_suggestions):
    """Create the classification prompt for Gemini based on title only"""
    
    optional_fields = []
    if include_confidence:
        optional_fields.append('"confidence_score": "number between 1-10"')
    if include_keywords:
        optional_fields.append('"keywords": ["array", "of", "key", "technical", "terms", "from", "title"]')
    if include_field_suggestions:
        optional_fields.append('"related_fields": ["array", "of", "related", "research", "areas"]')
    
    optional_fields_str = ",\n    ".join(optional_fields)
    if optional_fields_str:
        optional_fields_str = ",\n    " + optional_fields_str
    
    prompt = f"""
You are an expert research paper classifier. Based ONLY on the research paper title provided, classify the research using the predefined primary strategies below.

PRIMARY STRATEGIES (choose the most appropriate one):

1. Advancing Data Science and Computing for Biology: Develop AI/ML methods and improve data management systems to predict biological functions and enable multimodal data analysis across scales, while ensuring ethical AI standards and data security.
2. Growing Next-generation Omics and Gene-editing Tools - Expand genomic sequencing, metabolomics, and gene-editing capabilities to understand biological systems, with focus on single-cell approaches and plant spatial omics for bioenergy crops.
3. Developing Hardware to Support and Understand Biology - Create advanced bioreactors, growth chambers, sensors, and imaging systems to measure biological phenomena at relevant scales, including quantum sensing and transportable field systems.
4. Accelerating Experimentation by Integrating Technologies - Build integrated, automated laboratory systems moving toward self-driving labs that seamlessly combine data collection, robotics, and AI to accelerate biological discovery.
5. Developing New, Sustainable, Effective Bioproducts - Identify and engineer biological pathways to create renewable alternatives to petroleum-based materials, including polymers, chemicals, fuels, and coatings with improved properties.
6. Enabling Optimized Bioconversion of Diverse Feedstocks - Engineer biological systems to efficiently convert waste materials, agricultural residues, and unconventional feedstocks into valuable products using microbes and optimized bioprocesses.
7. Discovering Fundamentals in Photosynthesis and Beyond - Understand and improve natural photosynthetic processes and develop artificial photosynthesis systems to directly convert sunlight into fuels and chemicals.
8. Uncovering Molecular Foundations for Predictive Ecology - Connect genotype to phenotype in controlled laboratory settings to predict biological interactions and validate them in simplified communities and ecosystems.
9. Building Models to Bridge the Gap Between Lab and Natural Systems - Develop predictive computational models that integrate laboratory discoveries with field observations to understand complex environmental systems across scales.
10. Accelerating Environmental Solutions with Biology - Apply biological approaches to address environmental challenges like carbon sequestration, ecosystem resilience, and nutrient cycling from bench to field scale.
11. Understanding Biological Processes Vital to Health - Study fundamental human biology to establish health baselines, identify disease biomarkers, and understand genetic variations that affect disease susceptibility and treatment response.
12. Addressing Environmental Impacts on People - Investigate how environmental factors like air pollution, radiation, heat, and chemical exposures affect human health and the human microbiome.
13. Developing Diagnostics, Treatments, and Mitigations for Biopreparedness - Create rapid detection systems, therapeutic platforms, and biosecurity measures to respond to emerging infectious diseases and biological threats using synthetic biology and AI approaches.

Please return your response as a valid JSON object with the following structure:

{{
    "primary_strategy": "string (must be exactly one of the 13 strategies listed above)",
    "strategy_description": "string (2-3 sentence description explaining why this strategy was chosen for this title)"{optional_fields_str}
}}

Research Paper Title: "{title}"

Analyze the title carefully and select the most appropriate primary strategy from the 13 options above. Provide only the JSON response, no additional text.
"""
    
    return prompt

def classify_title(title, api_key):
    """Classify using Google Gemini API"""
    prompt = create_classification_prompt(
        title, include_confidence, include_keywords, include_field_suggestions
    )
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500,
            )
        )
        return response.text
            
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        if "API_KEY_INVALID" in str(e):
            st.error("Please check your API key. Get one from: https://makersuite.google.com/app/apikey")
        return None

def parse_json_response(response_text):
    """Parse the JSON response from Gemini"""
    try:
        # Clean the response text
        response_text = response_text.strip()
        
        # Remove any markdown formatting
        response_text = re.sub(r'```json\n?', '', response_text)
        response_text = re.sub(r'```\n?', '', response_text)
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            return json.loads(response_text)
    except Exception as e:
        st.error(f"Error parsing AI response: {str(e)}")
        return None

def display_results(title, classification_result):
    """Display classification results in a nice format"""
    if not classification_result:
        return
    
    # Show the title being classified
    st.subheader(f"📖 \"{title}\"")
    
    # Main classification info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Primary Strategy")
        strategy = classification_result.get('primary_strategy', 'Unknown')
        description = classification_result.get('strategy_description', 'No description provided')
        
        st.write(f"**Strategy:** {strategy}")
        st.write(f"**Explanation:** {description}")
        
    with col2:
        if 'confidence_score' in classification_result:
            confidence = classification_result['confidence_score']
            st.metric("Confidence Score", f"{confidence}/10")
    
    # Optional sections
    if 'keywords' in classification_result:
        st.subheader("🔍 Key Terms")
        keywords = classification_result['keywords']
        st.write(", ".join(keywords))
    
    if 'related_fields' in classification_result:
        st.subheader("🌐 Related Fields")
        related_fields = classification_result['related_fields']
        st.write(", ".join(related_fields))

# Initialize session state for storing results
if 'results_history' not in st.session_state:
    st.session_state.results_history = []

# Main interface
st.header("📝 Enter Research Paper Title")

# Single title input
title_input = st.text_input(
    "Research Paper Title:",
    placeholder="e.g., 'Deep Learning for Medical Image Segmentation: A Comprehensive Survey'"
)

# Batch input option
st.subheader("📋 Or enter multiple titles")
batch_input = st.text_area(
    "Enter multiple titles (one per line):",
    height=150,
    placeholder="Enter multiple research paper titles, one per line..."
)

# Process button
if st.button("🔍 Classify Title(s)", type="primary"):
    if not api_key:
        st.error("Please enter your Google Gemini API key in the sidebar")
        st.info("Get your free API key from: https://makersuite.google.com/app/apikey")
    elif not title_input and not batch_input:
        st.error("Please enter at least one research paper title")
    else:
        # Determine which titles to process
        titles_to_process = []
        if title_input:
            titles_to_process.append(title_input.strip())
        if batch_input:
            batch_titles = [title.strip() for title in batch_input.split('\n') if title.strip()]
            titles_to_process.extend(batch_titles)
        
        # Remove duplicates while preserving order
        titles_to_process = list(dict.fromkeys(titles_to_process))
        
        with st.spinner(f"Analyzing {len(titles_to_process)} title(s) with Google Gemini..."):
            results = []
            
            for i, title in enumerate(titles_to_process):
                st.write(f"Processing {i+1}/{len(titles_to_process)}: {title[:50]}...")
                
                response = classify_title(title, api_key)
                
                if response:
                    classification_result = parse_json_response(response)
                    if classification_result:
                        # Add timestamp and title to result
                        classification_result['title'] = title
                        classification_result['timestamp'] = datetime.now().isoformat()
                        results.append(classification_result)
                        
                        # Add to session history
                        st.session_state.results_history.append(classification_result)
            
            if results:
                st.success(f"✅ Successfully classified {len(results)} title(s)!")
                
                # Display results
                for result in results:
                    with st.container():
                        display_results(result['title'], result)
                        st.markdown("---")
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    # JSON download
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json.dumps(results, indent=2),
                        file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                with col2:
                    # CSV download
                    csv_data = []
                    for result in results:
                        row = {
                            'title': result.get('title', ''),
                            'primary_strategy': result.get('primary_strategy', ''),
                            'strategy_description': result.get('strategy_description', ''),
                        }
                        if 'confidence_score' in result:
                            row['confidence_score'] = result['confidence_score']
                        if 'keywords' in result:
                            row['keywords'] = ', '.join(result['keywords'])
                        if 'related_fields' in result:
                            row['related_fields'] = ', '.join(result['related_fields'])
                        csv_data.append(row)
                    
                    # Convert to CSV string
                    import io
                    output = io.StringIO()
                    if csv_data:
                        writer = csv.DictWriter(output, fieldnames=csv_data[0].keys())
                        writer.writeheader()
                        writer.writerows(csv_data)
                        csv_string = output.getvalue()
                        
                        st.download_button(
                            label="📊 Download Results (CSV)",
                            data=csv_string,
                            file_name=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

# Show history if available
if st.session_state.results_history:
    with st.expander(f"📚 Classification History ({len(st.session_state.results_history)} titles)"):
        for result in reversed(st.session_state.results_history[-10:]):  # Show last 10
            st.write(f"**{result['title']}** → {result.get('primary_strategy', 'Unknown')}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.results_history = []
            st.experimental_rerun()

# Instructions
with st.expander("📖 How to use"):
    st.markdown("""
    1. **Get your API key**: 
       - Go to https://makersuite.google.com/app/apikey
       - Sign in with your Google account
       - Click "Create API key"
       - Copy the key and paste it in the sidebar
       
    2. **Enter title(s)**: 
       - Single title: Use the text input above
       - Multiple titles: Use the text area (one title per line)
       
    3. **Configure options**: 
       - Choose what additional information you want
       
    4. **Classify**: 
       - Click "Classify Title(s)" and wait for results
       
    5. **Download**: 
       - Save results as JSON or CSV for spreadsheet analysis
    
    **What you'll get for each title**:
    - Primary strategy classification (from 13 predefined categories)
    - 2-3 sentence explanation of why that strategy was chosen
    - Optional: confidence scores, keywords, related fields
    
    **The 13 Primary Strategies:**
1. Advancing Data Science and Computing for Biology
2. Growing Next-generation Omics and Gene-editing Tools
3. Developing Hardware to Support and Understand Biology
4. Accelerating Experimentation by Integrating Technologies
5. Developing New, Sustainable, Effective Bioproducts
6. Enabling Optimized Bioconversion of Diverse Feedstocks
7. Discovering Fundamentals in Photosynthesis and Beyond
8. Uncovering Molecular Foundations for Predictive Ecology
9. Building Models to Bridge the Gap Between Lab and Natural Systems
10. Accelerating Environmental Solutions with Biology
11. Understanding Biological Processes Vital to Health
12. Addressing Environmental Impacts on People
13. Developing Diagnostics, Treatments, and Mitigations for Biopreparedness
    """)

# API key help
with st.expander("🔑 Need help getting a Gemini API key?"):
    st.markdown("""
    **Step-by-step guide:**
    
    1. Visit https://makersuite.google.com/app/apikey
    2. Sign in with your Google account
    3. Click "Create API key"
    4. Copy the generated key
    5. Paste it in the sidebar of this app
    
    **Note**: The Gemini API has a generous free tier, so you can start using it immediately without any payment.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Google Gemini AI | Optimized for research paper title classification")
