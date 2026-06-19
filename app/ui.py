import streamlit as st
import requests
import os

BACKEND_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="Advanced Video Analytics",
    page_icon="🎥",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    .frame-card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin-bottom: 15px; }
    .result-container { background-color: #161b22; border-left: 4px solid #1f6feb; border-radius: 8px; padding: 15px; margin-bottom: 12px; }
    .badge-time { background-color: #1f6feb; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; display: inline-block; margin-right: 8px; }
    .badge-score { background-color: #238636; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; display: inline-block; }
    .tracked-video { border: 3px solid #00ff00; border-radius: 8px; }
    .search-video { border: 3px solid #1f6feb; border-radius: 8px; }
    .video-info { font-size: 0.95em; color: #8b949e; margin-top: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎥 Advanced Video Analytics with Persistent Tracking")
st.markdown("**Semantic Search + YOLO Detection + Target Person Tracking**")

# Initialize session state
if "video_id" not in st.session_state:
    st.session_state["video_id"] = None
if "current_video_name" not in st.session_state:
    st.session_state["current_video_name"] = None
if "search_results" not in st.session_state:
    st.session_state["search_results"] = None

# Sidebar - Upload
with st.sidebar:
    st.header("📤 Upload Video")
    uploaded_file = st.file_uploader("Choose video file", type=["mp4", "avi", "mov", "mkv"])
    
    if st.button("⬆️ Upload & Process Video", use_container_width=True):
        if uploaded_file:
            with st.spinner("Processing video..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state["video_id"] = data["video_id"]
                        st.session_state["current_video_name"] = uploaded_file.name
                        st.success(f"✅ Video Processed! ID: {data['video_id']}")
                        st.info(f"Extracted {data.get('message', '')} frames" if "Extracted" in data.get('message', '') else "")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please select a video file.")

    st.divider()
    if st.session_state.get("video_id"):
        st.success(f"🟢 **Active Video:**\n{st.session_state['current_video_name']}\n\nID: `{st.session_state['video_id']}`")
    else:
        st.info("👈 Upload a video to start")

# Main Area
st.header("🔍 Semantic Search")
st.markdown("Search for objects or scenes using natural language (e.g., 'man in suit', 'person with glasses')")

col1, col2 = st.columns([3, 1])
with col1:
    query_text = st.text_input("Enter search query", placeholder="e.g., man in black suit, person walking...")
with col2:
    threshold = st.slider("Confidence", 0.0, 1.0, 0.20, 0.05, label_visibility="collapsed")

if st.button("🔍 Search Matching Frames", use_container_width=True):
    if not query_text:
        st.warning("Please enter a search query")
    elif not st.session_state.get("video_id"):
        st.error("Please upload a video first")
    else:
        with st.spinner("🔄 Searching for matching frames..."):
            payload = {"query": query_text, "threshold": threshold, "video_id": st.session_state["video_id"]}
            response = requests.post(f"{BACKEND_URL}/query", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                st.session_state["search_results"] = results
                
                if results:
                    st.success(f"✅ Found {len(results)} matching frames!")
                else:
                    st.info("No matches found. Try lowering the confidence threshold or changing your query.")
            else:
                st.error(f"Search failed: {response.text}")

# Display search results
if st.session_state.get("search_results"):
    st.divider()
    st.header("📹 Search Results")
    
    results = st.session_state["search_results"]
    
    # Display results in a grid
    cols = st.columns(3)
    for idx, result in enumerate(results):
        with cols[idx % 3]:
            with st.container(border=True):
                # Display frame image
                st.image(result["frame_path"], use_container_width=True)
                
                # Metadata
                st.markdown(f'<span class="badge-time">⏱️ {result["timestamp"]}s</span><span class="badge-score">Score: {result["similarity_score"]:.2%}</span>', 
                           unsafe_allow_html=True)
                
                st.markdown(f'<div class="video-info">📁 {st.session_state["current_video_name"]}</div>', 
                           unsafe_allow_html=True)
    
    # Video generation buttons
    st.divider()
    st.header("🎬 Generate Videos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎬 Generate Tracked Video (Re-ID)", use_container_width=True):
            with st.spinner("⏳ Generating tracked video with persistent target tracking... This may take a while"):
                track_payload = {
                    "query": query_text,
                    "threshold": threshold,
                    "video_id": st.session_state["video_id"]
                }
                track_resp = requests.post(f"{BACKEND_URL}/generate_tracked", json=track_payload)
                
                if track_resp.status_code == 200:
                    track_data = track_resp.json()
                    st.success("✅ Tracked Video Generated!")
                    st.markdown("**Tracked Video** (Green boxes show target person with persistent ID tracking)")
                    
                    # Display video from download endpoint
                    video_url = f"{BACKEND_URL}/download/tracked/{st.session_state['video_id']}"
                    st.video(video_url)
                    
                    # Download button
                    with st.spinner("Preparing download..."):
                        video_response = requests.get(video_url)
                        if video_response.status_code == 200:
                            st.download_button(
                                label="Download Tracked Video",
                                data=video_response.content,
                                file_name=f"{st.session_state['video_id']}_tracked.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
                else:
                    st.error(f"Failed to generate tracked video: {track_resp.text}")
    
    with col2:
        if st.button("🟦 Generate YOLO Boxes Video", use_container_width=True):
            with st.spinner("⏳ Generating video with YOLO detection boxes... This may take a while"):
                try:
                    search_resp = requests.post(f"{BACKEND_URL}/generate_search_video?video_id={st.session_state['video_id']}")
                    
                    if search_resp.status_code == 200:
                        search_data = search_resp.json()
                        st.success("✅ YOLO Boxes Video Generated!")
                        st.markdown("**YOLO Detection Video** (Blue boxes show all detected persons with confidence scores)")
                        
                        # Display video from download endpoint
                        video_url = f"{BACKEND_URL}/download/yolo/{st.session_state['video_id']}"
                        st.video(video_url)
                        
                        # Download button
                        with st.spinner("Preparing download..."):
                            video_response = requests.get(video_url)
                            if video_response.status_code == 200:
                                st.download_button(
                                    label="Download YOLO Boxes Video",
                                    data=video_response.content,
                                    file_name=f"{st.session_state['video_id']}_yolo_boxes.mp4",
                                    mime="video/mp4",
                                    use_container_width=True
                                )
                    else:
                        st.error(f"Failed to generate search video: {search_resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

st.divider()
st.caption("💡 **Tips:**\n- **Tracked Video**: Shows persistent tracking of a target person matching your query (green boxes)\n- **YOLO Boxes Video**: Shows ALL detected persons throughout the video (blue boxes with confidence scores)")