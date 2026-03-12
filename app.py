import streamlit as st
import pandas as pd
import os
import networkx as nx
import plotly.graph_objects as go
from analytics.insight_generator import generate_insights
from graphs.expertise_graph import build_expertise_graph, get_graph_analytics
from ml_models.team_builder import form_research_teams
from reports.report_generator import generate_university_report
from utils.rag_engine import RAGEngine

# load_dotenv(".env") # No longer needed for local Ollama


# --- Page Config ---
st.set_page_config(page_title="AmritaGPT v2 — Intel", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #1e1e1e; border-radius: 4px; color: #ffffff; }
    .stTabs [aria-selected="true"] { background-color: #2e2e2e; border-bottom: 2px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 AmritaGPT v2 — Research Intelligence Platform")
st.subheader("Next-Level Academic Intelligence & Expertise Mapping")

# --- Load Data ---
DATA_PATH = os.path.join("data", "student_data.csv")
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.error("Dataset not found! Please ensure data/student_data.csv exists.")
    st.stop()

# Initialize RAG
@st.cache_resource
def get_rag_engine(_df):
    return RAGEngine(_df)

rag_engine = get_rag_engine(df)

# --- Sidebar ---
st.sidebar.header("System Controls")
if st.sidebar.button("Refresh Data"):
    st.rerun()

st.sidebar.info(f"Total Students: {len(df)}")
st.sidebar.markdown("---")
st.sidebar.write("Built for Amrita University Research Labs")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Analytics", 
    "🧠 AI Insights", 
    "🕸 Knowledge Graph", 
    "🤝 Research Teams", 
    "💬 AI Advisor", 
    "📑 University Report"
])

# --- Tab 1: Analytics ---
with tab1:
    st.header("Student Population & Skill Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Department Distribution")
        dept_counts = df["Department"].value_counts()
        st.bar_chart(dept_counts)
        
    with col2:
        st.subheader("Top Technical Skills")
        all_skills = df["Technical Skills"].str.split(', ').explode().str.strip().value_counts().head(15)
        st.bar_chart(all_skills)

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

# --- Tab 2: AI Insights ---
with tab2:
    st.header("LLM-Powered Strategic Insights")
    if st.button("Generate Strategic Analysis"):
        with st.spinner("Analyzing dataset trends..."):
            insights = generate_insights(df)
            st.markdown(insights)
            st.session_state['latest_insights'] = insights

# --- Tab 3: Knowledge Graph ---
with tab3:
    st.header("Expertise Knowledge Graph")
    G = build_expertise_graph(df)
    metrics = get_graph_analytics(G)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", metrics["node_count"])
    col2.metric("Edges", metrics["edge_count"])
    col3.write("**Top Central Nodes (Hubs):**")
    for node, score in metrics["central_nodes"]:
        col3.write(f"- {node} ({score:.2f})")

    st.info("The graph models connections between Students, Skills, and Research topics. Visualization below (static placeholder for performance).")
    
    # Simple static visualization using Plotly for the demo
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0); edge_x.append(x1); edge_x.append(None)
        edge_y.append(y0); edge_y.append(y1); edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=[], line_width=2)
    )
    node_trace.text = node_text
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0)))
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 4: Research Teams ---
with tab4:
    st.header("AI-Generated Research Teams")
    n_teams = st.slider("Number of teams to form", 2, 15, 5)
    if st.button("Form Teams"):
        teams = form_research_teams(df, n_teams=n_teams)
        for tid, members in teams.items():
            with st.expander(f"Team {tid+1} ({len(members)} Members)"):
                for m in members:
                    st.write(f"👤 **{m['Name']}** ({m['Department']})")
                    st.caption(f"Skills: {m['Technical Skills']} | Research: {m['Research Areas']}")

# --- Tab 5: AI Advisor ---
with tab5:
    st.header("Student Intelligence Chat (RAG)")
    query = st.text_input("Ask about students, skills, or research expertise:")
    if query:
        with st.spinner("Searching knowledge base..."):
            results = rag_engine.search_students(query)
            st.subheader("Relevant Profiles:")
            for doc in results:
                st.info(doc.page_content)

# --- Tab 6: University Report ---
with tab6:
    st.header("Academic Intelligence Reporting")
    if 'latest_insights' not in st.session_state:
        st.warning("Please generate AI Insights first in the 'AI Insights' tab.")
    else:
        if st.button("Generate Professional Report"):
            with st.spinner("Compiling formal report..."):
                report = generate_university_report(st.session_state['latest_insights'])
                st.markdown(report)
                st.download_button("Download Report (TXT)", report, file_name="AmritaGPT_Intel_Report.txt")
