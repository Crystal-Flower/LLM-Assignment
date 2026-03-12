import networkx as nx
import pandas as pd

def build_expertise_graph(df):
    """
    Constructs a Knowledge Graph where nodes are Students, Skills, and Research Areas.
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        student = row["Name"]
        dept = row["Department"]
        
        # Add student node
        G.add_node(student, type="student", department=dept)
        
        # Add relationships for skills
        skills = [s.strip() for s in str(row["Technical Skills"]).split(",")]
        for skill in skills:
            if skill:
                G.add_node(skill, type="skill")
                G.add_edge(student, skill, relation="has_skill")
        
        # Add relationships for research areas
        research = [r.strip() for r in str(row["Research Areas"]).split(",")]
        for topic in research:
            if topic:
                G.add_node(topic, type="research_area")
                G.add_edge(student, topic, relation="research_interest")

    return G

def get_graph_analytics(G):
    """
    Computes key metrics from the expertise graph.
    """
    metrics = {
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "central_nodes": sorted(nx.degree_centrality(G).items(), key=lambda x: x[1], reverse=True)[:10]
    }
    return metrics

if __name__ == "__main__":
    # Test stub
    df_test = pd.DataFrame({
        "Name": ["Alice", "Bob"],
        "Department": ["AI", "CS"],
        "Technical Skills": ["Python, ML", "Java, Python"],
        "Research Areas": ["CV", "NLP"]
    })
    graph = build_expertise_graph(df_test)
    print(get_graph_analytics(graph))
