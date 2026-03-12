from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def form_research_teams(df, n_teams=None):
    """
    Groups students into research teams based on their Skills and Research Areas.
    Uses TF-IDF vectorization and K-Means clustering.
    """
    if df.empty:
        return {}

    # Combine Skills and Research Areas for clustering context
    df['combined_features'] = df['Technical Skills'] + " " + df['Research Areas']
    
    # Vectorize text features
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['combined_features'])
    
    # Determine optimal number of teams (approx 3 students per team)
    if n_teams is None:
        n_teams = max(1, len(df) // 3)
    
    # Run K-Means
    kmeans = KMeans(n_clusters=n_teams, random_state=42, n_init=10)
    df['team_id'] = kmeans.fit_predict(X)
    
    # Organize results
    teams = {}
    for team_id in range(n_teams):
        team_members = df[df['team_id'] == team_id]
        teams[team_id] = team_members[['Name', 'Department', 'Technical Skills', 'Research Areas']].to_dict('records')
        
    return teams

if __name__ == "__main__":
    # Test stub
    df_test = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
        "Technical Skills": ["Python, ML", "Java, SQL", "Python, Pandas", "C++, Embedded", "Python, Scikit", "C++, Arduino"],
        "Research Areas": ["CV", "Blockchain", "NLP", "Robotics", "Deep Learning", "IoT"]
    })
    teams = form_research_teams(df_test)
    for tid, members in teams.items():
        print(f"Team {tid}: {[m['Name'] for m in members]}")
