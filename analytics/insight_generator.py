from langchain_ollama import OllamaLLM
import pandas as pd

def generate_insights(df):
    """
    Analyzes student statistics and generates qualitative insights using local Ollama (Llama 3.2).
    """
    if df.empty:
        return "No data available to generate insights."

    # Prepare summary for LLM
    summary = {
        "total_students": len(df),
        "departments": df["Department"].value_counts().to_dict(),
        "top_skills": df["Technical Skills"].str.split(', ').explode().str.strip().value_counts().head(10).to_dict(),
        "top_research_areas": df["Research Areas"].str.split(', ').explode().str.strip().value_counts().head(10).to_dict(),
        "career_aspirations": df["Intended Career Path"].value_counts().head(5).to_dict()
    }

    prompt = f"""
    You are an expert Academic Intelligence Analyst.
    
    Based on the following student dataset statistics, generate a high-level strategic intelligence report.
    
    Data Summary:
    {summary}
    
    Please provide:
    1. **Key Trends**: What are the most popular skills and research interests?
    2. **Skill Gap Analysis**: Based on career aspirations and research interests, what skills might be missing or under-represented?
    3. **Strategic Recommendations**: How can the university better align its curriculum or research initiatives with these trends?
    
    Keep the tone professional and forward-looking.
    """

    try:
        llm = OllamaLLM(model="llama3.2")
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error generating insights via Ollama: {str(e)}. (Ensure Ollama is running and 'llama3.2' model is pulled)"

if __name__ == "__main__":
    # Test stub
    data = {
        "Department": ["AI & DS", "CSE", "AI & DS"],
        "Technical Skills": ["Python, ML", "Java, SQL", "Python, Pandas"],
        "Research Areas": ["CV", "Blockchain", "NLP"],
        "Intended Career Path": ["AI Researcher", "Software Engineer", "Data Scientist"]
    }
    df = pd.DataFrame(data)
    print(generate_insights(df))
