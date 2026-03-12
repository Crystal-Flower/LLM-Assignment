from langchain_ollama import OllamaLLM

def generate_university_report(insights):
    """
    Converts raw AI insights into a structured professional report using local Ollama (Llama 3.2).
    """
    prompt = f"""
    You are a Senior Academic Consultant.
    
    Transform the following raw AI insights into a formal 'University Strategic Intelligence Report'.
    
    Raw Insights:
    {insights}
    
    The report must be structured with the following sections:
    1. **Executive Summary**
    2. **Student Capability Assessment**
    3. **Innovation & Research Trends**
    4. **Infrastructural & Curricular Gap Analysis**
    5. **Actionable Strategic Roadmap**
    
    Use a professional, administrative tone suitable for a Vice-Chancellor or Dean.
    """

    try:
        llm = OllamaLLM(model="llama3.2")
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"Error generating report via Ollama: {str(e)}. (Ensure Ollama is running and 'llama3.2' model is pulled)"

if __name__ == "__main__":
    # Test stub
    dummy_insights = "Students like AI. They need more GPU clusters. CSE is crowded."
    print(generate_university_report(dummy_insights))
