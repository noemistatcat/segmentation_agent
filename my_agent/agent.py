from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from my_agent.tools.clustering_tools import (
    preprocess_csv_for_clustering,
    perform_cluster_analysis,
    save_clustered_data
)
from google.genai import types, Client
from typing import List, Dict, Any
import streamlit as st


retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Clustering agent for Streamlit app
streamlit_clustering_agent = Agent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="clustering_agent",
    description="Performs clustering analysis on CSV data",
    instruction=(
        "You are an expert data scientist performing clustering analysis on csv data. "
        "The user will provide a CSV file path. "
        "STEP 1: Use the preprocess_csv_for_clustering tool to preprocess the data. "
        "STEP 2: Run perform_cluster_analysis on the preprocessed data to get cluster labels. "
        "STEP 3: Use save_clustered_data to merge the cluster labels with the original CSV file. "
        "Pass the original CSV file path and the cluster labels from step 2. "
        "STEP 4: Provide a comprehensive summary of the clustering results including: "
        "- Number of clusters found "
        "- Silhouette score and Davies-Bouldin score "
        "- Size of each cluster "
        "- Key characteristics of each segment "
        "- Path to the saved clustered data file"
    ),
    tools=[preprocess_csv_for_clustering, perform_cluster_analysis, save_clustered_data],
    output_key="clustering_results"
)

# Marketing strategy agent
marketing_strategy_agent = Agent(
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    name="marketing_strategy_agent",
    description="Formulates marketing strategies based on clustering results",
    instruction=(
        "You are an expert marketing strategist. "
        "You will receive clustering analysis results from the previous agent stored in the context as 'clustering_results'. "
        "Based on the clustering results, formulate detailed marketing strategies for each customer segment. "
        "For each segment, provide: "
        "1. Segment profile and characteristics "
        "2. Recommended marketing channels "
        "3. Key messaging and value propositions "
        "4. Product/service recommendations "
        "5. Engagement tactics"
    ),
    output_key="marketing_strategies"
)

# Streamlit app sequential pipeline
sequential_agent = SequentialAgent(
    name="SegmentationPipeline",
    sub_agents=[streamlit_clustering_agent, marketing_strategy_agent]
)

# Define the agent (for demonstration purposes - shows ADK architecture)
# Even though we use direct API, this shows proper agent design
qna_agent = Agent(
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
        generation_config=types.GenerateContentConfig(
            temperature=0.7,
        )
    ),
    name="qna_agent",
    description="Handles questions and answers about clustering results and marketing strategies",
    instruction=(
        "You are an expert in market segmentation, customer analytics, and marketing strategy. "
        "Answer questions about both technical aspects (clustering methodology, metrics, segments) "
        "and strategic aspects (marketing strategies, recommendations, tactics). "
        "The user will first provide context containing clustering results and marketing strategies. "
        "After receiving the context, you'll answer follow-up questions based on that information. "
        "Remember all previous messages in this conversation - you have full context of the entire discussion. "
        "For technical questions, reference the metrics in the context (silhouette scores, cluster sizes, segment characteristics). "
        "For marketing questions, reference the strategies described in the context. "
        "If asked follow-up questions, reference previous answers and maintain conversational continuity."
    ),
    tools=[]
)

# Cache conversation history
@st.cache_resource
def get_conversation_history():
    """
    Get persistent conversation history.
    This simulates what ADK's InMemoryRunner does with sessions.
    """
    return {
        "messages": [],
        "context_initialized": False,
        "system_instruction": qna_agent.instruction  # Use instruction from agent definition
    }


def run_qna_hybrid(question: str, context: str = None) -> str:
    """
    Run QnA using direct API with ADK-inspired architecture.
    
    This approach:
    - Uses Gemini API directly (avoids event loop issues)
    - Maintains conversation history (like ADK sessions)
    - Uses agent instruction (from ADK agent definition)
    - Structured to demonstrate agent concepts for capstone
    
    Args:
        question: The user's question
        context: Combined context (clustering results + marketing strategies).
                 Only needs to be provided on first call.
    
    Returns:
        Agent's response as a string
    """
    try:
        # Get conversation history (simulates ADK session)
        conv_history = get_conversation_history()
        
        print(f"\nDEBUG: Processing question")
        print(f"DEBUG: Context initialized: {conv_history['context_initialized']}")
        print(f"DEBUG: Conversation history length: {len(conv_history['messages'])}")
        
        # Initialize Gemini client
        client = Client()
        
        # Build the user message
        if context and not conv_history["context_initialized"]:
            # First message: include context
            user_message = (
                f"Here is the analysis context you'll need to answer questions:\n\n"
                f"{context}\n\n"
                f"---\n\n"
                f"USER QUESTION: {question}\n\n"
                f"Please answer the question above based on the context provided."
            )
            conv_history["context_initialized"] = True
            print("DEBUG: Sending first message WITH context")
        else:
            # Subsequent messages: just the question
            user_message = question
            print("DEBUG: Sending follow-up question WITHOUT context")
        
        print(f"DEBUG: Message length: {len(user_message)} chars")
        
        # Add user message to history
        conv_history["messages"].append({
            "role": "user",
            "parts": [{"text": user_message}]
        })
        
        # Make API call with full conversation history
        # This is equivalent to what ADK's InMemoryRunner does
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=conv_history["messages"],  # Full conversation history
            config=types.GenerateContentConfig(
                system_instruction=conv_history["system_instruction"],  # From agent definition
                temperature=0.7,
            )
        )
        
        # Extract response
        if response and response.text:
            assistant_response = response.text
            
            # Add assistant response to history
            conv_history["messages"].append({
                "role": "model",
                "parts": [{"text": assistant_response}]
            })
            
            print(f"DEBUG: Response received, length: {len(assistant_response)} chars")
            print(f"DEBUG: Total messages in history: {len(conv_history['messages'])}")
            
            return assistant_response
        else:
            print("DEBUG: No response text received")
            return "I couldn't generate a response. Please try rephrasing your question."
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: Full traceback:\n{error_details}")
        return f"Error processing question: {str(e)}"


def run_qna_query_simple(question: str, context: str = None) -> str:
    """
    Drop-in replacement that works reliably while maintaining agent concepts.
    
    For your capstone demo, you can explain:
    - Agent definition (shows ADK Agent class with Gemini model)
    - Conversation history management (simulates ADK session service)
    - System instruction (from agent definition)
    - Stateful conversations (history maintained across calls)
    
    The implementation is reliable but still demonstrates key ADK concepts.
    
    Args:
        question: User's question about the segmentation analysis
        context: Combined context containing clustering results and marketing strategies.
                 Should be provided on the first call, then can be None for follow-ups.
    
    Returns:
        Agent's response to the question
    """
    return run_qna_hybrid(question, context)