"""Improved agent configuration using elbow-method clustering tools with ADK logging."""

import logging
from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from my_agent.tools.clustering_tools import (
    preprocess_csv_for_clustering,
    perform_cluster_analysis,
    generate_cluster_profiles,
    save_clustered_data
)
from google.genai import types, Client
from typing import List, Dict, Any
import streamlit as st

# Configure logging according to ADK documentation
logger = logging.getLogger(__name__)


retry_config = types.HttpRetryOptions(
    attempts=10,  # Increased from 5
    exp_base=2,   # Reduced from 7 for faster retries
    initial_delay=0.5,  # Reduced from 1
    max_delay=30,  # Add max delay
    http_status_codes=[429, 500, 503, 504, 408],  # Added 408 (timeout)
)

logger.info("Initialized retry configuration for agents")

# Clustering agent with IMPROVED clustering tools using elbow method
streamlit_clustering_agent = Agent(
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=retry_config,
        generation_config=types.GenerateContentConfig(
            response_modalities=["TEXT"],
        )
    ),
    name="clustering_agent_improved",
    description="Performs clustering analysis using ELBOW METHOD for optimal cluster selection",
    instruction=(
        "You perform clustering analysis using ONLY the provided tools. "
        "NEVER write code, NEVER explain methodology, NEVER describe theoretical approaches. "
        "\n\n"
        "MANDATORY STEPS - NO EXCEPTIONS:\n"
        "1. IMMEDIATELY call preprocess_csv_for_clustering(csv_file=<path>)\n"
        "2. IMMEDIATELY call perform_cluster_analysis(preprocessed_data=<result from step 1>)\n"
        "   - The tool uses the ELBOW METHOD to automatically find optimal clusters (2-8)\n"
        "   - It calculates inertia, silhouette, Davies-Bouldin, and Calinski-Harabasz scores\n"
        "   - The elbow point in the inertia curve determines the optimal k\n"
        "3. IMMEDIATELY call generate_cluster_profiles(csv_file=<original path>, cluster_labels=<labels from step 2>, preprocessed_data=<result from step 1>)\n"
        "   - This generates descriptive profiles and statistics for each cluster\n"
        "4. IMMEDIATELY call save_clustered_data(csv_file=<original path>, cluster_labels=<labels from step 2>)\n"
        "\n"
        "After tool execution completes, provide a comprehensive summary:\n"
        "- Optimal clusters found (by elbow method): <n_clusters from step 2>\n"
        "- Silhouette score: <silhouette_score from step 2>\n"
        "- Davies-Bouldin score: <davies_bouldin_score from step 2>\n"
        "- Calinski-Harabasz score: <calinski_harabasz_score from step 2>\n"
        "- Cluster sizes: <cluster_sizes from step 2>\n"
        "- Selection method: <selection_method from optimization_scores>\n"
        "- Saved to: <output_path from step 4>\n"
        "\n"
        "CLUSTER PROFILES:\n"
        "For each cluster from step 3, include:\n"
        "- The complete description from cluster_profiles\n"
        "- Key feature statistics to help understand each segment\n"
        "\n"
        "FORBIDDEN: Generating Python code, explaining algorithms, describing methodology, writing 'I will', 'I would', or 'First, let me'.\n"
        "REQUIRED: Execute tools immediately and report results."
    ),
    tools=[preprocess_csv_for_clustering, perform_cluster_analysis, generate_cluster_profiles, save_clustered_data],
    output_key="clustering_results"
)

logger.info("Initialized clustering agent with elbow-method tools")

# Marketing strategy agent (unchanged)
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

logger.info("Initialized marketing strategy agent")

# Streamlit app sequential pipeline with improved clustering
sequential_agent = SequentialAgent(
    name="SegmentationPipelineImproved",
    sub_agents=[streamlit_clustering_agent, marketing_strategy_agent]
)

logger.info("Initialized sequential agent pipeline: SegmentationPipelineImproved")

# QnA agent
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

logger.info("Initialized QnA agent")

# Cache conversation history
@st.cache_resource
def get_conversation_history():
    """
    Get persistent conversation history.
    This simulates what ADK's InMemoryRunner does with sessions.
    """
    logger.debug("Retrieving conversation history from cache")
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
    logger.info("Processing QnA query")
    logger.debug(f"Question length: {len(question)} chars, Context provided: {context is not None}")

    try:
        # Get conversation history (simulates ADK session)
        conv_history = get_conversation_history()

        logger.debug(f"Context initialized: {conv_history['context_initialized']}")
        logger.debug(f"Conversation history length: {len(conv_history['messages'])}")

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
            logger.info("Sending first message with context")
        else:
            # Subsequent messages: just the question
            user_message = question
            logger.info("Sending follow-up question without context")

        logger.debug(f"Message length: {len(user_message)} chars")

        # Add user message to history
        conv_history["messages"].append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        # Make API call with full conversation history
        # This is equivalent to what ADK's InMemoryRunner does
        logger.debug("Making API call to Gemini")
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

            logger.info(f"Response received, length: {len(assistant_response)} chars")
            logger.debug(f"Total messages in history: {len(conv_history['messages'])}")

            return assistant_response
        else:
            logger.warning("No response text received from API")
            return "I couldn't generate a response. Please try rephrasing your question."

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error processing QnA query: {str(e)}")
        logger.debug(f"Full traceback:\n{error_details}")
        return f"Error processing question: {str(e)}"


def run_qna_query_simple(question: str, context: str = None) -> str:
    """
    Drop-in replacement that works reliably while maintaining agent concepts.

    Args:
        question: User's question about the segmentation analysis
        context: Combined context containing clustering results and marketing strategies.
                 Should be provided on the first call, then can be None for follow-ups.

    Returns:
        Agent's response to the question
    """
    return run_qna_hybrid(question, context)
