# Market Segmentation Agent

An intelligent market segmentation application powered by Google's ADK (Agent Development Kit) and Gemini AI, designed to automate customer clustering analysis and generate actionable marketing strategies.

## Why This Project?

Market segmentation is critical for businesses to understand their customer base and tailor marketing strategies effectively. However, traditional segmentation analysis requires:
- Manual data preprocessing and feature engineering
- Determining the optimal number of clusters through trial and error
- Expert knowledge to interpret clustering results
- Time-consuming strategy formulation for each segment

This project automates the entire segmentation pipeline using AI agents, reducing analysis time from hours to minutes while providing expert-level insights.

## What It Does

The Market Segmentation Agent is an end-to-end solution that:

1. **Analyzes customer data** using advanced clustering algorithms with automatic cluster optimization
2. **Generates marketing strategies** tailored to each identified customer segment
3. **Provides interactive Q&A** allowing users to explore insights and ask follow-up questions
4. **Exports comprehensive reports** in multiple formats (DOCX, CSV)

Built with a Streamlit interface, the application makes sophisticated data science accessible to non-technical stakeholders like marketing managers, product developers, and business owners.

## The Problem

Traditional market segmentation faces several challenges:

- **Manual cluster selection**: Data scientists must manually test different cluster counts (k values) and compare metrics
- **Inconsistent methodology**: Different analysts may arrive at different segmentation solutions
- **Time-intensive process**: Each analysis requires multiple iterations of preprocessing, clustering, and validation
- **Strategy gap**: Technical clustering results often aren't translated into actionable marketing strategies
- **Limited interactivity**: Static reports don't allow stakeholders to ask follow-up questions

## The Solution

This application implements an **agentic AI pipeline** that orchestrates specialized AI agents to automate the entire workflow:

### 1. Clustering Agent
- Automatically preprocesses CSV data (handles numerical and categorical features)
- Uses the **elbow method** to determine optimal cluster count 
- Calculates multiple validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Generates cluster labels and saves enriched data

### 2. Marketing Strategy Agent
- Receives clustering results from the previous agent
- Formulates detailed marketing strategies for each segment
- Provides actionable recommendations including:
  - Segment profiles and characteristics
  - Recommended marketing channels
  - Key messaging and value propositions
  - Product/service recommendations
  - Engagement tactics

### 3. Q&A Agent
- Maintains conversational context about the analysis
- Answers technical questions about clustering methodology
- Answers strategic questions about marketing recommendations
- Provides follow-up insights and explanations

## Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                       │
│  (File Upload → Run Analysis → Download Reports → Chat)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Google ADK Layer                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Sequential Agent Pipeline                     │  │
│  │                                                        │  │
│  │  ┌─────────────────┐    ┌──────────────────────┐    │  │
│  │  │ Clustering Agent│ →  │Marketing Strategy    │    │  │
│  │  │                 │    │Agent                 │    │  │
│  │  │ • Preprocesses  │    │ • Receives context   │    │  │
│  │  │ • Clusters data │    │ • Generates          │    │  │
│  │  │ • Selects k     │    │   strategies         │    │  │
│  │  └─────────────────┘    └──────────────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Q&A Agent (Standalone)                   │  │
│  │  • Maintains conversation history                     │  │
│  │  • Answers follow-up questions                        │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Tool Layer                                 │
│                                                              │
│  • preprocess_csv_for_clustering()                          │
│  • perform_cluster_analysis()  ← Elbow method               │
│  • save_clustered_data()                                    │
└─────────────────────────────────────────────────────────────┘
```

### Agent Communication Flow

```
User uploads CSV
    │
    ▼
┌───────────────────────────────────────────────────┐
│ InMemoryRunner creates session                    │
│ • User ID: user1                                  │
│ • Session ID: unique UUID                         │
└───────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────┐
│ Clustering Agent                                  │
│ 1. Calls: preprocess_csv_for_clustering()        │
│    → Returns: cache_key, features metadata        │
│                                                   │
│ 2. Calls: perform_cluster_analysis()             │
│    → Elbow method finds optimal k                │
│    → Returns: labels, metrics, scores             │
│                                                   │
│ 3. Calls: save_clustered_data()                  │
│    → Returns: output_path to CSV                 │
│                                                   │
│ Output Key: "clustering_results"                 │
└───────────────────────────────────────────────────┘
    │
    ▼ (Sequential handoff)
┌───────────────────────────────────────────────────┐
│ Marketing Strategy Agent                          │
│ • Receives context: clustering_results            │
│ • Gemini 2.5 Flash generates strategies          │
│ • Output Key: "marketing_strategies"             │
└───────────────────────────────────────────────────┘
    │
    ▼
Results returned to Streamlit UI
```

### Key Architecture Components

#### 1. Google ADK Integration
- **InMemoryRunner**: Manages agent execution and session state
- **SequentialAgent**: Orchestrates clustering → marketing pipeline
- **Agent**: Individual AI agents with specific roles and tools
- **Gemini 2.5 Flash**: Underlying LLM with retry configuration

#### 2. Tool System
Tools are Python functions decorated with `@tool` that agents can call:
- Feature caching to avoid passing large arrays through LLM context
- Comprehensive logging for debugging and monitoring
- Type-safe interfaces with validation

#### 3. Elbow Method Implementation
The clustering optimization uses a mathematical approach to find the elbow point:
```python
# Normalize k and inertia values
# Calculate perpendicular distance from each point to line (first → last)
# Select k with maximum distance = maximum curvature = elbow
```

#### 4. Session Management
- Streamlit's `st.cache_resource` for persistent runners
- ADK's session service for conversation history
- Separate sessions for pipeline and Q&A

## Setup Instructions

### Prerequisites
- Python 3.11+
- Google API key with Gemini API access

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd segmentation_agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv_adk
   source .venv_adk/bin/activate  # On Windows: .venv_adk\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn python-docx google-genai google-adk nest-asyncio
   ```

4. **Configure API key**

   Create a `.env` file in the project root:
   ```bash
   GOOGLE_API_KEY="your-api-key-here"
   ```

   Or set as environment variable:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the UI**

   Open your browser to `http://localhost:8501`

### Project Structure

```
segmentation_agent/
├── app.py                          # Main Streamlit application
├── my_agent/
│   ├── __init__.py
│   ├── agent.py                    # Agent definitions and configurations
│   ├── compat/                     # Compatibility shims for ADK
│   └── tools/
│       ├── __init__.py
│       └── clustering_tools.py     # Clustering tool implementations
├── archive/                        # Development notebooks and experiments
├── .env                            # API keys (not committed)
├── .gitignore
└── README.md
```

## Usage

### Basic Workflow

1. **Upload CSV file** with customer data (demographic, behavioral, transactional data)
2. **Click "Run Analysis"** to start the automated pipeline
3. **Download results**:
   - Clustering Analysis Results (DOCX)
   - Marketing Strategies (DOCX)
   - Raw Data with Clusters (CSV)
4. **Ask questions** in the Q&A chat interface

### Input Data Format

Your CSV should contain:
- **Numerical columns**: Age, income, purchase amounts, frequency, etc.
- **Categorical columns**: Gender, product preferences, region, etc.
- **Optional ID columns**: Customer_ID (automatically excluded from clustering)

Columns to avoid using as features:
- `id`, `true_segment`, `segment`, `cluster`, `label` (automatically excluded)

### Example Questions for Q&A

- "How many clusters were there?"
- "How did you determine the number of clusters?"
- "What algorithm did you use?"
- "Is this a good solution?"
- "I work in product development - what are products that you can make for this segment?"
- "I work as an advertising manager - How would you advertise that product?"
- "Which segment should I prioritize if I am a small business owner?"

## Technical Details

### Clustering Algorithm

**MiniBatch K-Means** is used for computational efficiency:
- Batch size: `min(500, max(100, n_samples // 10))`
- Final clustering: 100 iterations, 5 initializations
- Optimization testing: 50 iterations, 3 initializations

### Cluster Optimization: Elbow Method

The system tests k=2 to k=8 and calculates:
- **Inertia**: Within-cluster sum of squares (lower is better)
- **Silhouette Score**: [-1, 1] - measures cluster cohesion (higher is better)
- **Davies-Bouldin Score**: Cluster separation (lower is better)
- **Calinski-Harabasz Score**: Variance ratio (higher is better)

The **elbow point** is identified using geometric distance calculation:
1. Plot k vs. inertia
2. Draw a line from first to last point
3. Find the point with maximum perpendicular distance
4. This point represents the elbow (optimal k)

### Performance Optimizations

- **Feature caching**: Preprocessed features stored in memory, not passed through LLM
- **Metric sampling**: For datasets > 500 rows, metrics calculated on 500-sample subset
- **Batch processing**: MiniBatch K-Means for large datasets
- **Retry configuration**: Exponential backoff for API rate limits

### Logging

Comprehensive logging follows ADK best practices:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
```

Log levels used:
- **INFO**: High-level operations (agent initialization, analysis start/complete)
- **DEBUG**: Detailed diagnostics (feature counts, k-testing, metric values)
- **ERROR**: Exception handling with full tracebacks

### LLM Configuration

**Gemini 2.5 Flash** parameters:
- Temperature: 0.7 (for Q&A agent)
- Response modality: TEXT only
- Retry attempts: 10
- Retry base: 2
- Initial delay: 0.5s
- Max delay: 30s
- Retry codes: 429, 500, 503, 504, 408

### Asynchronous Execution

The application uses `nest_asyncio` to handle nested event loops in Streamlit:
```python
nest_asyncio.apply()  # Allows ADK's async operations within Streamlit
```

## Key Features

- **Automatic cluster optimization**: No manual k selection required
- **Multi-metric validation**: Four clustering quality metrics
- **Contextual strategies**: Marketing recommendations based on actual cluster characteristics
- **Conversational interface**: Natural language Q&A about results
- **Export flexibility**: Multiple output formats for different stakeholders
- **Production-ready logging**: Full observability and debugging support
- **Error handling**: Graceful degradation with informative error messages

## Limitations

- Maximum cluster range: k=2 to k=8 (configurable)
- Large datasets (>10,000 rows) use MiniBatch sampling for speed
- Requires Google API key and internet connectivity
- Categorical features are label-encoded (ordinal encoding assumed)

## Future Enhancements

- Support for hierarchical clustering
- Interactive cluster visualization (UMAP/t-SNE plots)
- A/B testing recommendations
- Integration with CRM systems
- Custom metric weighting for cluster selection
- Multi-language support for global markets

## License

MIT License

## Acknowledgments

- Google ADK (Agent Development Kit) for the agentic framework
- Google Gemini 2.5 Flash for LLM capabilities
- Streamlit for the web interface
- scikit-learn for clustering algorithms
