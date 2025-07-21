from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import json
import threading
from typing import Optional, List, Dict
import logging
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SQL Optimizer", version="1.0.0")

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread lock for file operations
file_lock = threading.Lock()

# Data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SAVED_QUERIES_FILE = DATA_DIR / "saved_queries.json"


# Request/Response models
class SQLRequest(BaseModel):
    query: str


class SQLResponse(BaseModel):
    original_query: str
    optimized_query: str
    explanation: str
    query_plan: Optional[str] = None
    optimization_score: str


class SQLSaveRequest(BaseModel):
    title: str
    group: str
    original_query: str
    optimized_query: str
    explanation: str
    query_plan: Optional[str] = None
    optimization_score: str


class SavedQuery(BaseModel):
    title: str
    original_query: str
    optimized_query: str
    explanation: str
    query_plan: Optional[str] = None
    optimization_score: str


# Initialize OpenAI client
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in your .env file"
        )
    try:
        # Initialize with only the API key - no other parameters
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        # Try alternative initialization methods
        try:
            # Alternative method without any extra parameters
            import openai

            openai.api_key = api_key
            return OpenAI()
        except Exception as e2:
            logger.error(f"Alternative OpenAI client initialization also failed: {e2}")
            raise ValueError(f"Cannot initialize OpenAI client: {e}")


# Test OpenAI client initialization at startup
def test_openai_connection():
    """Test if OpenAI client can be initialized properly"""
    try:
        test_client = get_openai_client()
        logger.info("OpenAI client initialized successfully")
        return True
    except Exception as e:
        logger.error(f"OpenAI client test failed: {e}")
        return False


# Initialize client as None, will be created on first use
client = None


def sanitize_sql(query: str) -> str:
    """Basic SQL sanitization - removes dangerous patterns"""
    # Remove comments
    query = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)

    # Remove potentially dangerous keywords (for display purposes)
    dangerous_patterns = [
        r"\bDROP\s+TABLE\b",
        r"\bDROP\s+DATABASE\b",
        r"\bTRUNCATE\b",
        r"\bSHUTDOWN\b",
        r"\bEXEC\b",
        r"\bxp_\w+\b",
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            logger.warning(f"Potentially dangerous SQL pattern detected: {pattern}")

    return query.strip()


def get_sql_optimization_prompt(query: str) -> str:
    """Generate the optimization prompt for the LLM"""
    return f"""
You are a senior database optimization expert. Analyze the following SQL query and provide optimizations.

ORIGINAL QUERY:
{query}

Please provide your response in the following JSON format:
{{
    "optimized_query": "The optimized SQL query here",
    "explanation": "Concise explanation as bullet points. Use • prefix for each point.",
    "query_plan": "For SELECT queries: Brief execution plan with estimated steps and performance impact. For other queries: indexing recommendations.",
    "optimization_score": "Score from 1-10 with brief justification (e.g., '8/10 - Significant JOIN optimization applied')"
}}

OPTIMIZATION GUIDELINES:
1. **Performance Impact**: Focus on changes that provide the biggest performance gains
2. **Query Structure**: Convert subqueries to JOINs, optimize WHERE clauses, remove SELECT *
3. **Index Strategy**: Suggest specific indexes that would improve this query
4. **Readability**: Maintain clean, readable SQL structure
5. **Practical Changes**: Only suggest optimizations that are actually implementable

EXPLANATION FORMAT:
- Format as bullet points with • prefix
- Keep explanations concise (2-3 sentences max per point)
- Focus on WHY the change improves performance
- Avoid overly technical jargon

QUERY PLAN FORMAT:
- For SELECT: Show logical execution order with estimated cost impact
- For INSERT/UPDATE/DELETE: Focus on index recommendations
- Include performance estimates when possible (e.g., "reduces table scans by 80%")

RULES:
- Keep the query functionally equivalent
- Use generic SQL syntax (not database-specific)
- If query is already optimal, explain why and give suggestions for monitoring
- Provide actionable, specific recommendations
"""


async def optimize_sql_with_llm(query: str) -> dict:
    """Call OpenAI API to optimize the SQL query"""
    global client

    # Initialize client if not already done
    if client is None:
        try:
            client = get_openai_client()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise HTTPException(
                status_code=500, detail=f"OpenAI client initialization error: {str(e)}"
            )

    try:
        # Use a more compatible model name
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a SQL optimization expert. Always respond with valid JSON.",
                },
                {"role": "user", "content": get_sql_optimization_prompt(query)},
            ],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
        )

        result = response.choices[0].message.content

        # Parse JSON response
        import json

        try:
            if result is None:
                raise ValueError("Empty response from OpenAI")

            # Clean the response if it has markdown formatting
            if result.startswith("```json"):
                result = result.strip("```json").strip("```").strip()
            elif result.startswith("```"):
                result = result.strip("```").strip()

            parsed_result = json.loads(result)
            return parsed_result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Raw response: {result}")
            # Fallback if JSON parsing fails
            return {
                "optimized_query": query,
                "explanation": f"• Unable to parse optimization response\n• The query appears to be acceptable as-is\n• Error: {str(e)}",
                "query_plan": None,
                "optimization_score": "5/10 - Unable to analyze due to parsing error",
            }

    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Optimization service error: {str(e)}"
        )


def load_saved_queries() -> Dict[str, List[SavedQuery]]:
    """Load saved queries from JSON file"""
    with file_lock:
        try:
            if SAVED_QUERIES_FILE.exists():
                with open(SAVED_QUERIES_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert to SavedQuery objects
                    result = {}
                    for group, queries in data.items():
                        result[group] = [SavedQuery(**query) for query in queries]
                    return result
            return {}
        except Exception as e:
            logger.error(f"Error loading saved queries: {e}")
            return {}


def save_queries_to_file(queries: Dict[str, List[SavedQuery]]):
    """Save queries to JSON file"""
    with file_lock:
        try:
            # Convert SavedQuery objects to dict
            data = {}
            for group, query_list in queries.items():
                data[group] = [query.dict() for query in query_list]

            with open(SAVED_QUERIES_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving queries: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving query: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SQL Query Optimizer</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            'code-bg': '#1e293b',
                            'code-text': '#e2e8f0'
                        },
                        animation: {
                            'fade-in': 'fadeIn 0.5s ease-in-out',
                            'slide-down': 'slideDown 0.3s ease-out'
                        }
                    }
                }
            }
        </script>
        <style>
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes slideDown {
                from { opacity: 0; max-height: 0; }
                to { opacity: 1; max-height: 500px; }
            }
            .tooltip {
                position: relative;
            }
            .tooltip::after {
                content: attr(data-tooltip);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                background: #374151;
                color: white;
                padding: 0.5rem;
                border-radius: 0.25rem;
                font-size: 0.75rem;
                white-space: nowrap;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.3s;
                z-index: 1000;
            }
            .tooltip:hover::after {
                opacity: 1;
            }
        </style>
    </head>
    <body class="bg-gray-50 min-h-screen">
        <div class="container mx-auto px-4 py-8 max-w-7xl">
            <!-- Header -->
            <div class="text-center mb-12">
                <h1 class="text-5xl font-bold text-gray-900 mb-4">SQL Query Optimizer</h1>
                <p class="text-lg text-gray-500 max-w-2xl mx-auto">
                    Optimize your SQL queries with AI-powered suggestions and save them for future reference
                </p>
            </div>

            <!-- Main Content -->
            <div class="grid grid-cols-1 xl:grid-cols-3 gap-8">
                <!-- Input Section -->
                <div class="xl:col-span-2 bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-6 text-gray-800">Input SQL Query</h2>
                    
                    <!-- Saved Queries Loader -->
                    <div class="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                        <h3 class="text-sm font-semibold text-blue-900 mb-3">Load Saved Query</h3>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                            <select id="groupSelect" class="px-3 py-2 border border-blue-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                                <option value="">Select Group...</option>
                            </select>
                            <select id="querySelect" class="px-3 py-2 border border-blue-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent" disabled>
                                <option value="">Select Query...</option>
                            </select>
                        </div>
                        <button id="loadQueryBtn" class="mt-3 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed" disabled onclick="loadSelectedQuery()">
                            Load Query
                        </button>
                    </div>

                    <textarea 
                        id="sqlInput" 
                        placeholder="Enter your SQL query here...

Example:
SELECT * FROM users u, orders o 
WHERE u.id = o.user_id 
AND u.status = 'active'
ORDER BY u.created_at"
                        class="w-full h-64 p-4 border border-gray-300 rounded-md font-mono text-sm bg-code-bg text-code-text resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    ></textarea>
                    
                    <div class="mt-4 flex space-x-3">
                        <button 
                            id="optimizeBtn" 
                            class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-md transition-colors duration-200 flex items-center space-x-2"
                            onclick="optimizeQuery()"
                        >
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path>
                            </svg>
                            <span>Optimize Query</span>
                        </button>
                        
                        <button 
                            id="clearBtn" 
                            class="bg-gray-500 hover:bg-gray-600 text-white px-6 py-2 rounded-md transition-colors duration-200"
                            onclick="clearAll()"
                        >
                            Clear
                        </button>
                    </div>
                </div>

                <!-- Saved Queries Sidebar -->
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-6 text-gray-800">Saved Queries</h2>
                    <div id="savedQueriesList" class="space-y-3">
                        <div class="text-gray-500 text-center py-8">
                            <svg class="w-12 h-12 mx-auto mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>
                            </svg>
                            <p class="text-sm">No saved queries yet</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Output Section -->
            <div class="mt-8 bg-white rounded-lg shadow-lg p-6">
                <h2 class="text-xl font-semibold mb-6 text-gray-800">Optimization Results</h2>
                <div id="outputSection" class="space-y-4">
                    <div class="text-gray-500 text-center py-12">
                        <svg class="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                        </svg>
                        <p>Enter a SQL query and click "Optimize Query" to see results</p>
                    </div>
                </div>
            </div>

            <!-- Loading Overlay -->
            <div id="loadingOverlay" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 hidden">
                <div class="bg-white p-8 rounded-lg shadow-xl text-center">
                    <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p class="text-gray-700">Optimizing your query...</p>
                </div>
            </div>

            <!-- Save Modal -->
            <div id="saveModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 hidden">
                <div class="bg-white p-8 rounded-lg shadow-xl w-full max-w-md mx-4">
                    <h3 class="text-lg font-semibold mb-4 text-gray-900">Save Optimized Query</h3>
                    <div class="space-y-4">
                        <div>
                            <label for="saveTitle" class="block text-sm font-medium text-gray-700 mb-1">Title</label>
                            <input type="text" id="saveTitle" placeholder="e.g., User Login Query" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                        </div>
                        <div>
                            <label for="saveGroup" class="block text-sm font-medium text-gray-700 mb-1">Group</label>
                            <input type="text" id="saveGroup" placeholder="e.g., Auth Queries" class="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent" list="groupSuggestions">
                            <datalist id="groupSuggestions"></datalist>
                        </div>
                    </div>
                    <div class="mt-6 flex space-x-3">
                        <button onclick="saveQuery()" class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md transition-colors duration-200 flex-1">
                            Save Query
                        </button>
                        <button onclick="closeSaveModal()" class="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-md transition-colors duration-200">
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentOptimizationResult = null;

            async function optimizeQuery() {
                const query = document.getElementById('sqlInput').value.trim();
                
                if (!query) {
                    alert('Please enter a SQL query');
                    return;
                }

                showLoading(true);
                
                try {
                    const response = await fetch('/optimize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: query })
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'HTTP error! status: ' + response.status);
                    }

                    const result = await response.json();
                    currentOptimizationResult = result;
                    displayResults(result);
                } catch (error) {
                    console.error('Error:', error);
                    displayError(error.message);
                } finally {
                    showLoading(false);
                }
            }

            function displayResults(result) {
                const outputSection = document.getElementById('outputSection');
                
                // Escape HTML characters to prevent XSS
                function escapeHtml(text) {
                    const div = document.createElement('div');
                    div.textContent = text;
                    return div.innerHTML;
                }

                // Convert bullet point text to HTML list
                function formatExplanation(text) {
                    if (!text) return '';
                    const lines = text.split('\n').filter(line => line.trim());
                    const listItems = lines.map(line => {
                        const cleaned = line.replace(/^[•·\-\*]\s*/, '').trim();
                        return cleaned ? '<li class="mb-2">' + escapeHtml(cleaned) + '</li>' : '';
                    }).filter(item => item);
                    
                    return listItems.length > 0 ? '<ul class="list-disc list-inside space-y-2">' + listItems.join('') + '</ul>' : escapeHtml(text);
                }

                const queryPlanSection = result.query_plan ? 
                    '<button onclick="switchTab(\'plan\')" id="plan-tab" class="tab-button border-b-2 border-transparent py-2 px-1 text-sm font-medium text-gray-500 hover:text-gray-700">' +
                        'Execution Plan' +
                    '</button>' : '';

                const queryPlanContent = result.query_plan ? 
                    '<!-- Plan Tab -->' +
                    '<div id="plan-content" class="tab-pane hidden">' +
                        '<div class="bg-gray-50 border border-gray-200 rounded-lg">' +
                            '<button onclick="toggleExecutionPlan()" class="w-full px-4 py-3 text-left font-medium text-gray-700 hover:bg-gray-100 rounded-t-lg border-b border-gray-200 flex items-center justify-between">' +
                                '<span>Execution Plan Details</span>' +
                                '<svg id="planToggleIcon" class="w-5 h-5 transform transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                                    '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path>' +
                                '</svg>' +
                            '</button>' +
                            '<div id="planContent" class="hidden p-4 bg-green-50 border-green-200 animate-slide-down">' +
                                '<pre class="text-sm text-green-800 whitespace-pre-wrap overflow-x-hidden">' + escapeHtml(result.query_plan) + '</pre>' +
                            '</div>' +
                        '</div>' +
                    '</div>' : '';

                // Parse score for highlighting
                const scoreText = result.optimization_score;
                const scoreMatch = scoreText.match(/(\d+)\/10/);
                const scoreValue = scoreMatch ? parseInt(scoreMatch[1]) : 5;
                const scoreColor = scoreValue >= 8 ? 'green' : scoreValue >= 6 ? 'yellow' : 'red';
                const scoreBgClass = scoreColor === 'green' ? 'bg-green-100 border-green-300' : 
                                   scoreColor === 'yellow' ? 'bg-yellow-100 border-yellow-300' : 
                                   'bg-red-100 border-red-300';
                const scoreTextClass = scoreColor === 'green' ? 'text-green-800' : 
                                      scoreColor === 'yellow' ? 'text-yellow-800' : 
                                      'text-red-800';

                outputSection.innerHTML = 
                    '<div class="space-y-6 animate-fade-in">' +
                        '<!-- Header with Score and Action Buttons -->' +
                        '<div class="flex items-center justify-between ' + scoreBgClass + ' border rounded-lg p-6">' +
                            '<div>' +
                                '<h3 class="font-semibold ' + scoreTextClass + ' mb-2 text-lg">Optimization Score</h3>' +
                                '<p class="' + scoreTextClass + ' font-semibold text-lg">✅ ' + escapeHtml(result.optimization_score) + '</p>' +
                            '</div>' +
                            '<div class="flex space-x-3">' +
                                '<button onclick="copyToClipboard()" class="tooltip bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm transition-colors duration-200 flex items-center space-x-2" data-tooltip="Copy optimized query">' +
                                    '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                                        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>' +
                                    '</svg>' +
                                    '<span>Copy Query</span>' +
                                '</button>' +
                                '<button onclick="openSaveModal()" class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md text-sm transition-colors duration-200 flex items-center space-x-2">' +
                                    '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                                        '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"></path>' +
                                    '</svg>' +
                                    '<span>Save Query</span>' +
                                '</button>' +
                            '</div>' +
                        '</div>' +

                        '<!-- Tabs for Different Views -->' +
                        '<div class="border-b border-gray-200">' +
                            '<nav class="-mb-px flex space-x-8">' +
                                '<button onclick="switchTab(\'query\')" id="query-tab" class="tab-button active border-b-2 border-blue-500 py-2 px-1 text-sm font-medium text-blue-600">' +
                                    'Optimized Query' +
                                '</button>' +
                                '<button onclick="switchTab(\'explanation\')" id="explanation-tab" class="tab-button border-b-2 border-transparent py-2 px-1 text-sm font-medium text-gray-500 hover:text-gray-700">' +
                                    'Key Changes' +
                                '</button>' +
                                queryPlanSection +
                            '</nav>' +
                        '</div>' +

                        '<!-- Tab Content -->' +
                        '<div class="tab-content">' +
                            '<!-- Query Tab -->' +
                            '<div id="query-content" class="tab-pane active">' +
                                '<pre class="bg-code-bg text-code-text p-4 rounded-md text-sm border overflow-x-hidden whitespace-pre-wrap">' + escapeHtml(result.optimized_query) + '</pre>' +
                            '</div>' +

                            '<!-- Explanation Tab -->' +
                            '<div id="explanation-content" class="tab-pane hidden">' +
                                '<div class="bg-amber-50 border border-amber-200 rounded-lg p-6">' +
                                    '<h4 class="font-semibold text-amber-900 mb-4">Optimization Improvements</h4>' +
                                    '<div class="prose prose-sm max-w-none text-amber-800">' +
                                        formatExplanation(result.explanation) +
                                    '</div>' +
                                '</div>' +
                            '</div>' +

                            queryPlanContent +
                        '</div>' +
                    '</div>';
            }

            function displayError(message) {
                const outputSection = document.getElementById('outputSection');
                outputSection.innerHTML = 
                    '<div class="bg-red-50 border border-red-200 rounded-lg p-4 animate-fade-in">' +
                        '<div class="flex items-center">' +
                            '<svg class="w-5 h-5 text-red-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                                '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>' +
                            '</svg>' +
                            '<h3 class="font-semibold text-red-900">Error</h3>' +
                        '</div>' +
                        '<p class="text-red-800 mt-2">' + message + '</p>' +
                    '</div>';
            }

            function toggleExecutionPlan() {
                const content = document.getElementById('planContent');
                const icon = document.getElementById('planToggleIcon');
                
                if (content.classList.contains('hidden')) {
                    content.classList.remove('hidden');
                    icon.style.transform = 'rotate(180deg)';
                } else {
                    content.classList.add('hidden');
                    icon.style.transform = 'rotate(0deg)';
                }
            }

            function switchTab(tabName) {
                // Hide all tab content
                document.querySelectorAll('.tab-pane').forEach(pane => {
                    pane.classList.add('hidden');
                    pane.classList.remove('active');
                });
                
                // Remove active class from all tab buttons
                document.querySelectorAll('.tab-button').forEach(button => {
                    button.classList.remove('active', 'border-blue-500', 'text-blue-600');
                    button.classList.add('border-transparent', 'text-gray-500');
                });
                
                // Show selected tab content
                const contentElement = document.getElementById(tabName + '-content');
                if (contentElement) {
                    contentElement.classList.remove('hidden');
                    contentElement.classList.add('active');
                }
                
                // Activate selected tab button
                const activeButton = document.getElementById(tabName + '-tab');
                if (activeButton) {
                    activeButton.classList.add('active', 'border-blue-500', 'text-blue-600');
                    activeButton.classList.remove('border-transparent', 'text-gray-500');
                }
            }

            function copyToClipboard() {
                const queryContent = document.querySelector('#query-content pre');
                if (queryContent) {
                    const textToCopy = queryContent.textContent;
                    navigator.clipboard.writeText(textToCopy).then(() => {
                        // Show temporary success message
                        const button = document.querySelector('button[onclick="copyToClipboard()"]');
                        const originalContent = button.innerHTML;
                        button.innerHTML = 
                            '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                                '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>' +
                            '</svg>' +
                            '<span>Copied!</span>';
                        button.classList.remove('bg-blue-600', 'hover:bg-blue-700');
                        button.classList.add('bg-green-600');
                        setTimeout(() => {
                            button.innerHTML = originalContent;
                            button.classList.remove('bg-green-600');
                            button.classList.add('bg-blue-600', 'hover:bg-blue-700');
                        }, 2000);
                    }).catch(err => {
                        console.error('Failed to copy text: ', err);
                        alert('Failed to copy to clipboard');
                    });
                }
            }

            function clearAll() {
                document.getElementById('sqlInput').value = '';
                document.getElementById('outputSection').innerHTML = 
                    '<div class="text-gray-500 text-center py-12">' +
                        '<svg class="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                            '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>' +
                        '</svg>' +
                        '<p>Enter a SQL query and click "Optimize Query" to see results</p>' +
                    '</div>';
                currentOptimizationResult = null;
            }

            function showLoading(show) {
                const overlay = document.getElementById('loadingOverlay');
                const button = document.getElementById('optimizeBtn');
                
                if (show) {
                    overlay.classList.remove('hidden');
                    button.disabled = true;
                    button.classList.add('opacity-50', 'cursor-not-allowed');
                } else {
                    overlay.classList.add('hidden');
                    button.disabled = false;
                    button.classList.remove('opacity-50', 'cursor-not-allowed');
                }
            }

            // Save Modal Functions
            function openSaveModal() {
                if (!currentOptimizationResult) {
                    alert('Please optimize a query first');
                    return;
                }
                document.getElementById('saveModal').classList.remove('hidden');
                // Load existing groups for suggestions
                loadGroupSuggestions();
            }

            function closeSaveModal() {
                document.getElementById('saveModal').classList.add('hidden');
                document.getElementById('saveTitle').value = '';
                document.getElementById('saveGroup').value = '';
            }

            async function saveQuery() {
                const title = document.getElementById('saveTitle').value.trim();
                const group = document.getElementById('saveGroup').value.trim();

                if (!title) {
                    alert('Please enter a title for the query');
                    return;
                }

                if (!group) {
                    alert('Please enter a group for the query');
                    return;
                }

                if (!currentOptimizationResult) {
                    alert('No optimization result to save');
                    return;
                }

                try {
                    const saveData = {
                        title: title,
                        group: group,
                        original_query: currentOptimizationResult.original_query,
                        optimized_query: currentOptimizationResult.optimized_query,
                        explanation: currentOptimizationResult.explanation,
                        query_plan: currentOptimizationResult.query_plan,
                        optimization_score: currentOptimizationResult.optimization_score
                    };

                    const response = await fetch('/save', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(saveData)
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Failed to save query');
                    }

                    closeSaveModal();
                    loadSavedQueries();
                    loadGroups();
                    
                    // Show success message
                    const successDiv = document.createElement('div');
                    successDiv.className = 'fixed top-4 right-4 bg-green-600 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-fade-in';
                    successDiv.innerHTML = 
                        '<div class="flex items-center space-x-2">' +
                            '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                                '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>' +
                            '</svg>' +
                            '<span>Query saved successfully!</span>' +
                        '</div>';
                    document.body.appendChild(successDiv);
                    setTimeout(() => {
                        successDiv.remove();
                    }, 3000);

                } catch (error) {
                    console.error('Error saving query:', error);
                    alert('Error saving query: ' + error.message);
                }
            }

            // Load functions
            async function loadGroups() {
                try {
                    const response = await fetch('/groups');
                    const groups = await response.json();
                    
                    const groupSelect = document.getElementById('groupSelect');
                    groupSelect.innerHTML = '<option value="">Select Group...</option>';
                    
                    groups.forEach(group => {
                        const option = document.createElement('option');
                        option.value = group;
                        option.textContent = group;
                        groupSelect.appendChild(option);
                    });
                } catch (error) {
                    console.error('Error loading groups:', error);
                }
            }

            async function loadGroupSuggestions() {
                try {
                    const response = await fetch('/groups');
                    const groups = await response.json();
                    
                    const datalist = document.getElementById('groupSuggestions');
                    datalist.innerHTML = '';
                    
                    groups.forEach(group => {
                        const option = document.createElement('option');
                        option.value = group;
                        datalist.appendChild(option);
                    });
                } catch (error) {
                    console.error('Error loading group suggestions:', error);
                }
            }

            async function loadQueriesForGroup(group) {
                if (!group) {
                    const querySelect = document.getElementById('querySelect');
                    querySelect.innerHTML = '<option value="">Select Query...</option>';
                    querySelect.disabled = true;
                    document.getElementById('loadQueryBtn').disabled = true;
                    return;
                }

                try {
                    const response = await fetch('/queries?group=' + encodeURIComponent(group));
                    const queries = await response.json();
                    
                    const querySelect = document.getElementById('querySelect');
                    querySelect.innerHTML = '<option value="">Select Query...</option>';
                    
                    queries.forEach(function(query) {
                        const option = document.createElement('option');
                        option.value = query.title;
                        option.textContent = query.title;
                        querySelect.appendChild(option);
                    });
                    
                    querySelect.disabled = false;
                } catch (error) {
                    console.error('Error loading queries:', error);
                }
            }

            async function loadSelectedQuery() {
                const group = document.getElementById('groupSelect').value;
                const title = document.getElementById('querySelect').value;
                
                if (!group || !title) {
                    alert('Please select both group and query');
                    return;
                }

                try {
                    const response = await fetch('/query?group=' + encodeURIComponent(group) + '&title=' + encodeURIComponent(title));
                    const query = await response.json();
                    
                    document.getElementById('sqlInput').value = query.original_query;
                    
                    // Optionally display the optimization result immediately
                    currentOptimizationResult = query;
                    displayResults(query);
                    
                } catch (error) {
                    console.error('Error loading query:', error);
                    alert('Error loading query: ' + error.message);
                }
            }

            async function loadSavedQueries() {
                try {
                    const response = await fetch('/groups');
                    const groups = await response.json();
                    
                    const savedQueriesList = document.getElementById('savedQueriesList');
                    
                    if (groups.length === 0) {
                        savedQueriesList.innerHTML = 
                            '<div class="text-gray-500 text-center py-8">' +
                                '<svg class="w-12 h-12 mx-auto mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">' +
                                    '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"></path>' +
                                '</svg>' +
                                '<p class="text-sm">No saved queries yet</p>' +
                            '</div>';
                        return;
                    }

                    let html = '';
                    for (const group of groups) {
                        const queriesResponse = await fetch('/queries?group=' + encodeURIComponent(group));
                        const queries = await queriesResponse.json();
                        
                        html += 
                            '<div class="mb-4">' +
                                '<h4 class="font-semibold text-gray-800 mb-2 text-sm">' + group + '</h4>' +
                                '<div class="space-y-1">';
                        
                        queries.forEach(function(query) {
                            // Escape single quotes in group and title for onclick
                            const escapedGroup = group.replace(/'/g, "\\'");
                            const escapedTitle = query.title.replace(/'/g, "\\'");
                            
                            html += 
                                '<div class="flex items-center justify-between p-2 bg-gray-50 rounded text-xs hover:bg-gray-100">' +
                                    '<span class="truncate flex-1 mr-2" title="' + query.title + '">' + query.title + '</span>' +
                                    '<button onclick="loadSpecificQuery(\'' + escapedGroup + '\', \'' + escapedTitle + '\')" class="text-blue-600 hover:text-blue-800 text-xs px-2 py-1 rounded hover:bg-blue-50">' +
                                        'Load' +
                                    '</button>' +
                                '</div>';
                        });
                        
                        html += 
                                '</div>' +
                            '</div>';
                    }
                    
                    savedQueriesList.innerHTML = html;
                    
                } catch (error) {
                    console.error('Error loading saved queries:', error);
                }
            }

            async function loadSpecificQuery(group, title) {
                try {
                    const response = await fetch('/query?group=' + encodeURIComponent(group) + '&title=' + encodeURIComponent(title));
                    const query = await response.json();
                    
                    document.getElementById('sqlInput').value = query.original_query;
                    currentOptimizationResult = query;
                    displayResults(query);
                    
                } catch (error) {
                    console.error('Error loading query:', error);
                    alert('Error loading query: ' + error.message);
                }
            }

            // Event listeners
            document.getElementById('groupSelect').addEventListener('change', function() {
                loadQueriesForGroup(this.value);
            });

            document.getElementById('querySelect').addEventListener('change', function() {
                document.getElementById('loadQueryBtn').disabled = !this.value;
            });

            // Allow Enter key to submit (Ctrl+Enter)
            document.getElementById('sqlInput').addEventListener('keydown', function(e) {
                if (e.ctrlKey && e.key === 'Enter') {
                    optimizeQuery();
                }
            });

            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                loadGroups();
                loadSavedQueries();
            });
        </script>
    </body>
    </html>
    """
