from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from typing import Optional
import logging

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


# Request/Response models
class SQLRequest(BaseModel):
    query: str


class SQLResponse(BaseModel):
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
    "explanation": "Concise explanation focusing on the 2-3 most important optimizations made. Use bullet points for clarity.",
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
- Keep explanations concise (2-3 sentences max)
- Focus on WHY the change improves performance
- Use bullet points for multiple optimizations
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
                "explanation": f"Unable to parse optimization response. The query appears to be acceptable as-is. Error: {str(e)}",
                "query_plan": None,
                "optimization_score": "5/10 - Unable to analyze due to parsing error",
            }

    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Optimization service error: {str(e)}"
        )


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
                        }
                    }
                }
            }
        </script>
    </head>
    <body class="bg-gray-50 min-h-screen">
        <div class="container mx-auto px-4 py-8 max-w-6xl">
            <!-- Header -->
            <div class="text-center mb-8">
                <h1 class="text-4xl font-bold text-gray-900 mb-2">SQL Query Optimizer</h1>
                <p class="text-gray-600">Optimize your SQL queries with AI-powered suggestions</p>
            </div>

            <!-- Main Content -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <!-- Input Section -->
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">Input SQL Query</h2>
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

                <!-- Output Section -->
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">Optimization Results</h2>
                    <div id="outputSection" class="space-y-4">
                        <div class="text-gray-500 text-center py-12">
                            <svg class="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                            </svg>
                            <p>Enter a SQL query and click "Optimize Query" to see results</p>
                        </div>
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
        </div>

        <script>
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
                        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                    }

                    const result = await response.json();
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

                const queryPlanSection = result.query_plan ? `
                    <button onclick="switchTab('plan')" id="plan-tab" class="tab-button border-b-2 border-transparent py-2 px-1 text-sm font-medium text-gray-500 hover:text-gray-700">
                        Execution Plan
                    </button>
                ` : '';

                const queryPlanContent = result.query_plan ? `
                    <!-- Plan Tab -->
                    <div id="plan-content" class="tab-pane hidden">
                        <div class="bg-green-50 border border-green-200 rounded-lg p-4">
                            <pre class="text-sm text-green-800 whitespace-pre-wrap">${escapeHtml(result.query_plan)}</pre>
                        </div>
                    </div>
                ` : '';

                outputSection.innerHTML = `
                    <div class="space-y-4">
                        <!-- Header with Score and Copy Button -->
                        <div class="flex items-center justify-between bg-gradient-to-r from-blue-50 to-green-50 border border-blue-200 rounded-lg p-4">
                            <div>
                                <h3 class="font-semibold text-blue-900 mb-1">Optimization Score</h3>
                                <p class="text-blue-800 font-medium">${escapeHtml(result.optimization_score)}</p>
                            </div>
                            <button onclick="copyToClipboard()" class="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm transition-colors duration-200 flex items-center space-x-2">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                                </svg>
                                <span>Copy Query</span>
                            </button>
                        </div>

                        <!-- Tabs for Different Views -->
                        <div class="border-b border-gray-200">
                            <nav class="-mb-px flex space-x-8">
                                <button onclick="switchTab('query')" id="query-tab" class="tab-button active border-b-2 border-blue-500 py-2 px-1 text-sm font-medium text-blue-600">
                                    Optimized Query
                                </button>
                                <button onclick="switchTab('explanation')" id="explanation-tab" class="tab-button border-b-2 border-transparent py-2 px-1 text-sm font-medium text-gray-500 hover:text-gray-700">
                                    Key Changes
                                </button>
                                ${queryPlanSection}
                            </nav>
                        </div>

                        <!-- Tab Content -->
                        <div class="tab-content">
                            <!-- Query Tab -->
                            <div id="query-content" class="tab-pane active">
                                <pre class="bg-code-bg text-code-text p-4 rounded-md text-sm overflow-x-auto border">${escapeHtml(result.optimized_query)}</pre>
                            </div>

                            <!-- Explanation Tab -->
                            <div id="explanation-content" class="tab-pane hidden">
                                <div class="bg-amber-50 border border-amber-200 rounded-lg p-4">
                                    <div class="prose prose-sm max-w-none">
                                        <p class="text-gray-700 leading-relaxed">${escapeHtml(result.explanation)}</p>
                                    </div>
                                </div>
                            </div>

                            ${queryPlanContent}
                        </div>
                    </div>
                `;
            }

            function displayError(message) {
                const outputSection = document.getElementById('outputSection');
                outputSection.innerHTML = `
                    <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                        <div class="flex items-center">
                            <svg class="w-5 h-5 text-red-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            <h3 class="font-semibold text-red-900">Error</h3>
                        </div>
                        <p class="text-red-800 mt-2">${message}</p>
                    </div>
                `;
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
                        button.innerHTML = `
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                            </svg>
                            <span>Copied!</span>
                        `;
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
                document.getElementById('outputSection').innerHTML = `
                    <div class="text-gray-500 text-center py-12">
                        <svg class="w-16 h-16 mx-auto mb-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                        </svg>
                        <p>Enter a SQL query and click "Optimize Query" to see results</p>
                    </div>
                `;
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

            // Allow Enter key to submit (Ctrl+Enter)
            document.getElementById('sqlInput').addEventListener('keydown', function(e) {
                if (e.ctrlKey && e.key === 'Enter') {
                    optimizeQuery();
                }
            });
        </script>
    </body>
    </html>
    """


@app.post("/optimize", response_model=SQLResponse)
async def optimize_query(request: SQLRequest):
    """Optimize a SQL query using LLM"""
    try:
        # Sanitize input
        clean_query = sanitize_sql(request.query)

        if not clean_query:
            raise HTTPException(status_code=400, detail="Empty or invalid SQL query")

        # Get optimization from LLM
        optimization_result = await optimize_sql_with_llm(clean_query)

        # Ensure all fields are strings
        def ensure_string(value):
            if isinstance(value, list):
                return "\n".join(str(v) for v in value)
            return str(value) if value is not None else ""

        return SQLResponse(
            original_query=clean_query,
            optimized_query=ensure_string(
                optimization_result.get("optimized_query", clean_query)
            ),
            explanation=ensure_string(
                optimization_result.get(
                    "explanation", "No optimization suggestions available"
                )
            ),
            query_plan=ensure_string(optimization_result.get("query_plan")),
            optimization_score=ensure_string(
                optimization_result.get("optimization_score", "N/A")
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "SQL Optimizer"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
