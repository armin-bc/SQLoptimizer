# SQL Query Optimizer

A web-based tool that accepts SQL queries and returns optimized versions with explanations using AI.

## Features

- **Multi-SQL Support**: Handles SELECT, INSERT, UPDATE, DELETE queries
- **AI-Powered Optimization**: Uses GPT-4 for intelligent query optimization
- **Detailed Explanations**: Provides clear explanations of optimization changes
- **Query Plan Simulation**: Shows simulated execution plans for SELECT queries
- **Modern UI**: Clean, responsive interface with syntax highlighting
- **Real-time Processing**: Fast optimization with loading indicators

## Quick Start

### 1. Prerequisites

- Python 3.8+
- OpenAI API key

### 2. Installation

```bash
# Clone or create project directory
mkdir sql-optimizer
cd sql-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

#### **Create a .env file (recommended for development)**

Create a `.env` file in your project directory:

```
OPENAI_API_KEY=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
```

Then install python-dotenv:

```bash
pip install python-dotenv
```

Add to the top of your `fastapi_backend.py`:

```python
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file
```

### 4. Run the Application

```bash
python main.py
```

The application will be available at: `http://localhost:8000`

## Usage

1. **Enter SQL Query**: Paste your SQL query in the input textarea
2. **Optimize**: Click "Optimize Query" or press Ctrl+Enter
3. **Review Results**: See the optimized query, explanation, and query plan
4. **Copy Results**: Use the copy button to copy the optimized query

## API Endpoints

- `GET /` - Main web interface
- `POST /optimize` - Optimize SQL query endpoint
- `GET /health` - Health check endpoint

### API Example

```bash
curl -X POST "http://localhost:8000/optimize" \
     -H "Content-Type: application/json" \
     -d '{"query": "SELECT * FROM users WHERE status = '\''active'\''"}'
```

## Example Optimizations

### Input:

```sql
SELECT * FROM users u, orders o 
WHERE u.id = o.user_id 
AND u.status = 'active'
ORDER BY u.created_at
```

### Output:

```sql
SELECT u.id, u.name, u.email, o.id, o.amount, o.created_at
FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE u.status = 'active'
ORDER BY u.created_at
```

**Optimizations Applied:**

- Replaced implicit JOIN with explicit INNER JOIN
- Removed SELECT * and specified needed columns
- Improved query readability and performance

## Security Features

- Input sanitization to remove dangerous SQL patterns
- No actual database connections (safe for testing)
- Rate limiting ready for production use
- CORS configured for local development

## Production Deployment

For production deployment, consider:

1. **Environment Variables**: Use proper environment management
2. **Authentication**: Add user authentication if needed
3. **Rate Limiting**: Implement API rate limiting
4. **Database**: Add query history storage
5. **Monitoring**: Add logging and monitoring

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML + TailwindCSS + Vanilla JavaScript
- **AI**: OpenAI GPT-4 API
- **Server**: Uvicorn ASGI server

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - feel free to use this for your projects!

## Support

For issues or questions:

- Check the logs for error details
- Ensure your OpenAI API key is valid
- Verify all dependencies are installed correctly

## Roadmap

- [ ] Support for database-specific optimizations
- [ ] Query history and favorites
- [ ] Batch query optimization
- [ ] Integration with popular databases
- [ ] Performance benchmarking
- [ ] Advanced query analysis metrics
