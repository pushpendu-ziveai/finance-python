# Python Finance PDF Analysis

## Overview
FastAPI-based web application for analyzing PDF bank statements using AWS Textract and Bedrock Claude AI.

### Converted from Node.js
This is a complete Python conversion of the original Node.js finance application, maintaining all functionality:

**Technology Migration:**
- Express.js → FastAPI
- AWS SDK v3 → boto3
- multer → python-multipart
- express-session → starlette SessionMiddleware
- All PDF processing and AI logic preserved

## Features

✅ **Multi-Strategy PDF Extraction**
- Direct Textract table analysis (preserves structure)
- Fallback text detection (more reliable)
- Page-by-page processing (last resort)

✅ **Advanced Transaction Parsing**
- Multi-page document support
- Advanced table relationship handling
- Enhanced text pattern matching
- Per-page transaction tracking

✅ **AI-Powered Analysis**
- AWS Bedrock Claude 3 Haiku integration
- Natural language queries about transactions
- Formatted responses with proper structure

✅ **Modern Web Interface**
- FastAPI with automatic API documentation
- Bootstrap 5 responsive design
- Real-time upload progress
- Session-based transaction storage

## Quick Start

### 1. Setup Environment

```bash
# Clone or create project directory
cd finance-python

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure AWS

Create `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your AWS credentials:
```
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-s3-bucket
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
FASTAPI_SECRET_KEY=your-session-secret
```

### 3. Run Application

```bash
# Development with auto-reload
python app.py

# Or with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 3000 --reload
```

### 4. Access Application

- **Main App:** http://localhost:3000
- **API Docs:** http://localhost:3000/docs
- **Health Check:** http://localhost:3000/health

## Project Structure

```
finance-python/
├── app.py                     # Main FastAPI application
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── README.md                 # This file
│
├── libs/                     # Core libraries
│   ├── __init__.py
│   ├── extract_pdf.py        # Multi-strategy PDF extraction
│   ├── parse_transactions.py # Transaction parsing logic  
│   └── query_agent.py        # AI query handling
│
├── templates/                # Jinja2 HTML templates
│   ├── index.html           # Upload interface
│   └── query.html           # Query interface
│
├── static/                  # Static files (CSS, JS, images)
├── uploads/                 # Temporary file storage
└── temp_pages/             # Temporary page processing
```

## API Endpoints

### Core Endpoints
- `GET /` - Upload page
- `GET /query` - Query page  
- `POST /upload` - File upload and processing
- `POST /query` - AI query processing

### API Endpoints
- `GET /api/transactions` - Get stored transactions
- `DELETE /api/transactions` - Clear stored transactions
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## Usage Examples

### 1. Upload PDF
```bash
curl -X POST "http://localhost:3000/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@statement.pdf"
```

### 2. Query Transactions
```bash
curl -X POST "http://localhost:3000/query" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "query=What was my total spending?"
```

### 3. Get Transactions via API
```bash
curl -X GET "http://localhost:3000/api/transactions"
```

## AWS Services Required

1. **S3 Bucket** - File storage
2. **Textract** - PDF text extraction
3. **Bedrock** - AI queries (Claude access)

### IAM Permissions Needed

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "textract:AnalyzeDocument",
        "textract:DetectDocumentText"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow", 
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
    }
  ]
}
```

## Development

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run with Debug Logging
```bash
LOG_LEVEL=DEBUG python app.py
```

### Run Tests (if available)
```bash
python -m pytest
```

## Deployment

### Production with Gunicorn
```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:3000
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 3000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]
```

## Migration Notes

**From Node.js to Python:**
- ✅ All extraction strategies preserved
- ✅ Multi-page processing logic maintained  
- ✅ AI query functionality equivalent
- ✅ Session management converted
- ✅ Error handling improved
- ✅ API documentation added (FastAPI benefit)
- ✅ Type hints throughout codebase

## Troubleshooting

### Common Issues

1. **Import Errors** - Ensure virtual environment is activated
2. **AWS Credentials** - Check `.env` file and IAM permissions
3. **PDF Processing** - Verify S3 bucket access and Textract limits
4. **AI Queries** - Confirm Bedrock access and Claude model availability

### Logging

Enable debug logging in `.env`:
```
LOG_LEVEL=DEBUG
```

All extraction steps are logged for troubleshooting multi-page processing issues.

## License

Same as original project.