from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import os
import asyncio
import aiofiles
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Import our custom modules
from libs.extract_pdf import extract_pdf
from libs.parse_transactions import parse_transactions
from libs.query_agent import query_agent

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Finance PDF Analysis API",
    description="Analyze PDF bank statements using AWS Textract and Bedrock",
    version="2.0.0"
)

# Add session middleware
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.getenv("FASTAPI_SECRET_KEY", "finance-app-secret-key-python")
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Supported file types
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file(file: UploadFile) -> bool:
    """Validate uploaded file type and size"""
    if not file.filename:
        return False
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    
    return True

async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to disk and return the file path"""
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(file.filename).stem
    file_ext = Path(file.filename).suffix
    filename = f"{original_name}_{timestamp}{file_ext}"
    
    file_path = UPLOAD_DIR / filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return str(file_path)

async def upload_to_s3(file_path: str, s3_key: str, content_type: str) -> None:
    """Upload file to S3 bucket"""
    try:
        with open(file_path, 'rb') as file_data:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=s3_key,
                Body=file_data,
                ContentType=content_type
            )
        logger.info(f"File uploaded to S3: {S3_BUCKET_NAME}/{s3_key}")
    except ClientError as e:
        logger.error(f"S3 upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

def cleanup_file(file_path: str) -> None:
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main upload page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/query", response_class=HTMLResponse)
async def query_page(request: Request):
    """Serve the query page"""
    transactions = request.session.get("transactions", [])
    return templates.TemplateResponse("query.html", {
        "request": request, 
        "transactions": transactions
    })

@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """Handle PDF upload and processing"""
    
    # Validate file
    if not validate_file(file):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload PDF, JPG, JPEG, or PNG files only."
        )
    
    # Check file size
    file_size = 0
    content = await file.read()
    file_size = len(content)
    await file.seek(0)  # Reset file pointer
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, 
            detail="File size exceeds 10MB limit."
        )
    
    local_file_path = None
    
    try:
        # Save uploaded file locally
        local_file_path = await save_uploaded_file(file)
        logger.info(f"File saved locally: {local_file_path}")
        
        # Generate S3 key
        s3_key = f"uploads/{Path(local_file_path).name}"
        content_type = file.content_type or "application/pdf"
        
        # Upload to S3
        await upload_to_s3(local_file_path, s3_key, content_type)
        
        # Extract text using Textract
        logger.info("Starting PDF extraction with Textract...")
        textract_data = await extract_pdf(S3_BUCKET_NAME, s3_key)
        
        # Debug logging
        if textract_data and textract_data.get('Blocks'):
            logger.info(f"\n=== TEXTRACT EXTRACTION SUMMARY ===")
            logger.info(f"Total blocks received: {len(textract_data['Blocks'])}")
            
            block_types = {}
            for block in textract_data['Blocks']:
                block_type = block['BlockType']
                block_types[block_type] = block_types.get(block_type, 0) + 1
            
            logger.info(f"Block types: {block_types}")
            logger.info(f"Pages in document: {textract_data.get('DocumentMetadata', {}).get('Pages', 'unknown')}")
            logger.info("================================\n")
        
        # Check if extraction was successful
        if not textract_data:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract text from the uploaded document. Please try a different file or format."
            )
        
        # Parse transactions
        logger.info("Parsing transactions from extracted text...")
        transactions = parse_transactions(textract_data)
        
        # Store in session
        request.session["transactions"] = transactions
        request.session["extraction_info"] = {
            "filename": file.filename,
            "total_transactions": len(transactions),
            "extracted_at": datetime.now().isoformat()
        }
        
        logger.info(f"Successfully processed {len(transactions)} transactions")
        
        return JSONResponse({
            "success": True,
            "message": f"Successfully extracted {len(transactions)} transactions from your PDF!",
            "transaction_count": len(transactions),
            "redirect": "/query"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up local file
        if local_file_path:
            cleanup_file(local_file_path)

@app.post("/query")
async def handle_query(request: Request, query: str = Form(...)):
    """Handle AI query about transactions"""
    
    transactions = request.session.get("transactions", [])
    
    if not transactions:
        raise HTTPException(
            status_code=400, 
            detail="No transactions found. Please upload a PDF first."
        )
    
    try:
        logger.info(f"Processing query: {query}")
        response = await query_agent(query, transactions)
        
        return JSONResponse({
            "success": True,
            "query": query,
            "response": response,
            "transaction_count": len(transactions)
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/api/transactions")
async def get_transactions(request: Request):
    """Get stored transactions"""
    transactions = request.session.get("transactions", [])
    extraction_info = request.session.get("extraction_info", {})
    
    return JSONResponse({
        "transactions": transactions,
        "info": extraction_info,
        "count": len(transactions)
    })

@app.delete("/api/transactions")
async def clear_transactions(request: Request):
    """Clear stored transactions"""
    request.session.pop("transactions", None)
    request.session.pop("extraction_info", None)
    
    return JSONResponse({
        "success": True,
        "message": "Transactions cleared"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "service": "finance-pdf-analysis",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 3000
    port = int(os.getenv("PORT", 3000))
    
    logger.info(f"Starting FastAPI server on http://localhost:{port}")
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=port, 
        reload=True,
        log_level="info"
    )