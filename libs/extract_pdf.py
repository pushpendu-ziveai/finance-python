import os
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import boto3
from botocore.exceptions import ClientError
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
import tempfile
import shutil

# Setup logging
logger = logging.getLogger(__name__)

# AWS Configuration
REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize AWS clients
textract_client = boto3.client('textract', region_name=REGION)
s3_client = boto3.client('s3', region_name=REGION)

async def extract_pdf(bucket_name: str, document_name: str) -> Optional[Dict[str, Any]]:
    """
    Multi-strategy PDF extraction with fallbacks
    Strategy 1: Direct AnalyzeDocument (preserves table structures)
    Strategy 2: Direct DetectDocumentText (simpler, more reliable)
    Strategy 3: Page-by-page processing (last resort)
    """
    logger.info(f"\n=== Starting PDF extraction: {bucket_name}/{document_name} ===")
    
    try:
        # Strategy 1: Try direct table/form analysis (best for structured data)
        logger.info("Strategy 1: Attempting direct AnalyzeDocument with tables...")
        direct_analyze_result = await try_direct_analyze(bucket_name, document_name)
        if direct_analyze_result and direct_analyze_result.get('Blocks', []):
            logger.info(f"✅ Strategy 1 SUCCESS: Extracted {len(direct_analyze_result['Blocks'])} blocks with table analysis")
            return direct_analyze_result
        
        # Strategy 2: Try direct text detection (more reliable for difficult PDFs)
        logger.info("Strategy 2: Attempting direct DetectDocumentText...")
        direct_text_result = await try_direct_text_detection(bucket_name, document_name)
        if direct_text_result and direct_text_result.get('Blocks', []):
            logger.info(f"✅ Strategy 2 SUCCESS: Extracted {len(direct_text_result['Blocks'])} blocks with text detection")
            return direct_text_result
        
        # Strategy 3: Page-by-page processing (last resort)
        logger.info("Strategy 3: Attempting page-by-page processing as last resort...")
        page_by_page_result = await try_page_by_page_extraction(bucket_name, document_name)
        if page_by_page_result and page_by_page_result.get('Blocks', []):
            logger.info(f"✅ Strategy 3 SUCCESS: Extracted {len(page_by_page_result['Blocks'])} blocks from page-by-page processing")
            return page_by_page_result
        
        logger.info("❌ All extraction strategies failed")
        return None
        
    except Exception as error:
        logger.error(f"❌ Fatal error in PDF extraction: {error}")
        return None

async def try_direct_analyze(bucket_name: str, document_name: str) -> Optional[Dict[str, Any]]:
    """Strategy 1: Direct AnalyzeDocument with tables and forms"""
    try:
        response = textract_client.analyze_document(
            Document={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': document_name,
                }
            },
            FeatureTypes=['TABLES', 'FORMS']
        )
        
        if response.get('Blocks', []):
            table_count = len([b for b in response['Blocks'] if b['BlockType'] == 'TABLE'])
            logger.info(f"   Direct analyze found {len(response['Blocks'])} blocks, {table_count} tables")
            return response
        
        return None
        
    except ClientError as error:
        logger.info(f"   Direct analyze failed: {error.response['Error']['Code']} - {error.response['Error']['Message']}")
        return None
    except Exception as error:
        logger.info(f"   Direct analyze failed: {type(error).__name__} - {str(error)}")
        return None

async def try_direct_text_detection(bucket_name: str, document_name: str) -> Optional[Dict[str, Any]]:
    """Strategy 2: Direct DetectDocumentText (simpler, more reliable)"""
    try:
        response = textract_client.detect_document_text(
            Document={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': document_name,
                }
            }
        )
        
        if response.get('Blocks', []):
            logger.info(f"   Direct text detection found {len(response['Blocks'])} blocks")
            return response
        
        return None
        
    except ClientError as error:
        logger.info(f"   Direct text detection failed: {error.response['Error']['Code']} - {error.response['Error']['Message']}")
        return None
    except Exception as error:
        logger.info(f"   Direct text detection failed: {type(error).__name__} - {str(error)}")
        return None

async def try_page_by_page_extraction(bucket_name: str, document_name: str) -> Optional[Dict[str, Any]]:
    """Strategy 3: Page-by-page processing (last resort)"""
    temp_dir = None
    temp_files = []
    
    try:
        logger.info("   Starting page-by-page extraction...")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='pdf_extract_')
        
        # Download PDF from S3 to local temp file
        local_pdf_path = await download_pdf_from_s3(bucket_name, document_name, temp_dir)
        temp_files.append(local_pdf_path)
        
        # Split PDF into images (better for Textract compatibility)
        page_files = await split_pdf_to_images(local_pdf_path, temp_dir)
        temp_files.extend(page_files)
        
        logger.info(f"   Split PDF into {len(page_files)} page files")
        
        # Process each page
        all_blocks = []
        page_number = 1
        
        for page_file_path in page_files:
            logger.info(f"   Processing page {page_number}/{len(page_files)}...")
            
            try:
                # Determine content type
                is_image = page_file_path.lower().endswith(('.png', '.jpg', '.jpeg'))
                file_extension = '.png' if is_image else '.pdf'
                content_type = 'image/png' if is_image else 'application/pdf'
                
                # Upload page file to S3
                s3_page_key = f"temp_pages/{os.path.basename(document_name)}_{page_number}{file_extension}"
                await upload_file_to_s3(page_file_path, bucket_name, s3_page_key, content_type)
                
                # Extract text from this page
                page_blocks = await extract_from_page_file(bucket_name, s3_page_key)
                
                if page_blocks:
                    # Add page number to blocks for reference
                    for block in page_blocks:
                        block['PageNumber'] = page_number
                    all_blocks.extend(page_blocks)
                    logger.info(f"   Page {page_number}: Extracted {len(page_blocks)} blocks")
                else:
                    logger.info(f"   Page {page_number}: No blocks extracted")
                
            except Exception as page_error:
                logger.error(f"   Error processing page {page_number}: {page_error}")
                # Continue with other pages
            
            page_number += 1
        
        if not all_blocks:
            logger.info("   Page-by-page processing found no text")
            return None
        
        logger.info(f"   Total blocks extracted from all pages: {len(all_blocks)}")
        
        # Return combined result in Textract format
        return {
            'Blocks': all_blocks,
            'DocumentMetadata': {
                'Pages': len(page_files)
            }
        }
        
    except Exception as error:
        logger.error(f"   Page-by-page extraction failed: {error}")
        return None
    finally:
        # Clean up all temporary files and directory
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {e}")
        
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

async def download_pdf_from_s3(bucket_name: str, document_name: str, temp_dir: str) -> str:
    """Download PDF from S3 to local file"""
    temp_pdf_path = os.path.join(temp_dir, f"temp_{int(asyncio.get_event_loop().time())}.pdf")
    
    try:
        s3_client.download_file(bucket_name, document_name, temp_pdf_path)
        return temp_pdf_path
    except Exception as e:
        logger.error(f"Failed to download PDF from S3: {e}")
        raise

async def upload_file_to_s3(file_path: str, bucket_name: str, s3_key: str, content_type: str) -> None:
    """Upload any file to S3"""
    try:
        s3_client.upload_file(
            file_path, 
            bucket_name, 
            s3_key,
            ExtraArgs={'ContentType': content_type}
        )
    except Exception as e:
        logger.error(f"Failed to upload file to S3: {e}")
        raise

async def split_pdf_to_images(pdf_path: str, output_dir: str) -> List[str]:
    """Split PDF into individual page images"""
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path, dpi=200, fmt='png')
        
        page_files = []
        for i, image in enumerate(images):
            page_path = os.path.join(output_dir, f"page_{i+1}.png")
            image.save(page_path, 'PNG')
            page_files.append(page_path)
        
        return page_files
        
    except Exception as e:
        logger.error(f"Failed to split PDF to images: {e}")
        # Fallback to splitting PDF into separate PDF files
        return await split_pdf_to_pages(pdf_path, output_dir)

async def split_pdf_to_pages(pdf_path: str, output_dir: str) -> List[str]:
    """Split PDF into individual page PDF files"""
    try:
        reader = PdfReader(pdf_path)
        page_files = []
        
        for i, page in enumerate(reader.pages):
            writer = PdfWriter()
            writer.add_page(page)
            
            page_path = os.path.join(output_dir, f"page_{i+1}.pdf")
            with open(page_path, 'wb') as output_file:
                writer.write(output_file)
            
            page_files.append(page_path)
        
        return page_files
        
    except Exception as e:
        logger.error(f"Failed to split PDF to pages: {e}")
        raise

async def extract_from_page_file(bucket_name: str, file_key: str) -> List[Dict[str, Any]]:
    """Extract text from image or PDF using Textract"""
    try:
        # Try with table/form detection first
        response = textract_client.analyze_document(
            Document={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': file_key,
                }
            },
            FeatureTypes=['TABLES', 'FORMS']
        )
        return response.get('Blocks', [])
        
    except Exception:
        try:
            # Fallback to basic text detection
            response = textract_client.detect_document_text(
                Document={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': file_key,
                    }
                }
            )
            return response.get('Blocks', [])
            
        except Exception as text_error:
            logger.error(f"All methods failed for file: {text_error}")
            return []