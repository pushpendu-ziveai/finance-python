import logging
import re
from typing import List, Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

def parse_transactions(textract_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse transactions from Textract response with multi-page support
    """
    # Handle null or undefined response from extract_pdf
    if not textract_response or not textract_response.get('Blocks'):
        logger.warning("Invalid or empty Textract response received")
        return []  # Return empty array instead of crashing
    
    blocks = textract_response.get('Blocks', [])
    logger.info(f"Processing {len(blocks)} blocks from Textract")
    
    # Check if this is multi-page data
    has_page_numbers = any(block.get('PageNumber') for block in blocks)
    total_pages = textract_response.get('DocumentMetadata', {}).get('Pages', 1)
    
    if has_page_numbers:
        logger.info(f"Processing multi-page document with {total_pages} pages")
    
    tables = [block for block in blocks if block.get('BlockType') == 'TABLE']
    cells = [block for block in blocks if block.get('BlockType') == 'CELL']
    lines = [block for block in blocks if block.get('BlockType') == 'LINE']
    words = [block for block in blocks if block.get('BlockType') == 'WORD']
    
    logger.info(f"Found {len(tables)} tables, {len(cells)} cells, {len(lines)} lines, {len(words)} words in document")
    
    # Debug: Show a sample of blocks to understand structure
    if blocks:
        logger.info("Sample blocks structure:")
        for idx, block in enumerate(blocks[:5]):
            block_text = block.get('Text', 'N/A')
            block_page = block.get('PageNumber') or block.get('Page', 'N/A')
            logger.info(f"Block {idx}: Type={block.get('BlockType')}, Text='{block_text}', Page={block_page}")
    
    # If tables detected but cells have relationship issues, try advanced parsing
    if tables and cells:
        logger.info("Tables and cells found. Attempting advanced table parsing...")
        table_transactions = parse_advanced_tables(tables, cells, blocks)
        if table_transactions:
            logger.info(f"Successfully extracted {len(table_transactions)} transactions from tables")
            return table_transactions
    
    # If no tables found or table parsing failed, try to extract text-based transactions
    logger.info("Table parsing failed or no tables found. Attempting to parse transactions from text blocks...")
    text_transactions = parse_transactions_from_text(blocks)
    
    # Final summary
    logger.info(f"\n🎯 FINAL PARSING SUMMARY 🎯")
    logger.info(f"Total transactions found: {len(text_transactions)}")
    if text_transactions:
        page_breakdown = {}
        for t in text_transactions:
            page = t.get('page', 1)
            page_breakdown[page] = page_breakdown.get(page, 0) + 1
        logger.info(f"Per-page breakdown: {page_breakdown}")
    logger.info("==============================\n")
    
    return text_transactions

def parse_advanced_tables(tables: List[Dict], all_cells: List[Dict], blocks: List[Dict]) -> List[Dict[str, Any]]:
    """Advanced table parsing that handles relationship issues"""
    transactions = []
    
    logger.info(f"\n=== ADVANCED TABLE PARSING START ===")
    logger.info(f"Total cells to process: {len(all_cells)}")
    
    # Debug: Check what page properties cells have
    if all_cells:
        sample_cell = all_cells[0]
        logger.info(f"Sample cell properties: {{'PageNumber': {sample_cell.get('PageNumber')}, 'Page': {sample_cell.get('Page')}, 'RowIndex': {sample_cell.get('RowIndex')}, 'ColumnIndex': {sample_cell.get('ColumnIndex')}}}")
    
    # Group cells by page - check both Page and PageNumber properties
    cells_by_page = {}
    for cell in all_cells:
        page_num = cell.get('PageNumber') or cell.get('Page', 1)
        if page_num not in cells_by_page:
            cells_by_page[page_num] = []
        cells_by_page[page_num].append(cell)
    
    page_info = [f"Page {p}: {len(cells_by_page[p])} cells" for p in sorted(cells_by_page.keys())]
    logger.info(f"Cells grouped by pages: {page_info}")
    
    # Process each page
    for page_num in sorted(cells_by_page.keys()):
        page_cells = cells_by_page[page_num]
        logger.info(f"\n--- Processing page {page_num} with {len(page_cells)} cells ---")
        
        # Group cells by row and column to form a grid
        cell_grid = {}
        max_row, max_col = 0, 0
        
        for cell in page_cells:
            row_idx = cell.get('RowIndex')
            col_idx = cell.get('ColumnIndex')
            if row_idx and col_idx:
                if row_idx not in cell_grid:
                    cell_grid[row_idx] = {}
                cell_grid[row_idx][col_idx] = get_text(cell, blocks)
                max_row = max(max_row, row_idx)
                max_col = max(max_col, col_idx)
        
        logger.info(f"Page {page_num}: Found {max_row} rows and {max_col} columns")
        
        page_transaction_count = 0
        
        # Convert grid to transactions (skip row 1 assuming it's header)
        for row in range(2, max_row + 1):
            if row in cell_grid:
                row_data = []
                for col in range(1, max_col + 1):
                    row_data.append(cell_grid[row].get(col, ''))
                
                # Only add if row has meaningful data
                if any(cell and cell.strip() for cell in row_data):
                    transaction = {
                        'date': row_data[0] if len(row_data) > 0 else '',
                        'description': row_data[1] if len(row_data) > 1 else '',
                        'amount': row_data[2] if len(row_data) > 2 else '',
                        'balance': row_data[3] if len(row_data) > 3 else '',
                        'page': page_num,
                        'raw_data': row_data,
                        'source': 'advanced_table_parsing'
                    }
                    transactions.append(transaction)
                    page_transaction_count += 1
        
        logger.info(f"Page {page_num}: Extracted {page_transaction_count} transactions")
    
    logger.info(f"\n=== ADVANCED TABLE PARSING COMPLETE ===")
    logger.info(f"Total transactions extracted: {len(transactions)}")
    page_summary = []
    for page_num in sorted(cells_by_page.keys()):
        page_transactions = [t for t in transactions if t['page'] == page_num]
        page_summary.append(f"Page {page_num}: {len(page_transactions)}")
    logger.info(f"Transactions per page: {page_summary}")
    logger.info("=====================================\n")
    
    return transactions

def parse_transactions_from_text(blocks: List[Dict]) -> List[Dict[str, Any]]:
    """Fallback function to parse transactions from text when no tables are found"""
    text_blocks = [block for block in blocks if block.get('BlockType') == 'LINE']
    transactions = []
    
    logger.info(f"\n=== TEXT PARSING START ===")
    logger.info(f"Attempting to parse {len(text_blocks)} text lines for transaction patterns")
    
    # Group text by pages first
    text_by_page = {}
    for block in text_blocks:
        page_num = block.get('PageNumber') or block.get('Page', 1)
        if page_num not in text_by_page:
            text_by_page[page_num] = []
        text_by_page[page_num].append(block.get('Text', ''))
    
    page_info = [f"Page {p}: {len(text_by_page[p])} lines" for p in sorted(text_by_page.keys())]
    logger.info(f"Text lines grouped by pages: {page_info}")
    
    for page_num in sorted(text_by_page.keys()):
        logger.info(f"\n--- Parsing text from page {page_num} with {len(text_by_page[page_num])} lines ---")
        page_lines = text_by_page[page_num]
        page_transaction_count = 0
        
        for text in page_lines:
            # Enhanced pattern matching for various transaction formats
            patterns = [
                # Pattern 1: Date Amount Description Balance
                re.compile(r'^(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+(.+?)\s+(\$?[-]?\d+[,.]?\d*\.?\d{2})\s+(\$?\d+[,.]?\d*\.?\d{2})$'),
                # Pattern 2: Date Description Amount Balance  
                re.compile(r'^(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+(.+?)\s+(\$?[-]?\d+[,.]?\d*\.?\d{2})\s+(\$?\d+[,.]?\d*\.?\d{2})$'),
                # Pattern 3: More flexible with various separators
                re.compile(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}).+?(\$?[-]?\d+[,.]?\d*\.?\d{2})'),
            ]
            
            for pattern in patterns:
                match = pattern.match(text)
                if match:
                    groups = match.groups()
                    description = groups[1] if len(groups) > 1 else ''
                    if len(groups) < 4:  # For pattern 3, extract description differently
                        # Remove date and amount from text to get description
                        temp_text = text.replace(groups[0], '').replace(groups[1] if len(groups) > 1 else '', '').strip()
                        description = temp_text
                    
                    transaction = {
                        'date': groups[0] if len(groups) > 0 else '',
                        'description': description,
                        'amount': groups[2] if len(groups) > 2 else (groups[1] if len(groups) == 2 else ''),
                        'balance': groups[3] if len(groups) > 3 else '',
                        'page': page_num,
                        'raw_data': text,
                        'source': 'enhanced_text_parsing'
                    }
                    
                    # Validate that we have meaningful data
                    if transaction['date'] and (transaction['amount'] or transaction['description']):
                        transactions.append(transaction)
                        page_transaction_count += 1
                        break  # Don't try other patterns for this line
        
        logger.info(f"Page {page_num}: Extracted {page_transaction_count} transactions from text")
    
    logger.info(f"\n=== TEXT PARSING COMPLETE ===")
    logger.info(f"Total transactions extracted: {len(transactions)}")
    page_summary = []
    for page_num in sorted(text_by_page.keys()):
        page_transactions = [t for t in transactions if t['page'] == page_num]
        page_summary.append(f"Page {page_num}: {len(page_transactions)}")
    logger.info(f"Transactions per page: {page_summary}")
    logger.info("============================\n")
    
    return transactions

def get_text(cell: Dict[str, Any], blocks: List[Dict[str, Any]]) -> str:
    """Recursively extract text inside cell from CHILD relationships"""
    if not cell.get('Relationships'):
        return ""
    
    TEXT_TYPE = "WORD"
    texts = []
    
    for rel in cell.get('Relationships', []):
        if rel.get('Type') == 'CHILD':
            for block_id in rel.get('Ids', []):
                word = next((b for b in blocks if b.get('Id') == block_id and b.get('BlockType') == TEXT_TYPE), None)
                if word and word.get('Text'):
                    texts.append(word['Text'])
    
    return ' '.join(texts)