import os
import json
import logging
from typing import List, Dict, Any
import boto3
from botocore.exceptions import ClientError

# Setup logging
logger = logging.getLogger(__name__)

# AWS Configuration
REGION = os.getenv("AWS_REGION", "us-east-1")

# Initialize Bedrock client
bedrock_client = boto3.client('bedrock-runtime', region_name=REGION)

async def query_agent(prompt: str, transactions: List[Dict[str, Any]] = None) -> str:
    """
    Query the AI agent with transaction data using AWS Bedrock Claude
    """
    # Use Claude 3 Haiku - more cost effective for simple queries
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    
    # Prepare transaction context if provided
    transaction_context = ""
    if transactions:
        transaction_context = f"\n\nHere are the transactions to analyze:\n"
        for i, transaction in enumerate(transactions[:50], 1):  # Limit to first 50 for context
            date = transaction.get('date', 'N/A')
            desc = transaction.get('description', 'N/A')
            amount = transaction.get('amount', 'N/A')
            balance = transaction.get('balance', 'N/A')
            page = transaction.get('page', 'N/A')
            
            transaction_context += f"{i}. Date: {date}, Description: {desc}, Amount: {amount}, Balance: {balance}, Page: {page}\n"
        
        if len(transactions) > 50:
            transaction_context += f"\n... and {len(transactions) - 50} more transactions.\n"
        
        transaction_context += f"\nTotal transactions: {len(transactions)}\n"
    
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,  # Increased to handle larger transaction sets
        "messages": [
            {
                "role": "user",
                "content": f"""You are a helpful financial assistant analyzing bank transactions. 

When listing transactions or providing detailed responses:
- Use proper line breaks and formatting
- Number items clearly (1., 2., 3., etc.)
- Use bullet points for lists when appropriate
- Keep responses well-structured and easy to read
- Preserve currency symbols and formatting
- Use clear sections for different types of information

{transaction_context}

User Query: {prompt}

Please format your response with proper line breaks and structure for easy reading."""
            }
        ]
    }
    
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"Bedrock ClientError: {error_code} - {error_message}")
        raise Exception(f"Failed to get response from Bedrock: {error_message}")
    
    except Exception as e:
        logger.error(f"Bedrock error: {str(e)}")
        raise Exception(f"Failed to get response from Bedrock: {str(e)}")

def format_transactions_for_query(transactions: List[Dict[str, Any]]) -> str:
    """
    Format transactions into a readable string for AI processing
    """
    if not transactions:
        return "No transactions available."
    
    formatted = "Transaction Summary:\n"
    formatted += "=" * 50 + "\n"
    
    for i, transaction in enumerate(transactions, 1):
        date = transaction.get('date', 'N/A')
        description = transaction.get('description', 'N/A')
        amount = transaction.get('amount', 'N/A')
        balance = transaction.get('balance', 'N/A')
        page = transaction.get('page', 'N/A')
        
        formatted += f"{i:3}. {date:<12} | {description:<30} | {amount:<15} | Balance: {balance:<15} | Page: {page}\n"
    
    formatted += "\n" + "=" * 50 + "\n"
    formatted += f"Total Transactions: {len(transactions)}\n"
    
    return formatted

def get_transaction_statistics(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate basic statistics about transactions
    """
    if not transactions:
        return {"total": 0, "pages": 0, "error": "No transactions available"}
    
    total_transactions = len(transactions)
    pages = set()
    amounts = []
    
    for transaction in transactions:
        if transaction.get('page'):
            pages.add(transaction['page'])
        
        # Try to extract numeric amount
        amount_str = transaction.get('amount', '')
        if amount_str:
            try:
                # Remove currency symbols and commas, handle negative amounts
                amount_clean = amount_str.replace('$', '').replace(',', '').strip()
                if amount_clean.startswith('(') and amount_clean.endswith(')'):
                    amount_clean = '-' + amount_clean[1:-1]
                amounts.append(float(amount_clean))
            except (ValueError, TypeError):
                continue  # Skip invalid amounts
    
    stats = {
        "total": total_transactions,
        "pages": len(pages),
        "unique_pages": sorted(list(pages)),
        "amounts_parsed": len(amounts)
    }
    
    if amounts:
        stats.update({
            "total_amount": sum(amounts),
            "average_amount": sum(amounts) / len(amounts),
            "max_amount": max(amounts),
            "min_amount": min(amounts)
        })
    
    return stats