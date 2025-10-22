"""
Enhanced Query Agent with Semantic Understanding
Implements retrieval-augmented reasoning similar to Perplexity's approach.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import boto3
from botocore.exceptions import ClientError
from sentence_transformers import SentenceTransformer

from .enhanced_pdf_extractor import DocumentStructure, LayoutElement

# Setup logging
logger = logging.getLogger(__name__)

class EnhancedQueryAgent:
    """
    Advanced query agent that performs semantic search and reasoning over document structures
    """
    
    def __init__(self):
        # AWS Configuration
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
        
        # Initialize embedding model (same as extractor for consistency)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Financial reasoning templates
        self.reasoning_templates = {
            'sum_calculation': """
                Based on the transactions provided, calculate the total for: {query}
                
                Relevant transactions found:
                {transactions}
                
                Please provide:
                1. The sum calculation
                2. Individual amounts included
                3. Any patterns or insights
            """,
            
            'beneficiary_analysis': """
                Analyze transactions related to: {query}
                
                Relevant transactions:
                {transactions}
                
                Please provide:
                1. Total amount to/from this beneficiary
                2. Transaction patterns (dates, amounts)
                3. Transaction types (debit/credit)
            """,
            
            'time_analysis': """
                Analyze transactions for the time period: {query}
                
                Relevant transactions:
                {transactions}
                
                Please provide:
                1. Summary of activity in this period
                2. Total inflows and outflows
                3. Key insights or patterns
            """,
            
            'general_analysis': """
                Analyze the following financial data for: {query}
                
                Document context:
                {context}
                
                Relevant information:
                {transactions}
                
                Please provide a comprehensive analysis addressing the user's query.
            """
        }
    
    async def query_with_semantic_search(
        self, 
        query: str, 
        document_structure: DocumentStructure,
        transactions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Main query method that combines semantic search with LLM reasoning
        """
        logger.info(f"🔍 Processing semantic query: '{query}'")
        
        try:
            # Step 1: Semantic search over document structure
            semantic_results = await self._semantic_search(query, document_structure)
            
            # Step 2: Filter and rank relevant transactions
            relevant_transactions = await self._find_relevant_transactions(query, transactions or [])
            
            # Step 3: Determine query type and select reasoning template
            query_type = self._classify_query_type(query)
            
            # Step 4: Generate context-aware response using LLM
            llm_response = await self._generate_llm_response(
                query, query_type, semantic_results, relevant_transactions, document_structure
            )
            
            # Step 5: Compile comprehensive response
            response = {
                "answer": llm_response,
                "semantic_matches": [
                    {
                        "text": result[0].text,
                        "confidence": float(result[1]),
                        "page": result[0].page_num,
                        "element_type": result[0].element_type,
                        "semantic_tags": result[0].semantic_tags
                    }
                    for result in semantic_results[:10]
                ],
                "relevant_transactions": relevant_transactions[:20],
                "query_type": query_type,
                "document_insights": self._generate_document_insights(document_structure),
                "processing_method": "enhanced_semantic_rag"
            }
            
            logger.info(f"✅ Generated response with {len(semantic_results)} semantic matches and {len(relevant_transactions)} transactions")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in semantic query processing: {e}")
            return {
                "error": str(e),
                "answer": "I encountered an error while processing your query. Please try rephrasing or contact support."
            }
    
    async def _semantic_search(self, query: str, structure: DocumentStructure, top_k: int = 20) -> List[Tuple[LayoutElement, float]]:
        """
        Perform semantic search over document structure
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities with all elements that have embeddings
        similarities = []
        for element in structure.elements:
            if element.semantic_embedding is not None:
                similarity = np.dot(query_embedding, element.semantic_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(element.semantic_embedding)
                )
                similarities.append((element, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def _find_relevant_transactions(self, query: str, transactions: List[Dict[str, Any]], top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Find transactions relevant to the query using semantic matching
        """
        if not transactions:
            return []
        
        query_lower = query.lower()
        relevant_transactions = []
        
        # Score transactions based on multiple criteria
        for transaction in transactions:
            score = 0.0
            
            # Text matching in description
            description = transaction.get('description', '').lower()
            if any(word in description for word in query_lower.split()):
                score += 2.0
            
            # Semantic similarity for description
            if description:
                desc_embedding = self.embedding_model.encode([description])[0]
                query_embedding = self.embedding_model.encode([query])[0]
                semantic_sim = np.dot(query_embedding, desc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(desc_embedding)
                )
                score += semantic_sim
            
            # Check for specific query patterns
            if self._matches_beneficiary_query(query_lower, transaction):
                score += 3.0
            if self._matches_amount_query(query_lower, transaction):
                score += 2.0
            if self._matches_date_query(query_lower, transaction):
                score += 2.0
            if self._matches_type_query(query_lower, transaction):
                score += 1.5
            
            if score > 0.3:  # Threshold for relevance
                transaction_with_score = transaction.copy()
                transaction_with_score['relevance_score'] = score
                relevant_transactions.append(transaction_with_score)
        
        # Sort by relevance score
        relevant_transactions.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_transactions[:top_k]
    
    def _matches_beneficiary_query(self, query: str, transaction: Dict[str, Any]) -> bool:
        """Check if query matches beneficiary patterns"""
        beneficiary_keywords = ['to', 'from', 'paid', 'received', 'sent', 'transfer']
        description = transaction.get('description', '').lower()
        
        if any(keyword in query for keyword in beneficiary_keywords):
            # Extract potential names from query
            query_words = query.split()
            for word in query_words:
                if len(word) > 2 and word.isalpha() and word in description:
                    return True
        return False
    
    def _matches_amount_query(self, query: str, transaction: Dict[str, Any]) -> bool:
        """Check if query matches amount patterns"""
        amount_keywords = ['amount', 'total', 'sum', 'spent', 'received', 'paid']
        return any(keyword in query for keyword in amount_keywords)
    
    def _matches_date_query(self, query: str, transaction: Dict[str, Any]) -> bool:
        """Check if query matches date patterns"""
        date_keywords = ['date', 'when', 'month', 'year', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
        return any(keyword in query for keyword in date_keywords)
    
    def _matches_type_query(self, query: str, transaction: Dict[str, Any]) -> bool:
        """Check if query matches transaction type patterns"""
        type_keywords = ['debit', 'credit', 'withdrawal', 'deposit', 'transfer', 'payment']
        transaction_type = transaction.get('type', '').lower()
        
        return any(keyword in query for keyword in type_keywords) or transaction_type in query
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify the type of query to select appropriate reasoning template
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['sum', 'total', 'add', 'calculate', 'how much']):
            return 'sum_calculation'
        elif any(word in query_lower for word in ['to', 'from', 'paid', 'sent', 'received']):
            return 'beneficiary_analysis'
        elif any(word in query_lower for word in ['date', 'when', 'month', 'year', 'period']):
            return 'time_analysis'
        else:
            return 'general_analysis'
    
    async def _generate_llm_response(
        self, 
        query: str, 
        query_type: str, 
        semantic_results: List[Tuple[LayoutElement, float]], 
        relevant_transactions: List[Dict[str, Any]], 
        document_structure: DocumentStructure
    ) -> str:
        """
        Generate LLM response using context and reasoning templates
        """
        # Prepare context from semantic results
        context_elements = []
        for element, score in semantic_results[:10]:
            context_elements.append(f"- {element.text} (Page {element.page_num}, Type: {element.element_type}, Score: {score:.3f})")
        
        context = "\\n".join(context_elements)
        
        # Prepare transaction data
        transaction_text = ""
        if relevant_transactions:
            transaction_text = "\\n".join([
                f"• {t.get('date', 'N/A')}: {t.get('description', 'N/A')} - Amount: {t.get('amount', 'N/A')} (Score: {t.get('relevance_score', 0):.2f})"
                for t in relevant_transactions[:15]
            ])
        else:
            transaction_text = "No directly relevant transactions found."
        
        # Select and fill reasoning template
        template = self.reasoning_templates.get(query_type, self.reasoning_templates['general_analysis'])
        
        if query_type == 'general_analysis':
            prompt_content = template.format(
                query=query,
                context=context,
                transactions=transaction_text
            )
        else:
            prompt_content = template.format(
                query=query,
                transactions=transaction_text
            )
        
        # Add document metadata for better context
        document_info = f"""
        Document Information:
        - Type: {document_structure.document_type}
        - Pages: {document_structure.pages}
        - Total Elements: {len(document_structure.elements)}
        - Semantic Categories: {list(document_structure.semantic_index.keys())}
        """
        
        full_prompt = f"{document_info}\\n\\n{prompt_content}"
        
        # Call Claude via Bedrock
        return await self._call_claude(full_prompt)
    
    async def _call_claude(self, prompt: str) -> str:
        """
        Call Claude 3 Haiku via AWS Bedrock
        """
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "messages": [
                {
                    "role": "user",
                    "content": f"""You are an expert financial analyst with access to bank statement data. 
                    
Please analyze the provided information and answer the user's question with:
1. Direct answer to their question
2. Supporting evidence from the data
3. Relevant calculations if applicable
4. Key insights or patterns you notice

Be precise, factual, and cite specific information from the provided data.

{prompt}"""
                }
            ],
            "temperature": 0.1,
            "top_p": 0.9
        }
        
        try:
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _generate_document_insights(self, structure: DocumentStructure) -> Dict[str, Any]:
        """
        Generate high-level insights about the document structure
        """
        insights = {
            "document_type": structure.document_type,
            "total_pages": structure.pages,
            "semantic_categories": len(structure.semantic_index),
            "transaction_clusters": len(structure.transaction_clusters),
            "category_distribution": {}
        }
        
        # Calculate distribution of semantic categories
        for category, indices in structure.semantic_index.items():
            insights["category_distribution"][category] = len(indices)
        
        return insights

# Usage example
async def query_enhanced_document(
    query: str, 
    document_structure: DocumentStructure, 
    transactions: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Query a document using the enhanced semantic approach
    """
    agent = EnhancedQueryAgent()
    return await agent.query_with_semantic_search(query, document_structure, transactions)