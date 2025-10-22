"""
Enhanced PDF Extraction System
Inspired by Perplexity's multilayer hybrid extraction approach.

This system implements:
1. Layout-aware parsing with positional metadata
2. Structural segmentation using ML models 
3. Semantic field detection with embeddings
4. Query-time inference with RAG
"""

import os
import logging
import asyncio
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import boto3
from botocore.exceptions import ClientError
import fitz  # PyMuPDF for advanced PDF parsing
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class LayoutElement:
    """Represents a layout element with positional and semantic metadata"""
    text: str
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    page_num: int
    element_type: str  # 'header', 'table_cell', 'paragraph', 'transaction', 'total'
    confidence: float
    font_info: Dict[str, Any]
    semantic_embedding: Optional[np.ndarray] = None
    semantic_tags: Optional[List[str]] = None

@dataclass 
class DocumentStructure:
    """Represents the complete document structure"""
    elements: List[LayoutElement]
    pages: int
    document_type: str  # 'bank_statement', 'invoice', 'report'
    semantic_index: Dict[str, List[int]]  # Tag -> element indices
    transaction_clusters: List[List[int]]  # Clustered transaction indices

class EnhancedPDFExtractor:
    """
    Advanced PDF extraction system using multilayer hybrid approach
    """
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
        self.textract_client = boto3.client('textract', region_name=os.getenv("AWS_REGION", "us-east-1"))
        self.s3_client = boto3.client('s3', region_name=os.getenv("AWS_REGION", "us-east-1"))
        
        # Pre-defined semantic patterns for financial documents
        self.financial_patterns = {
            'date': [r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', r'\d{4}-\d{2}-\d{2}', r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b'],
            'amount': [r'[\$₹€£]\s*[\d,]+\.?\d*', r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'],
            'account_number': [r'\b\d{4,16}\b', r'\*+\d{4}'],
            'transaction_type': ['debit', 'credit', 'transfer', 'withdrawal', 'deposit', 'payment', 'charge'],
            'beneficiary': [r'to\s+([A-Za-z\s]+)', r'from\s+([A-Za-z\s]+)']
        }
    
    async def extract_with_multilayer_approach(self, bucket_name: str, document_name: str) -> Optional[DocumentStructure]:
        """
        Main extraction method implementing the multilayer hybrid approach
        """
        logger.info(f"🚀 Starting enhanced multilayer extraction for {document_name}")
        
        try:
            # Step 1: Low-level layout parsing
            layout_elements = await self._extract_layout_elements(bucket_name, document_name)
            if not layout_elements:
                logger.error("❌ Failed to extract layout elements")
                return None
            
            # Step 2: Structural segmentation 
            classified_elements = await self._classify_layout_elements(layout_elements)
            
            # Step 3: Semantic field detection
            semantic_elements = await self._add_semantic_embeddings(classified_elements)
            
            # Step 4: Build document structure
            doc_structure = await self._build_document_structure(semantic_elements)
            
            logger.info(f"✅ Successfully created document structure with {len(doc_structure.elements)} elements")
            return doc_structure
            
        except Exception as e:
            logger.error(f"❌ Error in multilayer extraction: {e}")
            return None
    
    async def _extract_layout_elements(self, bucket_name: str, document_name: str) -> List[LayoutElement]:
        """
        Step 1: Low-level layout parsing using PyMuPDF for precise positioning
        """
        logger.info("🔍 Step 1: Extracting layout elements with positional metadata")
        
        # Download PDF from S3
        temp_pdf_path = f"/tmp/{document_name}"
        try:
            self.s3_client.download_file(bucket_name, document_name, temp_pdf_path)
        except Exception as e:
            logger.error(f"Failed to download PDF: {e}")
            return []
        
        elements = []
        
        try:
            # Open PDF with PyMuPDF for detailed layout analysis
            pdf_doc = fitz.open(temp_pdf_path)
            
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                
                # Extract text blocks with position and font information
                blocks = page.get_text("dict")
                
                for block in blocks["blocks"]:
                    if "lines" not in block:  # Skip image blocks
                        continue
                        
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if not text:
                                continue
                            
                            # Create layout element with rich metadata
                            element = LayoutElement(
                                text=text,
                                bbox=(span["bbox"][0], span["bbox"][1], span["bbox"][2], span["bbox"][3]),
                                page_num=page_num + 1,
                                element_type="unknown",  # Will be classified later
                                confidence=1.0,
                                font_info={
                                    "font": span["font"],
                                    "size": span["size"],
                                    "flags": span["flags"],
                                    "color": span["color"]
                                }
                            )
                            elements.append(element)
            
            pdf_doc.close()
            
            # Also get Textract data for comparison and validation
            textract_elements = await self._get_textract_elements(bucket_name, document_name)
            elements.extend(textract_elements)
            
            logger.info(f"   Extracted {len(elements)} layout elements from {pdf_doc.page_count} pages")
            return elements
            
        except Exception as e:
            logger.error(f"Error in layout extraction: {e}")
            return []
        finally:
            # Clean up temp file
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)
    
    async def _get_textract_elements(self, bucket_name: str, document_name: str) -> List[LayoutElement]:
        """
        Get Textract data and convert to LayoutElements for validation
        """
        try:
            response = self.textract_client.analyze_document(
                Document={'S3Object': {'Bucket': bucket_name, 'Name': document_name}},
                FeatureTypes=['TABLES', 'FORMS']
            )
            
            elements = []
            for block in response.get('Blocks', []):
                if block['BlockType'] in ['LINE', 'WORD'] and 'Text' in block:
                    bbox = block.get('Geometry', {}).get('BoundingBox', {})
                    element = LayoutElement(
                        text=block['Text'],
                        bbox=(bbox.get('Left', 0), bbox.get('Top', 0), 
                             bbox.get('Left', 0) + bbox.get('Width', 0),
                             bbox.get('Top', 0) + bbox.get('Height', 0)),
                        page_num=block.get('Page', 1),
                        element_type="textract_" + block['BlockType'].lower(),
                        confidence=block.get('Confidence', 0) / 100.0,
                        font_info={}
                    )
                    elements.append(element)
            
            return elements
            
        except Exception as e:
            logger.warning(f"Textract validation failed: {e}")
            return []
    
    async def _classify_layout_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Step 2: Structural segmentation - classify elements into logical types
        """
        logger.info("🏗️ Step 2: Classifying layout elements by structure and content")
        
        for element in elements:
            # Rule-based classification (can be enhanced with ML models)
            element.element_type = self._classify_element_type(element)
        
        # Group related elements (e.g., table rows, transaction blocks)
        grouped_elements = self._group_related_elements(elements)
        
        logger.info(f"   Classified elements: {self._get_classification_summary(grouped_elements)}")
        return grouped_elements
    
    def _classify_element_type(self, element: LayoutElement) -> str:
        """
        Classify an element based on content patterns and position
        """
        text = element.text.lower()
        
        # Check for different element types
        if any(pattern in text for pattern in ['statement', 'account summary', 'balance']):
            return 'header'
        elif any(pattern in text for pattern in ['total', 'balance:', 'amount:']):
            return 'total'
        elif self._matches_date_pattern(element.text):
            return 'date_field'
        elif self._matches_amount_pattern(element.text):
            return 'amount_field'
        elif any(pattern in text for pattern in ['debit', 'credit', 'transfer', 'payment', 'upi', 'imps', 'neft']):
            return 'transaction'
        elif element.font_info.get('size', 0) > 14:
            return 'header'
        else:
            return 'paragraph'
    
    def _matches_date_pattern(self, text: str) -> bool:
        """Check if text matches date patterns"""
        import re
        for pattern in self.financial_patterns['date']:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _matches_amount_pattern(self, text: str) -> bool:
        """Check if text matches amount patterns"""
        import re
        for pattern in self.financial_patterns['amount']:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _group_related_elements(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Group spatially and semantically related elements
        """
        # Sort elements by page and vertical position
        elements.sort(key=lambda e: (e.page_num, e.bbox[1]))
        
        # Simple grouping by proximity (can be enhanced)
        return elements
    
    def _get_classification_summary(self, elements: List[LayoutElement]) -> Dict[str, int]:
        """Get summary of element classifications"""
        summary = defaultdict(int)
        for element in elements:
            summary[element.element_type] += 1
        return dict(summary)
    
    async def _add_semantic_embeddings(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        """
        Step 3: Add semantic embeddings for each element
        """
        logger.info("🧠 Step 3: Adding semantic embeddings for content understanding")
        
        # Batch process embeddings for efficiency
        texts = [element.text for element in elements]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        
        for i, element in enumerate(elements):
            element.semantic_embedding = embeddings[i]
            element.semantic_tags = self._generate_semantic_tags(element)
        
        logger.info(f"   Added embeddings to {len(elements)} elements")
        return elements
    
    def _generate_semantic_tags(self, element: LayoutElement) -> List[str]:
        """
        Generate semantic tags based on content analysis
        """
        tags = []
        text = element.text.lower()
        
        # Financial semantic tags
        if any(word in text for word in ['debit', 'withdrawal', 'payment', 'charge']):
            tags.append('outgoing_transaction')
        if any(word in text for word in ['credit', 'deposit', 'received']):
            tags.append('incoming_transaction')
        if any(word in text for word in ['balance', 'total']):
            tags.append('balance_info')
        if self._matches_date_pattern(element.text):
            tags.append('date')
        if self._matches_amount_pattern(element.text):
            tags.append('amount')
        
        return tags
    
    async def _build_document_structure(self, elements: List[LayoutElement]) -> DocumentStructure:
        """
        Step 4: Build the final document structure with semantic indexing
        """
        logger.info("🏛️ Step 4: Building semantic document structure")
        
        # Create semantic index
        semantic_index = defaultdict(list)
        for i, element in enumerate(elements):
            for tag in element.semantic_tags or []:
                semantic_index[tag].append(i)
            semantic_index[element.element_type].append(i)
        
        # Cluster transactions using embeddings
        transaction_indices = semantic_index.get('transaction', [])
        transaction_clusters = self._cluster_transactions(elements, transaction_indices)
        
        # Determine document type
        doc_type = self._determine_document_type(elements)
        
        pages = max(element.page_num for element in elements) if elements else 0
        
        structure = DocumentStructure(
            elements=elements,
            pages=pages,
            document_type=doc_type,
            semantic_index=dict(semantic_index),
            transaction_clusters=transaction_clusters
        )
        
        logger.info(f"   Built structure: {pages} pages, {len(semantic_index)} semantic categories")
        return structure
    
    def _cluster_transactions(self, elements: List[LayoutElement], transaction_indices: List[int]) -> List[List[int]]:
        """
        Cluster transactions using semantic similarity and spatial proximity
        """
        if len(transaction_indices) < 2:
            return [transaction_indices] if transaction_indices else []
        
        # Combine semantic and spatial features
        features = []
        for idx in transaction_indices:
            element = elements[idx]
            embedding = element.semantic_embedding
            spatial_features = [element.bbox[0], element.bbox[1], element.page_num]
            
            # Combine embedding with normalized spatial features
            combined_features = np.concatenate([
                embedding,
                np.array(spatial_features) / 1000.0  # Normalize spatial features
            ])
            features.append(combined_features)
        
        # Cluster using DBSCAN
        if len(features) > 1:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            clustering = DBSCAN(eps=0.3, min_samples=2).fit(features_scaled)
            
            # Group indices by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(clustering.labels_):
                clusters[label].append(transaction_indices[i])
            
            return list(clusters.values())
        
        return [transaction_indices]
    
    def _determine_document_type(self, elements: List[LayoutElement]) -> str:
        """
        Determine document type based on content analysis
        """
        all_text = ' '.join(element.text.lower() for element in elements)
        
        if any(keyword in all_text for keyword in ['statement', 'account', 'balance', 'transaction']):
            return 'bank_statement'
        elif any(keyword in all_text for keyword in ['invoice', 'bill', 'payment due']):
            return 'invoice'
        else:
            return 'document'
    
    async def semantic_query(self, structure: DocumentStructure, query: str, top_k: int = 10) -> List[Tuple[LayoutElement, float]]:
        """
        Step 5: Query-time semantic search and inference
        """
        logger.info(f"🔍 Performing semantic query: '{query}'")
        
        # Encode the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = []
        for element in structure.elements:
            if element.semantic_embedding is not None:
                similarity = np.dot(query_embedding, element.semantic_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(element.semantic_embedding)
                )
                similarities.append((element, float(similarity)))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = similarities[:top_k]
        
        logger.info(f"   Found {len(results)} relevant elements")
        return results
    
    def extract_transactions_from_structure(self, structure: DocumentStructure) -> List[Dict[str, Any]]:
        """
        Extract structured transaction data from document structure
        """
        transactions = []
        
        # Get transaction clusters
        for cluster in structure.transaction_clusters:
            # Analyze each cluster to extract transaction details
            cluster_elements = [structure.elements[i] for i in cluster]
            
            # Group elements by spatial proximity (transaction rows)
            transaction_rows = self._group_elements_into_rows(cluster_elements)
            
            for row_elements in transaction_rows:
                transaction = self._parse_transaction_row(row_elements)
                if transaction:
                    transactions.append(transaction)
        
        logger.info(f"Extracted {len(transactions)} structured transactions")
        return transactions
    
    def _group_elements_into_rows(self, elements: List[LayoutElement]) -> List[List[LayoutElement]]:
        """
        Group elements into transaction rows based on spatial proximity
        """
        if not elements:
            return []
        
        # Sort by page and Y position
        elements.sort(key=lambda e: (e.page_num, e.bbox[1]))
        
        rows = []
        current_row = [elements[0]]
        
        for element in elements[1:]:
            # If element is on same page and similar Y position, add to current row
            last_element = current_row[-1]
            if (element.page_num == last_element.page_num and
                abs(element.bbox[1] - last_element.bbox[1]) < 10):  # 10 pixel tolerance
                current_row.append(element)
            else:
                # Start new row
                rows.append(current_row)
                current_row = [element]
        
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _parse_transaction_row(self, row_elements: List[LayoutElement]) -> Optional[Dict[str, Any]]:
        """
        Parse a transaction row into structured data
        """
        if not row_elements:
            return None
        
        # Sort elements by X position (left to right)
        row_elements.sort(key=lambda e: e.bbox[0])
        
        transaction = {
            'date': None,
            'description': None,
            'amount': None,
            'balance': None,
            'type': None,
            'page': row_elements[0].page_num,
            'confidence': sum(e.confidence for e in row_elements) / len(row_elements)
        }
        
        # Extract fields from row elements
        for element in row_elements:
            text = element.text.strip()
            
            if self._matches_date_pattern(text) and not transaction['date']:
                transaction['date'] = text
            elif self._matches_amount_pattern(text):
                # First amount is usually transaction amount, second is balance
                if not transaction['amount']:
                    transaction['amount'] = text
                elif not transaction['balance']:
                    transaction['balance'] = text
            elif not transaction['description'] and len(text) > 3:
                # Longest text element is usually description
                if not transaction['description'] or len(text) > len(transaction['description']):
                    transaction['description'] = text
        
        # Determine transaction type
        if transaction['description']:
            desc_lower = transaction['description'].lower()
            if any(word in desc_lower for word in ['debit', 'withdrawal', 'payment', 'transfer']):
                transaction['type'] = 'debit'
            elif any(word in desc_lower for word in ['credit', 'deposit', 'received']):
                transaction['type'] = 'credit'
        
        # Only return transaction if we have minimum required fields
        if transaction['date'] or transaction['description'] or transaction['amount']:
            return transaction
        
        return None

# Example usage function
async def process_pdf_with_enhanced_extraction(bucket_name: str, document_name: str) -> Dict[str, Any]:
    """
    Process a PDF using the enhanced multilayer extraction approach
    """
    extractor = EnhancedPDFExtractor()
    
    # Extract document structure
    structure = await extractor.extract_with_multilayer_approach(bucket_name, document_name)
    
    if not structure:
        return {"error": "Failed to extract document structure"}
    
    # Extract transactions
    transactions = extractor.extract_transactions_from_structure(structure)
    
    # Example semantic queries
    queries = [
        "total amount spent",
        "transactions to Alpana Kundu",
        "credit transactions",
        "balance information"
    ]
    
    query_results = {}
    for query in queries:
        results = await extractor.semantic_query(structure, query, top_k=5)
        query_results[query] = [
            {"text": result[0].text, "confidence": result[1], "page": result[0].page_num}
            for result in results
        ]
    
    return {
        "document_structure": {
            "pages": structure.pages,
            "document_type": structure.document_type,
            "total_elements": len(structure.elements),
            "semantic_categories": list(structure.semantic_index.keys()),
            "transaction_clusters": len(structure.transaction_clusters)
        },
        "transactions": transactions,
        "semantic_queries": query_results,
        "extraction_method": "enhanced_multilayer_hybrid"
    }