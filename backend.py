"""
Backend Logic for Financial NER Analysis
Memory-optimized version with chunking and batch processing
"""

import pdfplumber
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import Dict, List, Optional
import torch
import gc
import os

# Memory optimization flags
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

class FinancialAnalyzer:
    """Memory-optimized Financial NER Analyzer"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.ner_pipeline = None
        self.tokenizer = None
        self.model = None
        self.max_length = 256  # Reduced from 512 to save memory
        
    def load_model(self):
        """Load model with memory optimization"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32  # Use float32 for CPU
            )
            
            # Force CPU and optimize
            self.model.eval()  # Set to evaluation mode
            
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.model, 
                tokenizer=self.tokenizer, 
                aggregation_strategy="simple",
                device=-1,  # Force CPU
                batch_size=1  # Process one at a time to save memory
            )
            
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 50) -> str:
        """
        Extract text with memory limits
        
        Args:
            pdf_path: Path to PDF
            max_pages: Maximum pages to process (prevents memory overflow)
        """
        text_chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = min(len(pdf.pages), max_pages)
                print(f"Extracting text from {total_pages} pages...")
                
                for i, page in enumerate(pdf.pages[:total_pages]):
                    if i % 10 == 0:
                        print(f"Processing page {i+1}/{total_pages}...")
                    
                    page_text = page.extract_text()
                    if page_text:
                        text_chunks.append(page_text)
                    
                    # Free memory every 10 pages
                    if i % 10 == 0:
                        gc.collect()
                
                return "\n".join(text_chunks)
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """
        Split text into smaller chunks to prevent memory issues
        
        Args:
            text: Full text
            chunk_size: Characters per chunk
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            
            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def extract_financial_data_regex(self, text: str) -> List[Dict]:
        """
        Regex-based extraction (no AI, no memory overhead)
        Use this as fallback or primary method for large files
        """
        results = []
        
        # Patterns
        company_pattern = r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b'
        metric_pattern = r'(?:total assets|total liabilities|revenue|net income|operating income|profit|sales|cash flow|earnings per share)'
        value_pattern = r'(?:[\$€₹₩]\s?)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?(?:billion|million|trillion|crore|%)?'
        date_pattern = r'(?:Q[1-4]|fiscal year|FY)\s?\d{2,4}|\b20\d{2}\b'
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)
        
        for sentence in sentences[:100]:  # Limit to first 100 sentences
            # Check if sentence contains financial info
            if re.search(metric_pattern, sentence, re.IGNORECASE) and re.search(value_pattern, sentence):
                metric_match = re.search(metric_pattern, sentence, re.IGNORECASE)
                value_match = re.search(value_pattern, sentence)
                date_match = re.search(date_pattern, sentence)
                company_match = re.search(company_pattern, sentence)
                
                results.append({
                    "company": company_match.group(0) if company_match else "Unknown",
                    "metric": metric_match.group(0) if metric_match else "N/A",
                    "value": value_match.group(0) if value_match else "N/A",
                    "period": date_match.group(0) if date_match else "N/A",
                    "source": "Regex"
                })
        
        return results
    
    def extract_mda_data(self, text: str) -> List[Dict]:
        """Extract MD&A metrics using regex only"""
        results = []
        lines = text.split('\n')
        
        metric_pattern = r'(?:revenue|operating income|net income|profit|sales)'
        value_pattern = r'(?:[\$€₹₩]\s?)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?(?:billion|million)?'
        
        for line in lines[:500]:  # Process first 500 lines only
            if re.search(value_pattern, line) and re.search(metric_pattern, line, re.IGNORECASE):
                metric = re.search(metric_pattern, line, re.IGNORECASE)
                val = re.search(value_pattern, line)
                
                if val and val.group(0) != "365":  # Filter noise
                    results.append({
                        "metric": metric.group(0).lower() if metric else "N/A",
                        "value": val.group(0) if val else "N/A",
                        "context": line.strip()[:100]
                    })
        
        return results[:10]
    
    def extract_balance_sheet(self, text: str) -> Optional[Dict]:
        """Extract balance sheet - regex only"""
        rows = []
        
        patterns = {
            "Total Assets": r"Total assets.*?[\$]?\s?([\d,]{3,})",
            "Total Liabilities": r"Total liabilities.*?[\$]?\s?([\d,]{3,})",
            "Total Equity": r"Total (?:shareholders'?|stockholders'?)?\s?equity.*?[\$]?\s?([\d,]{3,})"
        }
        
        for label, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                rows.append({"item": label, "value": f"${match.group(1)}"})
        
        return {"section": "Balance Sheet", "rows": rows} if rows else None
    
    def process_pdf(self, pdf_path: str, use_ai: bool = False) -> Dict:
        """
        Main processing with memory optimization
        
        Args:
            pdf_path: PDF file path
            use_ai: Whether to use AI (set False for large files)
        """
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # Step 1: Extract text with limits
            full_text = self.extract_text_from_pdf(pdf_path, max_pages=50)
            
            if not full_text.strip():
                return {"error": "No text extracted from PDF"}
            
            # Step 2: Regex-based extraction (always works)
            print("Running regex extraction...")
            mda_data = self.extract_mda_data(full_text)
            balance_sheet = self.extract_balance_sheet(full_text)
            regex_extractions = self.extract_financial_data_regex(full_text)
            
            # Step 3: AI extraction (optional, memory-intensive)
            ner_results = []
            if use_ai and self.ner_pipeline:
                print("Running AI extraction on small sample...")
                try:
                    # Only process first 2000 characters
                    sample = full_text[:2000]
                    chunks = self.chunk_text(sample, chunk_size=400)
                    
                    for i, chunk in enumerate(chunks[:3]):  # Max 3 chunks
                        print(f"Processing chunk {i+1}/3...")
                        try:
                            results = self.ner_pipeline(chunk)
                            ner_results.extend(results)
                            gc.collect()  # Force garbage collection
                        except Exception as e:
                            print(f"Chunk {i+1} failed: {e}")
                            break
                except Exception as e:
                    print(f"AI extraction failed: {e}")
                    print("Continuing with regex results only...")
            
            return {
                "success": True,
                "mda_metrics": mda_data,
                "balance_sheet": balance_sheet,
                "regex_extractions": regex_extractions,
                "ner_entities": ner_results,
                "total_pages": full_text.count('\f') + 1,
                "text_length": len(full_text),
                "ai_used": use_ai and len(ner_results) > 0
            }
            
        except Exception as e:
            return {"error": f"Error processing PDF: {str(e)}"}
        finally:
            # Always clean up memory
            gc.collect()


def create_summary_dataframe(analysis_results: Dict) -> pd.DataFrame:
    """Convert results to DataFrame"""
    if "error" in analysis_results:
        return pd.DataFrame({"Error": [analysis_results["error"]]})
    
    data = []
    
    # MD&A
    for item in analysis_results.get("mda_metrics", []):
        data.append({
            "Source": "MD&A",
            "Metric": item.get("metric", "N/A"),
            "Value": item.get("value", "N/A"),
            "Context": item.get("context", "")[:100]
        })
    
    # Balance Sheet
    if analysis_results.get("balance_sheet"):
        for row in analysis_results["balance_sheet"]["rows"]:
            data.append({
                "Source": "Balance Sheet",
                "Metric": row["item"],
                "Value": row["value"],
                "Context": "Financial Statement"
            })
    
    # Regex extractions
    for item in analysis_results.get("regex_extractions", [])[:10]:
        data.append({
            "Source": "Regex Analysis",
            "Metric": f"{item['metric']} ({item['company']})",
            "Value": item["value"],
            "Context": f"Period: {item['period']}"
        })
    
    return pd.DataFrame(data)


def get_entity_statistics(analysis_results: Dict) -> Dict:
    """Get entity stats"""
    entities = analysis_results.get("ner_entities", [])
    
    if not entities:
        return {"total_entities": 0, "entity_breakdown": {}, "unique_types": 0}
    
    entity_counts = {}
    for entity in entities:
        entity_type = entity.get("entity_group", "Unknown")
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    return {
        "total_entities": len(entities),
        "entity_breakdown": entity_counts,
        "unique_types": len(entity_counts)
    }