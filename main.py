#!/usr/bin/env python3
"""
Document Taxonomy Analyzer

This application uses LLM to analyze documents and assign taxonomy tags with confidence scoring.
It reads taxonomy rules, processes documents from a configured folder, and outputs a table
with 13 taxonomy tags per document.

Author: AI Assistant
Date: 2025
"""

import os
import sys
import logging
from pathlib import Path
import yaml
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Import custom modules
from taxonomy_analyzer import TaxonomyAnalyzer
from llm_client import LLMClient
from document_processor import DocumentProcessor
from confidence_scorer import ConfidenceScorer
from tag_assigner import TagAssigner
from vector_store import VectorStore


class DocumentTaxonomyAnalyzer:
    """Main application class for document taxonomy analysis."""
    
    # Add this import at the top of the file
    from vector_store import VectorStore
    
    # Modify the DocumentTaxonomyAnalyzer class initialization
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the analyzer with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_directories()
        
        # Initialize components
        self.taxonomy_analyzer = None
        self.llm_client = None
        self.document_processor = None
        self.confidence_scorer = None
        self.tag_assigner = None
        self.vector_store = None  # Add this line
        
        # Cache for taxonomy data
        self.taxonomy_loaded = False
        
        # RAG enabled flag
        self.rag_enabled = self.config.get('rag', {}).get('enabled', False)  # Add this line

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Override API key with environment variable if present
            if 'OPENAI_API_KEY' in os.environ:
                config['openai']['api_key'] = os.environ['OPENAI_API_KEY']
                
            return config
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config['logging']['file']).parent
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['logging']['file']),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_directories(self):
        """Create necessary directories."""
        dirs_to_create = [
            self.config['documents']['input_folder'],
            Path(self.config['output']['output_file']).parent,
            Path(self.config['logging']['file']).parent
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Modify the initialize_components method
    def initialize_components(self):
        """Initialize all components. Only loads taxonomy on first run."""
        if not self.taxonomy_loaded:
            self.logger.info("First run detected. Loading and analyzing taxonomy file...")
            self.taxonomy_analyzer = TaxonomyAnalyzer(self.config['taxonomy'])
            self.taxonomy_analyzer.load_taxonomy()
            self.taxonomy_loaded = True
            self.logger.info("Taxonomy loaded successfully and cached in memory.")
        
        # Initialize other components
        self.llm_client = LLMClient(self.config['openai'])
        self.document_processor = DocumentProcessor(self.config['documents'])
        self.confidence_scorer = ConfidenceScorer(self.taxonomy_analyzer)
        self.tag_assigner = TagAssigner(self.taxonomy_analyzer, self.confidence_scorer)
        
        # Initialize vector store for RAG if enabled
        if self.rag_enabled:
            self.logger.info("Initializing RAG components...")
            self.vector_store = VectorStore({**self.config['openai'], **self.config['rag']})
            
            # Create taxonomy embeddings if not already done
            if not self.vector_store.taxonomy_embeddings:
                self.logger.info("Creating taxonomy embeddings...")
                self.vector_store.create_taxonomy_embeddings(self.taxonomy_analyzer.taxonomy_data)

    # Add this new method for RAG-enhanced tag assignment
    def _assign_tags_with_rag(self, document_content: str, intent_analysis: Dict, filename: str) -> Dict[str, str]:
        """Assign tags using RAG-enhanced approach."""
        tag_results = {}
        
        # Process each required facet
        for required_facet in self.tag_assigner.REQUIRED_FACETS:
            try:
                # Get possible values for this facet
                possible_values = self.tag_assigner._get_facet_values(required_facet)
                
                # Retrieve relevant context for this facet
                facet_query = f"Document classification for {required_facet}: {document_content[:500]}"
                relevant_context = self.vector_store.retrieve_relevant_context(
                    facet_query,
                    self.config['rag'].get('top_k_chunks', 3)
                )
                
                # Get relevant taxonomy matches
                taxonomy_matches = self.vector_store.retrieve_taxonomy_matches(
                    document_content[:500],
                    required_facet,
                    3
                )
                
                # Classify with RAG enhancement
                classification = self.llm_client.classify_document_facet_with_rag(
                    document_content,
                    intent_analysis,
                    required_facet,
                    possible_values,
                    relevant_context,
                    taxonomy_matches
                )
                
                # Extract the best tag
                best_tag = classification.get('classification', 'Unknown')
                tag_results[required_facet] = best_tag
                
                self.logger.debug(f"Assigned {required_facet}: {best_tag} with RAG")
                
            except Exception as e:
                self.logger.error(f"Error processing facet {required_facet} with RAG: {e}")
                tag_results[required_facet] = "Unknown"
        
        # Ensure all required facets have values
        tag_results = self.tag_assigner._ensure_all_facets_populated(tag_results)
        
        self.logger.info(f"RAG-enhanced tag assignment complete for {filename}")
        return tag_results

    # Modify the process_documents method to use RAG
    def process_documents(self) -> pd.DataFrame:
        """Process all documents and assign taxonomy tags."""
        input_folder = Path(self.config['documents']['input_folder'])
        results = []
        
        # Get list of files to process
        files_to_process = self.document_processor.get_document_files(input_folder)
        
        if not files_to_process:
            self.logger.warning(f"No documents found in {input_folder}")
            return pd.DataFrame()
        
        self.logger.info(f"Processing {len(files_to_process)} documents...")
        
        for file_path in files_to_process:
            try:
                self.logger.info(f"Processing: {file_path.name}")
                
                # Extract document content
                content = self.document_processor.extract_content(file_path)
                if not content.strip():
                    self.logger.warning(f"No content extracted from {file_path.name}")
                    continue
                
                # For RAG: Create document chunks and embeddings
                if self.rag_enabled:
                    chunks = self.document_processor.chunk_document(content)
                    self.vector_store.create_document_embeddings(file_path.name, chunks)
                    
                    # Retrieve relevant context for document analysis
                    relevant_context = self.vector_store.retrieve_relevant_context(
                        content[:1000],  # Use beginning of document as query
                        self.config['rag'].get('top_k_chunks', 5)
                    )
                    
                    # Analyze document intent using RAG-enhanced LLM
                    intent_analysis = self.llm_client.analyze_document_intent_with_rag(
                        content, relevant_context
                    )
                else:
                    # Standard document analysis without RAG
                    intent_analysis = self.llm_client.analyze_document_intent(content)
                
                # Assign tags from all 13 facets using taxonomy rules and LLM analysis
                if self.rag_enabled:
                    # RAG-enhanced tag assignment
                    tag_results = self._assign_tags_with_rag(content, intent_analysis, file_path.name)
                else:
                    # Standard tag assignment
                    tag_results = self.tag_assigner.assign_tags(
                        content, 
                        intent_analysis, 
                        file_path.name
                    )
                
                # Add file information to results
                tag_results['filename'] = file_path.name
                results.append(tag_results)
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {e}")
        
        # Convert results to DataFrame
        if results:
            df = pd.DataFrame(results)
            # Reorder columns to put filename first
            cols = ['filename'] + [col for col in df.columns if col != 'filename']
            df = df[cols]
            return df
        else:
            return pd.DataFrame()
    
    def save_results(self, results_df: pd.DataFrame):
        """Save results to output file."""
        output_path = Path(self.config['output']['output_file'])
        
        if self.config['output']['table_format'] == 'csv':
            results_df.to_csv(output_path, index=False)
        elif self.config['output']['table_format'] == 'xlsx':
            results_df.to_excel(output_path, index=False)
        elif self.config['output']['table_format'] == 'json':
            results_df.to_json(output_path, orient='records', indent=2)
        
        self.logger.info(f"Results saved to: {output_path}")
    
    def run(self):
        """Main execution method."""
        try:
            self.logger.info("Starting Document Taxonomy Analyzer...")
            
            # Initialize components (taxonomy loaded only on first run)
            self.initialize_components()
            
            # Process all documents
            results = self.process_documents()
            
            if results.empty:
                self.logger.warning("No documents were successfully processed.")
                return
            
            # Save results
            self.save_results(results)
            
            self.logger.info(f"Analysis complete. Processed {len(results)} documents.")
            print(f"\nAnalysis complete! Results saved to: {self.config['output']['output_file']}")
            print(f"Processed {len(results)} documents successfully.")
            
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    analyzer = DocumentTaxonomyAnalyzer()
    analyzer.run()