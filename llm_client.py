"""
LLM Client Module

Handles communication with OpenAI's language model to analyze document intent
and assist with taxonomy classification.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from openai import OpenAI


class LLMClient:
    """Client for interacting with OpenAI's language model."""
    
    def __init__(self, config: Dict):
        """Initialize LLM client with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get API key from config or environment
        api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment variables")
        
        self.client = OpenAI(api_key=api_key)
        
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.model = config.get('model', 'gpt-5')
        self.max_tokens = config.get('max_tokens', 4000)
        self.temperature = config.get('temperature', 0.1)
    
    def analyze_document_intent(self, document_content: str) -> Dict[str, Any]:
        """Analyze the primary intent and characteristics of a document."""
        try:
            system_prompt = self._get_intent_analysis_system_prompt()
            user_prompt = self._get_intent_analysis_user_prompt(document_content)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
            else:
                result = self._get_default_intent_analysis()
            self.logger.debug("Document intent analysis completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing document intent: {e}")
            return self._get_default_intent_analysis()
    
    def classify_document_facet(self, document_content: str, intent_analysis: Dict, 
                               facet_name: str, possible_values: List[str]) -> Dict[str, Any]:
        """Classify a document for a specific facet using LLM."""
        try:
            system_prompt = self._get_classification_system_prompt(facet_name, possible_values)
            user_prompt = self._get_classification_user_prompt(document_content, intent_analysis)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
            else:
                result = {"classification": "Unknown", "confidence": 0.1, "reasoning": "No response from LLM"}
            self.logger.debug(f"Facet classification completed for {facet_name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying facet {facet_name}: {e}")
            return {"classification": "Unknown", "confidence": 0.1, "reasoning": "Classification failed"}
    
    def _get_intent_analysis_system_prompt(self) -> str:
        """Get the system prompt for document intent analysis."""
        return """You are an expert document classifier that analyzes documents to understand their primary intent and characteristics.

Your task is to analyze documents and identify key characteristics that will help with taxonomy classification. Focus on:

1. Document Purpose: What is the main purpose/intent of this document?
2. Document Type Indicators: What type of document does this appear to be?
3. Key Characteristics: What are the defining characteristics?
4. Domain/Subject Area: What domain or subject area does this relate to?
5. Lifecycle Stage: What stage of a process/project lifecycle does this represent?
6. Audience: Who is the intended audience?
7. Formality Level: How formal/official is this document?
8. Action Orientation: Does this document require action, provide information, or set rules?

Respond in JSON format with the following structure:
{
  "primary_purpose": "string",
  "document_type_indicators": ["list", "of", "indicators"],
  "key_characteristics": ["list", "of", "characteristics"],
  "domain_area": "string",
  "lifecycle_stage": "string",
  "audience": "string",
  "formality_level": "high|medium|low",
  "action_orientation": "prescriptive|informational|directive",
  "confidence": 0.8
}"""
    
    def _get_intent_analysis_user_prompt(self, document_content: str) -> str:
        """Get the user prompt for document intent analysis."""
        # Truncate content if too long
        max_content_length = 8000
        if len(document_content) > max_content_length:
            document_content = document_content[:max_content_length] + "... [content truncated]"
        
        return f"""Please analyze the following document and provide a comprehensive assessment of its intent and characteristics:

Document Content:
{document_content}

Provide your analysis in the requested JSON format."""
    
    def _get_classification_system_prompt(self, facet_name: str, possible_values: List[str]) -> str:
        """Get the system prompt for facet classification."""
        values_list = ", ".join(possible_values[:20])  # Limit to first 20 to avoid token limits
        
        return f"""You are an expert document classifier focusing on the "{facet_name}" facet.

Your task is to classify the document into one of the possible values for this facet based on the document content and intent analysis.

Possible values for {facet_name}: {values_list}

Consider:
1. The document's primary intent and purpose
2. Keywords and phrases that indicate specific classifications
3. The context and domain of the document
4. Standard patterns and conventions for this type of classification

If none of the provided values seem appropriate, you may suggest "Other" or "Unknown".

Respond in JSON format:
{{
  "classification": "selected_value",
  "confidence": 0.8,
  "reasoning": "Brief explanation of why this classification was chosen"
}}"""
    
    def _get_classification_user_prompt(self, document_content: str, intent_analysis: Dict) -> str:
        """Get the user prompt for facet classification."""
        # Truncate content if too long
        max_content_length = 6000
        if len(document_content) > max_content_length:
            document_content = document_content[:max_content_length] + "... [content truncated]"
        
        intent_summary = json.dumps(intent_analysis, indent=2)
        
        return f"""Based on the document content and intent analysis, please classify this document for the specified facet.

Document Content:
{document_content}

Intent Analysis:
{intent_summary}

Provide your classification in the requested JSON format."""
    
    def _get_default_intent_analysis(self) -> Dict[str, Any]:
        """Return default intent analysis when LLM analysis fails."""
        return {
            "primary_purpose": "Unknown",
            "document_type_indicators": [],
            "key_characteristics": [],
            "domain_area": "General",
            "lifecycle_stage": "Unknown",
            "audience": "General",
            "formality_level": "medium",
            "action_orientation": "informational",
            "confidence": 0.1
        }
    
    def enhance_rule_matching(self, document_content: str, rule_matches: List[Dict], 
                            facet_name: str) -> List[Dict]:
        """Use LLM to enhance rule-based matching with contextual understanding."""
        try:
            system_prompt = f"""You are helping to refine document classification for the "{facet_name}" facet.

You have been provided with rule-based matches and their confidence scores. Your task is to:
1. Evaluate whether these matches are contextually appropriate
2. Adjust confidence scores based on document context
3. Suggest the most appropriate classification

Consider the overall document context, not just keyword matches."""
            
            rule_summary = json.dumps([{
                "rule_id": match.get("Rule ID", ""),
                "classification": match.get("Set (value)", ""),
                "confidence": match.get("confidence", 0)
            } for match in rule_matches[:5]], indent=2)
            
            user_prompt = f"""Document excerpt (first 3000 chars):
{document_content[:3000]}

Rule-based matches:
{rule_summary}

Please provide enhanced classification with adjusted confidence scores in JSON format:
{{
  "recommended_classification": "string",
  "confidence": 0.8,
  "reasoning": "explanation"
}}"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            if content:
                enhancement = json.loads(content)
            else:
                enhancement = {"recommended_classification": "Unknown", "confidence": 0.1, "reasoning": "No enhancement available"}
            
            # Apply enhancement to rule matches
            enhanced_matches = []
            for match in rule_matches:
                enhanced_match = match.copy()
                if match.get("Set (value)") == enhancement.get("recommended_classification"):
                    enhanced_match["confidence"] = max(match.get("confidence", 0), 
                                                     enhancement.get("confidence", 0))
                enhanced_matches.append(enhanced_match)
            
            return enhanced_matches
            
        except Exception as e:
            self.logger.error(f"Error enhancing rule matching: {e}")
            return rule_matches
    
    # Add these methods to the LLMClient class
    
    def analyze_document_intent_with_rag(self, document_content: str, relevant_context: List[str]) -> Dict[str, Any]:
        """Analyze document intent with RAG enhancement using relevant context."""
        try:
            system_prompt = self._get_intent_analysis_system_prompt()
            user_prompt = self._get_rag_enhanced_intent_prompt(document_content, relevant_context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
            else:
                result = self._get_default_intent_analysis()
            self.logger.debug("RAG-enhanced document intent analysis completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing document intent with RAG: {e}")
            return self._get_default_intent_analysis()
    
    def classify_document_facet_with_rag(self, document_content: str, intent_analysis: Dict, 
                                   facet_name: str, possible_values: List[str],
                                   relevant_context: List[str],
                                   taxonomy_matches: List[str]) -> Dict[str, Any]:
        """Classify a document for a specific facet using RAG-enhanced LLM."""
        try:
            system_prompt = self._get_classification_system_prompt(facet_name, possible_values)
            user_prompt = self._get_rag_enhanced_classification_prompt(
                document_content, intent_analysis, relevant_context, taxonomy_matches
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
            else:
                result = {"classification": "Unknown", "confidence": 0.1, "reasoning": "No response from LLM"}
            self.logger.debug(f"RAG-enhanced facet classification completed for {facet_name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying facet {facet_name} with RAG: {e}")
            return {"classification": "Unknown", "confidence": 0.1, "reasoning": "Classification failed"}
    
    def _get_rag_enhanced_intent_prompt(self, document_content: str, relevant_context: List[str]) -> str:
        """Get the user prompt for RAG-enhanced document intent analysis."""
        # Prepare relevant context section
        context_section = "\n\nRELEVANT CONTEXT FROM SIMILAR DOCUMENTS:\n"
        for i, context in enumerate(relevant_context, 1):
            context_section += f"\n[Context {i}]\n{context}\n"
        
        # Create the full prompt
        prompt = f"""Analyze the following document to understand its primary intent and characteristics.
    
    DOCUMENT CONTENT:
    {document_content[:3000]}...
    
    {context_section if relevant_context else ''}
    
    Provide a JSON response with the following fields:
    - primary_intent: The main purpose of this document
    - document_type: The type of document (e.g., policy, procedure, specification)
    - target_audience: The intended audience for this document
    - key_topics: List of 3-5 main topics covered
    - formality_level: Rating from 1-5 (1=very informal, 5=highly formal)
    - technical_complexity: Rating from 1-5 (1=non-technical, 5=highly technical)
    """
        return prompt
    
    def _get_rag_enhanced_classification_prompt(self, document_content: str, intent_analysis: Dict,
                                          relevant_context: List[str], taxonomy_matches: List[str]) -> str:
        """Get the user prompt for RAG-enhanced facet classification."""
        # Prepare relevant context section
        context_section = "\n\nRELEVANT CONTEXT FROM SIMILAR DOCUMENTS:\n"
        for i, context in enumerate(relevant_context, 1):
            context_section += f"\n[Context {i}]\n{context}\n"
        
        # Prepare taxonomy matches section
        taxonomy_section = "\n\nRELEVANT TAXONOMY MATCHES:\n"
        for i, match in enumerate(taxonomy_matches, 1):
            taxonomy_section += f"\n[Match {i}] {match}"
        
        # Create the full prompt
        prompt = f"""Classify the following document based on its content and characteristics.
    
    DOCUMENT CONTENT:
    {document_content[:3000]}...
    
    DOCUMENT ANALYSIS:
    {json.dumps(intent_analysis, indent=2)}
    
    {context_section if relevant_context else ''}
    {taxonomy_section if taxonomy_matches else ''}
    
    Provide a JSON response with the following fields:
    - classification: The most appropriate classification for this document
    - confidence: A confidence score between 0.0 and 1.0
    - reasoning: Your reasoning for this classification
    """
        return prompt